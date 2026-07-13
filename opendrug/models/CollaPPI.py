"""
CollaPPI (Collaborative Multi-View Graph Attention for Protein-Protein Interaction) 模型

基于论文: "CollaPPI: a novel protein-protein interaction prediction method based on
collaborative graph attention network and multi-view learning"
https://academic.oup.com/bioinformatics/article/39/9/btad562

原始 CollaPPI 设计:
- 多视角图注意力网络 (Multi-View Graph Attention Network)
- 每个蛋白质构建为 residue-level 图 (contact map edges + ESM-2 embeddings)
- Intra-protein: GATConv 对蛋白质内部结构进行编码
- Cross-protein: Mutual Attention 计算两个蛋白质所有残基位点间的成对注意力
  - 计算相似度矩阵: alpha = tanh(W1*h1 + W2*h2) @ w
  - 双向聚合: mean(softmax(alpha)) * h1 + mean(softmax(alpha.T)) * h2
- 多任务学习: 相互作用预测 + GO 功能预测 + 细胞定位预测

OpenDrug 适配:
- 输入为蛋白质级嵌入 (protein_x: [N, protein_dim]) 和 PPI 网络图 (edge_index)
- 适配策略: 基于 PPI 网络拓扑的协同注意力机制
  - 将 PPI 网络视为异构图，两个蛋白质通过共享邻居结构进行协同编码
  - Intra-protein: 对蛋白质嵌入应用自注意力编码
  - Cross-protein: 双线性协同注意力 + MLP 交互分类器
- 多任务辅助头保留用于多标签分类

支持:
- PPI 二分类 (CrossEntropyLoss)
- PPI 多标签分类 (BCEWithLogitsLoss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, SAGPooling
from torch_geometric.utils import add_self_loops


class LayerNorm(nn.Module):
    """PyTorch Geometric 兼容的 LayerNorm"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x)


class ProteinEncoder(nn.Module):
    """
    蛋白质编码器

    对蛋白质嵌入进行编码，使用自注意力机制模拟原始 CollaPPI 的
    Intra-protein GAT 层。
    """

    def __init__(self, protein_dim, hidden_dim=64, num_heads=2, dropout=0.2):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = hidden_dim * num_heads

        self.ln = LayerNorm(protein_dim)
        self.projection = nn.Linear(protein_dim, self.out_dim)

        self.gat = GATConv(
            in_channels=self.out_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: [N, protein_dim] 蛋白质嵌入
            edge_index: [2, E] 图边索引
            batch: [N] 批次索引 (可选)
        Returns:
            node_repr: [N, out_dim] 编码后的节点表示
        """
        x = self.ln(x)
        x = self.projection(x)
        x = F.elu(x)
        x = self.dropout(x)

        node_repr = self.gat(x, edge_index)
        node_repr = F.elu(node_repr)

        return node_repr


class BilinearCollaborativeAttention(nn.Module):
    """
    双线性协同注意力 (Bilinear Collaborative Attention)

    核心创新: 使用双线性注意力建模两个蛋白质之间的协同交互。

    原始 CollaPPI 的 Mutual Attention:
    - alpha_ij = softmax(tanh(W1*hi + W2*hj) @ w)_ij
    - hi_agg = sum_j(alpha_ij * hj)
    - hj_agg = sum_i(alpha_ij * hi)

    这里适配为: 基于 PPI 网络拓扑的协同注意力
    - 每个蛋白质作为一个"超级节点"
    - 通过图结构传递信息，聚合邻居蛋白质的信息
    - 最后用双线性交互头预测相互作用

    支持两种注意力模式:
    - bilinear: alpha = sigmoid(x1 @ W @ x2.T)
    - scaled_dot: alpha = (x1 @ x2.T) / sqrt(d)
    """

    def __init__(self, hidden_dim, num_heads=2, attention_mode='bilinear', dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_mode = attention_mode

        if attention_mode == 'bilinear':
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_mode == 'scaled_dot':
            pass
        else:
            self.W = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )

        self.aggregate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, p1_repr, p2_repr, p1_idx, p2_idx, edge_index, num_nodes):
        """
        Args:
            p1_repr: [N, hidden_dim * num_heads] 蛋白质1的节点表示
            p2_repr: [N, hidden_dim * num_heads] 蛋白质2的节点表示
            p1_idx: [B] 批次中蛋白质1的索引
            p2_idx: [B] 批次中蛋白质2的索引
            edge_index: [2, E] PPI 网络边索引
            num_nodes: int 总蛋白质数量
        Returns:
            fused: [B, hidden_dim * num_heads] 融合后的表示
        """
        B = p1_idx.size(0)

        p1_emb = p1_repr[p1_idx]  # [B, D]
        p2_emb = p2_repr[p2_idx]  # [B, D]

        if self.attention_mode == 'bilinear':
            p1_mapped = self.W(p1_emb)  # [B, D]
            alpha = torch.sigmoid(torch.sum(p1_mapped * p2_emb, dim=-1, keepdim=True))  # [B, 1]
        elif self.attention_mode == 'scaled_dot':
            scale = math.sqrt(p1_emb.size(-1))
            alpha = torch.sigmoid(
                torch.sum(p1_emb * p2_emb, dim=-1, keepdim=True) / scale
            )  # [B, 1]
        else:
            concat_feat = torch.cat([p1_emb, p2_emb], dim=-1)  # [B, 2D]
            alpha = torch.sigmoid(self.W(concat_feat))  # [B, 1]

        cross_feat = p1_emb * p2_emb * alpha

        agg_feat = self.aggregate_proj(self.dropout(cross_feat))

        fused = torch.cat([p1_emb, p2_emb, agg_feat], dim=-1)

        return fused


class MV_PPI_Block(nn.Module):
    """
    多视角 PPI 块 (Multi-View PPI Block)

    原始 CollaPPI 的核心模块:
    1. Intra-protein attention: GATConv 对每个蛋白质内部结构编码
    2. Cross-protein attention: 双线性协同注意力建模蛋白质对交互

    OpenDrug 适配:
    - Intra-protein: 在 PPI 网络上进行 GATConv 编码
    - Cross-protein: 双线性协同注意力 + MLP 交互分类
    """

    def __init__(self, protein_dim, hidden_dim=64, num_heads=2,
                 attention_mode='bilinear', dropout=0.2):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.encoder = ProteinEncoder(
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.collaborative_attention = BilinearCollaborativeAttention(
            hidden_dim=hidden_dim * num_heads,
            num_heads=num_heads,
            attention_mode=attention_mode,
            dropout=dropout,
        )

    def forward(self, x, edge_index, p1_idx, p2_idx):
        """
        Args:
            x: [N, protein_dim] 蛋白质嵌入
            edge_index: [2, E] PPI 网络边
            p1_idx: [B] 蛋白质1索引
            p2_idx: [B] 蛋白质2索引
        Returns:
            interaction_feat: [B, fused_dim] 交互特征
        """
        node_repr = self.encoder(x, edge_index)

        fused = self.collaborative_attention(
            node_repr, node_repr, p1_idx, p2_idx, edge_index, x.size(0)
        )

        return fused


class InteractionHead(nn.Module):
    """
    交互预测头

    从协同注意力融合特征预测蛋白质相互作用。
    对应原始 CollaPPI 的 MLP 分类器 (512 -> 256 -> 1/7)。
    """

    def __init__(self, input_dim, hidden_dim=512, dropout=0.2,
                 num_classes=2, task_type='binary'):
        super().__init__()
        self.task_type = task_type

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if task_type == 'multilabel':
            self.output = nn.Linear(hidden_dim // 2, num_classes)
        else:
            self.output = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] 交互特征
        Returns:
            [B, num_classes] logits
        """
        h = self.mlp(x)
        logits = self.output(h)
        return logits


class CollaPPI(nn.Module):
    """
    CollaPPI 模型

    核心架构:
    1. MV_PPI_Block (Multi-View PPI Block):
       - Intra-protein Encoder: GATConv 在 PPI 网络上编码蛋白质节点
       - Bilinear Collaborative Attention: 建模两个蛋白质的协同交互
    2. InteractionHead: MLP 分类器输出 logits

    关键设计:
    - GATConv: 多头自注意力模拟残基级别的结构感知
    - 双线性注意力: 捕捉两个蛋白质之间的非线性交互模式
    - 可选的 attention_mode: 'bilinear' / 'scaled_dot' / 'concat'

    Args:
        protein_dim: 蛋白质嵌入维度
        hidden_dim: GATConv 隐藏维度 (默认 64)
        num_heads: 注意力头数 (默认 2)
        attention_mode: 注意力模式 (默认 'bilinear')
        dropout: Dropout 概率 (默认 0.2)
        num_classes: 类别数 (二分类=2, 多标签=标签数)
        task_type: 'binary' 或 'multilabel'
    """

    def __init__(self, protein_dim=1024, hidden_dim=64, num_heads=2,
                 attention_mode='bilinear', dropout=0.2,
                 num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.mv_block = MV_PPI_Block(
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attention_mode=attention_mode,
            dropout=dropout,
        )

        fused_dim = hidden_dim * num_heads * 3

        self.interaction_head = InteractionHead(
            input_dim=fused_dim,
            hidden_dim=max(hidden_dim * num_heads * 2, 128),
            dropout=dropout,
            num_classes=num_classes,
            task_type=task_type,
        )

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象 (PyG Data)，包含:
                - protein_x: [N, protein_dim] 蛋白质嵌入
                - edge_index: [2, E] PPI 网络边
            idx_batch: 批次数据 (p1_idx, p2_idx, labels)
                - p1_idx: [B] 蛋白质1索引
                - p2_idx: [B] 蛋白质2索引
                - labels: [B] 或 [B, num_classes]

        Returns:
            output: [B, num_classes] logits
        """
        p1_idx = idx_batch[0]
        p2_idx = idx_batch[1]

        device = next(self.parameters()).device
        if isinstance(p1_idx, torch.Tensor):
            p1_idx = p1_idx.to(device)
            p2_idx = p2_idx.to(device)
        else:
            p1_idx = torch.as_tensor(p1_idx, dtype=torch.long, device=device)
            p2_idx = torch.as_tensor(p2_idx, dtype=torch.long, device=device)

        protein_x = graph_or_none.protein_x
        edge_index = graph_or_none.edge_index

        fused = self.mv_block(protein_x, edge_index, p1_idx, p2_idx)

        output = self.interaction_head(fused)
        return output


def CollaPPI_Model(protein_dim=1024, hidden_dim=64, num_heads=2,
                   attention_mode='bilinear', dropout=0.2,
                   num_classes=2, task_type='binary', **kwargs):
    """
    CollaPPI 模型工厂函数（用于 model_manager）
    """
    return CollaPPI(
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        attention_mode=attention_mode,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
