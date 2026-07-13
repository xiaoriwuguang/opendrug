"""
TAGPPI (Topology-Aware Graph and Protein PPI) 模型

适配 OpenDrug Pipeline:
- 使用 OpenDrug 蛋白质嵌入作为节点特征 (代替原版 contact-map subgraph)
- TextCNN 分支: 将 1D 嵌入 reshape 为 [protein_dim, 1] 的伪序列，
  用 Conv1d 提取局部模式（跨嵌入维度的模式）
- GAT 分支: 在 PPI 网络上进行图注意力卷积 (PyG GATConv)
- 融合: 可学习的加权融合 + MLP 分类头

支持 PPI 二分类和多标签分类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TextCNNLayer(nn.Module):
    """
    TextCNN: 将蛋白质嵌入视为 "伪序列"
    输入: [B, seq_len, 1]  (seq_len = output_dim，feature_proj 压缩后的维度)
    Conv1d: in=1, out=out_channels, kernel=3 -> 输出 [B, out_channels, seq_len]
    平均池化: 沿最后一维 -> [B, out_channels, 1] -> squeeze -> [B, out_channels]
    """

    def __init__(self, seq_len, out_channels=128):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # x: [B, seq_len, 1]
        x = x.permute(0, 2, 1)          # -> [B, 1, seq_len]
        x = self.conv(x)                # -> [B, out_channels, seq_len]
        x = F.relu(x)
        x = x.mean(dim=-1, keepdim=True) # -> [B, out_channels, 1]
        x = x.squeeze(-1)               # -> [B, out_channels]
        return x


class TAGPPI(nn.Module):
    """
    TAGPPI 模型

    架构:
    1. TextCNN 分支: protein_emb [num_proteins, protein_dim]
                       reshape -> [num_proteins, protein_dim, 1]
                       TextCNN -> [num_proteins, output_dim]
    2. GAT 分支: PPI 图上做 3 层 GAT (PyG) -> 每个节点的上下文感知嵌入 [num_proteins, output_dim]
    3. 加权融合: w * gnn + (1-w) * seq  (每对蛋白质)
    4. MLP 分类: [B, output_dim*2] -> [B, max(output_dim*2,256)] -> [B, max(output_dim,128)] -> [B, num_classes]

    Args:
        protein_dim: 蛋白质嵌入维度
        output_dim: TextCNN 输出维度 / GAT 输出维度 (默认 128)
        dropout: Dropout 概率
        num_classes: 类别数 (二分类=2, 多标签=7)
        task_type: 'binary' 或 'multilabel'
    """

    def __init__(self, protein_dim=1024, output_dim=128, dropout=0.5,
                 num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.output_dim = min(output_dim, 128)   # 显存安全上限
        self.num_classes = num_classes
        self.task_type = task_type

        # TextCNN 分支（输入为 feature_proj 的输出，序列长度为 output_dim）
        self.textcnn = TextCNNLayer(output_dim, out_channels=output_dim)
        self.text_proj = nn.Linear(output_dim, output_dim)

        # 特征压缩投影层（参照 DL_PPI 策略）
        # 将高维 protein_dim 压缩到 output_dim，显著降低 GAT 内存占用
        self.feature_proj = nn.Sequential(
            nn.Linear(protein_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # GAT 分支 (3 层 GAT，PyG GATConv)
        # 压缩后输入/输出均为 output_dim，内存开销大幅降低
        # Layer1: output_dim -> output_dim (2 heads, concat → 2*output_dim)
        self.gat1 = GATConv(output_dim, output_dim, heads=2, concat=True)
        # Layer2: output_dim*2 -> output_dim (2 heads, concat → 2*output_dim)
        self.gat2 = GATConv(output_dim * 2, output_dim, heads=2, concat=True)
        # Layer3: output_dim*2 -> output_dim (1 head, no concat)
        self.gat3 = GATConv(output_dim * 2, output_dim, heads=1, concat=False)

        # 可学习的加权融合
        self.fusion_w = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        # MLP 分类头（隐藏层维度与 output_dim 成比例，避免 output_dim=512 时显存爆炸）
        mlp_hidden = max(output_dim * 2, 256)
        mlp_mid = max(output_dim, 128)
        self.fc1 = nn.Linear(output_dim * 2, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_mid)
        self.fc_out = nn.Linear(mlp_mid, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward_gnn(self, x, edge_index):
        """
        对 PPI 图做 3 层 GAT，返回每个节点的上下文感知嵌入

        Args:
            x: [num_proteins, protein_dim] 节点特征（原始高维嵌入）
            edge_index: [2, num_edges] 边索引
        Returns:
            h: [num_proteins, output_dim]
        """
        # 特征压缩：protein_dim -> output_dim，大幅降低后续 GAT 内存占用
        x = self.feature_proj(x)          # [N, protein_dim] -> [N, output_dim]
        return self._forward_gnn_from_proj(x, edge_index)

    def _forward_gnn_from_proj(self, x, edge_index):
        """
        3 层 GAT 前向传播（x 已经是 output_dim 维）

        Args:
            x: [num_proteins, output_dim] 节点特征（已压缩）
            edge_index: [2, num_edges] 边索引
        Returns:
            h: [num_proteins, output_dim]
        """
        h = self.gat1(x, edge_index)     # [N, output_dim*2]
        h = F.relu(h)
        h = self.gat2(h, edge_index)    # [N, output_dim*2]
        h = F.relu(h)
        h = self.gat3(h, edge_index)    # [N, output_dim]
        return h

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: PyG Data 对象，包含:
                - protein_x: [num_proteins, protein_dim] 蛋白质嵌入
                - edge_index: [2, num_edges] PPI 网络边
            idx_batch: tuple (p1_idx, p2_idx, labels)
                - p1_idx: [B] 蛋白质1索引
                - p2_idx: [B] 蛋白质2索引
        Returns:
            output: [B, num_classes]
        """
        p1_idx, p2_idx = idx_batch[0], idx_batch[1]

        if isinstance(p1_idx, torch.Tensor):
            p1_idx = p1_idx.to(next(self.parameters()).device)
        else:
            p1_idx = torch.as_tensor(p1_idx, dtype=torch.long,
                                     device=next(self.parameters()).device)
        if isinstance(p2_idx, torch.Tensor):
            p2_idx = p2_idx.to(next(self.parameters()).device)
        else:
            p2_idx = torch.as_tensor(p2_idx, dtype=torch.long,
                                     device=next(self.parameters()).device)

        protein_x = graph_or_none.protein_x
        edge_index = graph_or_none.edge_index

        # 只计算一次 feature_proj，供 GNN 和 TextCNN 共用
        protein_proj = self.feature_proj(protein_x)     # [N, protein_dim] -> [N, output_dim]

        # --- GNN: 每个节点的上下文感知嵌入 ---
        gnn_emb = self._forward_gnn_from_proj(protein_proj, edge_index)  # [N, output_dim]

        # --- TextCNN ---
        seq_emb = protein_proj.unsqueeze(-1)              # [N, output_dim, 1]
        seq_out = self.textcnn(seq_emb)                   # [N, output_dim]
        seq_emb_proj = self.text_proj(seq_out)            # [N, output_dim]

        # --- 取当前 batch 的蛋白质表示 ---
        p1_seq = seq_emb_proj[p1_idx]                       # [B, output_dim]
        p2_seq = seq_emb_proj[p2_idx]
        p1_gnn = gnn_emb[p1_idx]
        p2_gnn = gnn_emb[p2_idx]

        # --- 加权融合 ---
        w = torch.sigmoid(self.fusion_w)
        gc1 = w * p1_gnn + (1 - w) * p1_seq
        gc2 = w * p2_gnn + (1 - w) * p2_seq

        # --- 拼接 + MLP ---
        gc = torch.cat([gc1, gc2], dim=1)                   # [B, output_dim*2]
        gc = F.relu(self.fc1(gc))
        gc = self.dropout(gc)
        gc = F.relu(self.fc2(gc))
        gc = self.dropout(gc)
        out = self.fc_out(gc)                              # [B, num_classes]

        return out


def TAGPPI_Model(protein_dim=1024, hidden_dim=256, dropout=0.5,
                 num_classes=2, task_type='binary', **kwargs):
    """
    TAGPPI 模型工厂函数（用于 model_manager）
    hidden_dim -> output_dim
    """
    return TAGPPI(
        protein_dim=protein_dim,
        output_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
