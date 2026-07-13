"""
MAPE-PPI (Microenvironment-Aware Protein Embedding for PPI) 模型

基于 Graph Isomorphism Network (GIN) 的 PPI 预测模型，适配 OpenDrug 预计算蛋白质嵌入输入。

架构设计参考自 baseline/ppi/MAPE-PPI/src/models.py 中的 GIN 类：
1. 蛋白质嵌入投影：protein_dim -> ppi_hidden_dim
2. 2 层 GINConv（图同构卷积）在 PPI 网络上进行消息传递
3. 节点嵌入提取：根据 ppi_list 提取配对蛋白质嵌入
4. 交互建模：mul(p1_emb, p2_emb) 元素乘积
5. MLP 分类头

支持：
- PPI 二分类
- PPI 多标签分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GINConvBlock(nn.Module):
    """
    单层 GIN（Graph Isomorphism Network）卷积块

    核心思想：GIN 通过 MLP 逼近 1-Weisfeiler-Lehman 测试，
    能够在图同构意义上区分不同结构，比普通 GCN/GAT 更强。

    MLP(x) = Linear(Linear(x) + (1+eps)*neighbor_aggregate)

    Args:
        in_channels: 输入特征维度
        out_channels: 输出特征维度
        train_eps: 是否学习 epsilon（可学习的自环权重）
    """

    def __init__(self, in_channels, out_channels, train_eps=True):
        super().__init__()
        self.train_eps = train_eps
        self.eps = nn.Parameter(torch.zeros(1)) if train_eps else 0.0

        self.gin_conv = GINConv(
            nn=nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
            ),
            eps=0.0,
            train_eps=train_eps,
        )

    def forward(self, x, edge_index):
        return self.gin_conv(x, edge_index)


class MAPE_PPI(nn.Module):
    """
    MAPE-PPI 模型

    基于 GIN（图同构网络）的 PPI 预测模型，核心思想：
    - 在 PPI 图上通过 GIN 层进行节点嵌入学习
    - GIN 通过多层感知机逼近 1-WL 图同构测试，具有强大的图结构表达能力
    - 通过元素乘积建模蛋白质对的交互特征

    架构（参考 MAPE-PPI/src/models.py GIN 类）：
    1. 蛋白质嵌入投影：protein_x [N, protein_dim] -> [N, ppi_hidden_dim]
    2. GIN 层 1：ppi_hidden_dim -> ppi_hidden_dim
    3. GIN 层 2：ppi_hidden_dim -> ppi_hidden_dim
    4. Dropout(0.5)
    5. 配对嵌入提取 + 元素乘积：mul(emb1, emb2)
    6. 线性层：ppi_hidden_dim -> ppi_hidden_dim -> ReLU
    7. 分类头：ppi_hidden_dim -> num_classes

    Args:
        protein_dim: 蛋白质嵌入维度
        ppi_hidden_dim: PPI 编码器隐藏维度（默认 512）
        dropout: Dropout 概率
        num_classes: 类别数
        task_type: 'binary' 或 'multilabel'
    """

    def __init__(self, protein_dim=1024, ppi_hidden_dim=512, dropout=0.2,
                 num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.ppi_hidden_dim = ppi_hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        # 蛋白质嵌入投影：protein_dim -> ppi_hidden_dim
        # 对应原始 GIN 第一层：Linear(prot_hidden_dim * 2, ppi_hidden_dim)
        # *2 是因为 GIN 中边信息也被concat进来；这里简化为只投影节点特征
        self.proj = nn.Sequential(
            nn.Linear(protein_dim, ppi_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(ppi_hidden_dim),
        )

        # GIN 层 1：输入维度 = ppi_hidden_dim（第一层将 proj 输出翻倍以匹配 concat 后的维度）
        # 原始代码: Linear(prot_hidden_dim * 2, ppi_hidden_dim)
        # 这里简化为: Linear(ppi_hidden_dim, ppi_hidden_dim)
        self.gin1 = GINConvBlock(ppi_hidden_dim, ppi_hidden_dim, train_eps=True)

        # GIN 层 2
        self.gin2 = GINConvBlock(ppi_hidden_dim, ppi_hidden_dim, train_eps=True)

        self.dropout = nn.Dropout(0.5)

        # 交互层
        self.interaction_fc = nn.Sequential(
            nn.Linear(ppi_hidden_dim, ppi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 分类头
        if task_type == 'multilabel':
            self.classifier = nn.Linear(ppi_hidden_dim, num_classes)
        else:
            self.classifier = nn.Linear(ppi_hidden_dim, num_classes)

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象，包含 protein_x 和 edge_index
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

        # Step 1: 蛋白质嵌入投影
        x = self.proj(protein_x)  # [N, protein_dim] -> [N, ppi_hidden_dim]

        # Step 2: GIN 层消息传递
        x = self.gin1(x, edge_index)
        x = self.dropout(x)
        x = self.gin2(x, edge_index)

        if self.training:
            x = F.dropout(F.relu(self.interaction_fc(x)), p=0.5, training=True)
        else:
            x = F.dropout(F.relu(self.interaction_fc(x)), p=0.5, training=False)

        # Step 3: 提取配对蛋白质嵌入
        p1_emb = x[p1_idx]   # [B, ppi_hidden_dim]
        p2_emb = x[p2_idx]    # [B, ppi_hidden_dim]

        # Step 4: 交互建模（元素乘积）
        interaction = p1_emb * p2_emb

        # Step 5: 分类
        output = self.classifier(interaction)
        return output


def MAPE_PPI_Model(protein_dim=1024, hidden_dim=512, dropout=0.2,
                    num_classes=2, task_type='binary', **kwargs):
    """
    MAPE_PPI 模型工厂函数（用于 model_manager）
    hidden_dim -> ppi_hidden_dim
    """
    return MAPE_PPI(
        protein_dim=protein_dim,
        ppi_hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
