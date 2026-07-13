"""
DL-PPI 模型（适配 OpenDrug Pipeline）

原版架构: CNN(Conv1d+GRU) 序列特征提取 + GIN 图卷积 + NTN/拼接交互融合
适配: 使用 OpenDrug 预训练嵌入作为节点特征，其余流程与原版一致

支持 PPI 二分类和多标签分类。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.branch1x3_1 = nn.Conv1d(in_channels, 3, kernel_size=3)
        self.branch1x3_2 = nn.Conv1d(3, 1, kernel_size=1)
        self.branch1x5_1 = nn.Conv1d(in_channels, 5, kernel_size=5, padding=0)
        self.branch1x5_2 = nn.Conv1d(5, 1, kernel_size=1)
        self.branch_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x3 = self.branch1x3_1(x)
        branch1x3 = self.branch1x3_2(branch1x3)
        branch1x5 = self.branch1x5_1(x)
        branch1x5 = self.branch1x5_2(branch1x5)
        branch_pool = F.avg_pool1d(x, 1, 1, 0)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch_pool, branch1x1, branch1x3, branch1x5]
        return torch.cat(outputs, dim=0)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.matmul(x.T, x)
        b = F.softmax(a, dim=0)
        c = torch.matmul(x, b)
        return c


class TenorNetworkModule(nn.Module):
    """
    SimGNN-style Tensor Network module.
    NTN 硬编码的 hidden=16 对 gin_hidden=512 不适用，
    故简化为基于元素乘积+MLP 的交互建模。
    """

    def __init__(self, hidden=256):
        super().__init__()
        self.hidden = hidden
        self.atten = Attention()
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, embedding_1, embedding_2):
        """
        embedding_1, embedding_2: [batch, hidden]
        Returns: [batch, hidden]
        """
        embedding_1 = self.atten(embedding_1)
        embedding_2 = self.atten(embedding_2)
        combined = torch.cat([embedding_1, embedding_2], dim=1)
        scores = self.fc(combined)
        return scores


class GIN_Net2(nn.Module):
    """
    DL-PPI 核心模型（适配 OpenDrug）

    适配说明:
    - 原版: 每次处理一个蛋白质，CNN 把 [seq_len, in_feature] → 标量 → 投影
    - 适配: 批量处理多个蛋白质，先对嵌入做池化压缩维度，再过 CNN
      - protein_dim 嵌入 → max pooling → [1, 1] → reshape → [1, 1]
      - 即把嵌入视为 "一个 token"，CNN 退化为单层 MLP

    流程:
    1. 嵌入压缩: MaxPool1d(protein_dim, dim=1) → [B, 1, 1]
    2. 特征变换: Conv1d(1, 1, kernel=1) → [B, 1, 1]（等价于缩放）
    3. 池化: AdaptiveAvgPool1d(1) → [B, 1, 1]
    4. 投影: fc1([B, 1] → [B, gin_in_feature])
    5. GIN 图卷积
    6. NTN/Concat/Mult 交互融合
    7. 分类 MLP
    """

    def __init__(self, in_feature=13, gin_in_feature=256, hidden=512, num_layers=1,
                 pool_size=3, num_classes=7, feature_fusion='NTN', dropout=0.5):
        super().__init__()
        self.feature_fusion = feature_fusion
        self.num_classes = num_classes

        self.compress_conv = nn.Conv1d(1, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(1)
        self.biGRU = nn.GRU(1, 1, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(1, gin_in_feature)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=True
        )
        self.gin_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gin_convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                ), train_eps=True
            ))

        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden * 2, hidden)   # concat 专用
        self.fc2 = nn.Linear(hidden, num_classes)   # 多标签: hidden -> num_classes

        if feature_fusion == 'NTN':
            self.ntn = TenorNetworkModule(hidden=hidden)

    def forward(self, x, edge_index, edge_batch_idx, dropout=0.5):
        """
        Args:
            x: [N_proteins, seq_len, 1] 蛋白质嵌入（由数据集 reshape）
            edge_index: [2, num_edges] 图边索引
            edge_batch_idx: [batch_size] 需要预测的边索引
        Returns:
            output: [batch_size, num_classes]
        """
        # x: [N, protein_dim, 1]
        # CNN 部分：直接 AdaptiveAvgPool 压缩 seq 维度到 1，再投影
        x = x.transpose(1, 2)                   # [N, seq_len, 1] -> [N, 1, seq_len]
        x = self.compress_conv(x)             # [N, 1, seq_len] -> [N, 1, seq_len]
        x = self.bn1(x)
        x = self.global_avgpool1d(x)          # [N, 1, seq_len] -> [N, 1, 1]
        x = x.squeeze(-1)                     # [N, 1]
        x = self.fc1(x)                      # [N, 1] -> [N, gin_in_feature]

        x = self.gin_conv1(x, edge_index)
        xs = [x]
        for conv in self.gin_convs:
            x = conv(x, edge_index)
            xs.append(x)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)

        node_id = edge_index[:, edge_batch_idx]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]

        if self.feature_fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)   # [B, 2*hidden]
            x = self.lin3(x)                  # [B, 2*hidden] -> [B, hidden]
        elif self.feature_fusion == 'NTN':
            x = self.ntn(x1, x2)              # [B, hidden]
        else:
            x = torch.mul(x1, x2)            # [B, hidden]

        # 二分类和多标签统一输出 [B, num_classes]
        x = self.fc2(x)                    # Linear(hidden, num_classes) -> [B, num_classes]
        return x


def DL_PPI_Model(in_feature=13, gin_in_feature=256, hidden=512, num_layers=1,
                 pool_size=3, num_classes=7, feature_fusion='NTN', dropout=0.5, **kwargs):
    """
    DL-PPI 模型工厂函数（用于 model_manager）
    """
    return GIN_Net2(
        in_feature=in_feature,
        gin_in_feature=gin_in_feature,
        hidden=hidden,
        num_layers=num_layers,
        pool_size=pool_size,
        num_classes=num_classes,
        feature_fusion=feature_fusion,
        dropout=dropout,
    )
