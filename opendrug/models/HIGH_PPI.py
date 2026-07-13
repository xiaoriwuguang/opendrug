"""
HIGH-PPI (Hierarchical Interaction Graph for Protein-Protein Interaction) 模型

基于 D-SCRIPT (Scalable Protein-Protein Interaction Prediction from Protein Sequence)
的核心设计思路，针对 OpenDrug 的蛋白质级嵌入输入进行适配。

D-SCRIPT 原始设计:
1. FullyConnectedEmbed: 将每个氨基酸的 LM 嵌入投影到低维空间
2. ContactCNN: 通过差值/乘积特征预测残基-残基接触图
3. ModelInteraction: 对接触图进行加权池化得到相互作用概率

OpenDrug 适配:
- 输入为蛋白质级嵌入 (protein_x: [N, protein_dim])，而非每个残基的嵌入
- 用配对蛋白质的嵌入向量构建"交互矩阵"：
  - z_dif = |emb1 - emb2|  (差异特征)
  - z_mul = emb1 * emb2  (乘积特征)
- 2D CNN 对交互矩阵进行空间建模
- 全局池化 + MLP 分类头

支持:
- PPI 二分类 (CrossEntropyLoss)
- PPI 多标签分类 (BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionEncoder(nn.Module):
    """
    交互编码模块

    对两个蛋白质的嵌入构建广播式交互特征张量，
    灵感来自 D-SCRIPT 的 ContactCNN 中的差值/乘积特征构建方式。

    对于蛋白质级嵌入:
    - p1_emb: [B, protein_dim]
    - p2_emb: [B, protein_dim]
    - 构建: [B, protein_dim, 2] 其中 [:,:,0]=差值, [:,:,1]=乘积
    """

    def __init__(self, protein_dim, hidden_dim):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim

        self.fc_dif = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_mul = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, p1_emb, p2_emb):
        """
        Args:
            p1_emb: [B, protein_dim]
            p2_emb: [B, protein_dim]
        Returns:
            combined: [B, 2, hidden_dim]
        """
        dif = torch.abs(p1_emb - p2_emb)
        mul = p1_emb * p2_emb

        dif_feat = self.fc_dif(dif)
        mul_feat = self.fc_mul(mul)

        combined = torch.stack([dif_feat, mul_feat], dim=1)
        return combined


class ContactMapPredictor(nn.Module):
    """
    接触图预测模块 (适配蛋白质级嵌入版本)

    原始 D-SCRIPT ContactCNN 在残基对上应用 2D 卷积。
    这里对配对蛋白质的交互特征应用 1D 卷积来建模"隐式接触图"。

    原理:
    - 将 [B, 2, hidden_dim] 视为"宽度=2"的"接触图"
    - 用 1D 卷积 + 全局池化来聚合交互信号
    """

    def __init__(self, hidden_dim, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [B, 2, hidden_dim]
        Returns:
            pooled: [B, hidden_dim // 2]
        """
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        return x


class WeightedPooling(nn.Module):
    """
    加权池化模块 (灵感来自 D-SCRIPT ModelInteraction)

    D-SCRIPT 使用可学习的 Gaussian 衰减权重矩阵 + 全局均值池化。
    这里简化为: 可学习的序列位置编码 + 加权聚合。
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor([0.5]))
        self.gamma = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x):
        """
        Args:
            x: [B, hidden_dim]
        Returns:
            pooled: [B, hidden_dim]
        """
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        theta = torch.sigmoid(self.theta)
        pooled = theta * x + (1 - theta) * x_norm
        return pooled


class HIGH_PPI(nn.Module):
    """
    HIGH-PPI 模型

    核心架构 (基于 D-SCRIPT 设计思路):
    1. InteractionEncoder: 构建蛋白质对的差值/乘积交互特征
    2. ContactMapPredictor: 1D 卷积建模隐式接触图信号
    3. WeightedPooling: 可学习的加权池化
    4. MLP 分类头: 输出二分类或多标签分类 logits

    与 D-SCRIPT 的区别:
    - D-SCRIPT: 每个蛋白质有多个残基嵌入 -> 预测 N×M 接触图
    - HIGH-PPI: 每个蛋白质有单个嵌入向量 -> 预测隐式交互分数

    Args:
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 模型内部隐藏维度 (默认 256)
        dropout: Dropout 概率
        num_classes: 类别数 (二分类=2, 多标签=标签数)
        task_type: 'binary' 或 'multilabel'
    """

    def __init__(self, protein_dim=1024, hidden_dim=256, dropout=0.3,
                 num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.interaction_encoder = InteractionEncoder(protein_dim, hidden_dim)

        self.contact_predictor = ContactMapPredictor(hidden_dim, num_classes)

        self.weighted_pool = WeightedPooling(hidden_dim // 2)

        if task_type == 'multilabel':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象，包含 protein_x
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

        p1_emb = protein_x[p1_idx]
        p2_emb = protein_x[p2_idx]

        combined = self.interaction_encoder(p1_emb, p2_emb)

        interaction_feat = self.contact_predictor(combined)

        pooled_feat = self.weighted_pool(interaction_feat)

        output = self.classifier(pooled_feat)
        return output


def HIGH_PPI_Model(protein_dim=1024, hidden_dim=256, dropout=0.3,
                   num_classes=2, task_type='binary', **kwargs):
    """
    HIGH_PPI 模型工厂函数（用于 model_manager）
    """
    return HIGH_PPI(
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
