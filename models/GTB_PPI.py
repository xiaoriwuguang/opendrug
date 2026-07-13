"""
GTB-PPI (Gradient Tree Boosting for Protein-Protein Interaction) 模型

基于论文: "Prediction of Protein-Protein Interactions Based on L1-Regularized
Logistic Regression and Gradient Tree Boosting"

原始 GTB-PPI 设计:
- 使用 sklearn GradientBoostingClassifier
- 特征: PseAAC, PsePSSM, RSIV, AD 等手工特征
- L1 正则化逻辑回归进行特征选择

OpenDrug 适配:
- 输入为蛋白质级嵌入 (protein_x: [N, protein_dim])
- 用配对蛋白质的嵌入构建交互特征:
  - concat(emb1, emb2)          (连接特征)
  - |emb1 - emb2|               (差异特征)
  - emb1 * emb2                 (乘积特征)
- 多层残差连接 (Residual Connection) 模拟梯度提升的累加效应
- 模拟 GTB 的"分步学习"思想：逐层细化交互特征

支持:
- PPI 二分类 (CrossEntropyLoss)
- PPI 多标签分类 (BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    残差块，模拟 GTB 的分步学习

    每个残差块建模: output = F(x) + x
    其中 F(x) 是一个非线性变换，与 GTB 中每轮对残差的拟合相对应。
    """

    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class FeatureInteractionModule(nn.Module):
    """
    特征交互模块

    将两个蛋白质的嵌入构建为丰富的交互特征向量。
    灵感来自 GTB-PPI 中将多个特征源拼接的思想。

    特征构成:
    - concat: [protein_dim * 2]
    - abs_diff: [protein_dim]
    - product: [protein_dim]
    总输出维度: protein_dim * 4
    """

    def __init__(self, protein_dim):
        super().__init__()
        self.protein_dim = protein_dim
        self.output_dim = protein_dim * 4

    def forward(self, p1_emb, p2_emb):
        """
        Args:
            p1_emb: [B, protein_dim]
            p2_emb: [B, protein_dim]
        Returns:
            interaction_feat: [B, protein_dim * 4]
        """
        concat_feat = torch.cat([p1_emb, p2_emb], dim=1)
        abs_diff_feat = torch.abs(p1_emb - p2_emb)
        product_feat = p1_emb * p2_emb

        interaction_feat = torch.cat([concat_feat, abs_diff_feat, product_feat], dim=1)
        return interaction_feat


class GTBBlock(nn.Module):
    """
    GTB 块，模拟 Gradient Tree Boosting 的分步学习

    原始 GTB: 每轮学习 F_t(x) = F_{t-1}(x) + learning_rate * tree(x)
    这里: 用带残差连接的 MLP 层来模拟多轮提升的效果

    每个 GTB 块包含:
    1. 线性变换 + BatchNorm + ReLU
    2. 残差连接
    3. 与 GTB 的 max_depth 对应的层深度
    """

    def __init__(self, input_dim, hidden_dim, n_layers=3, dropout=0.3):
        super().__init__()

        layers = []
        dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        self.main_path = nn.Sequential(*layers)
        self.skip = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        return self.main_path(x) + self.skip(x)


class GTB_PPI(nn.Module):
    """
    GTB-PPI 模型

    核心架构 (基于梯度提升树的思想):
    1. FeatureInteractionModule: 构建蛋白质对的 concat/diff/product 交互特征
    2. GTB 编码器: 多层残差连接，模拟分步提升
    3. 全局聚合: mean pooling
    4. 分类头: 输出二分类或多标签分类 logits

    关键设计:
    - 残差连接: 模拟 GTB 的累加更新 F_t = F_{t-1} + eta * tree_t(x)
    - 多层结构: 模拟 GTB 的多轮迭代 (n_estimators)
    - BatchNorm + Dropout: 防止过拟合，与 GTB 的正则化相对应

    Args:
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 模型内部隐藏维度 (默认 256)
        n_estimators: 模拟 GTB 的树数量/提升轮次 (默认 4)
        max_depth: 模拟 GTB 的树深度 (默认 3)
        dropout: Dropout 概率
        num_classes: 类别数 (二分类=2, 多标签=标签数)
        task_type: 'binary' 或 'multilabel'
    """

    def __init__(self, protein_dim=1024, hidden_dim=256, n_estimators=4,
                 max_depth=3, dropout=0.3, num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.interaction_module = FeatureInteractionModule(protein_dim)
        interaction_output_dim = self.interaction_module.output_dim

        self.projection = nn.Sequential(
            nn.Linear(interaction_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.gtb_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(n_estimators)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if task_type == 'multilabel':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
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

        interaction_feat = self.interaction_module(p1_emb, p2_emb)

        x = self.projection(interaction_feat)

        for block in self.gtb_blocks:
            x = block(x)

        pooled = x

        output = self.classifier(pooled)
        return output


def GTB_PPI_Model(protein_dim=1024, hidden_dim=256, n_estimators=4,
                  max_depth=3, dropout=0.3, num_classes=2, task_type='binary', **kwargs):
    """
    GTB_PPI 模型工厂函数（用于 model_manager）
    """
    return GTB_PPI(
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        n_estimators=n_estimators,
        max_depth=max_depth,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
