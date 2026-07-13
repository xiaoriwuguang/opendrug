"""
PPI_TUnA (Topology-aware Protein-protein Interaction predictor with Unified Architecture)

基于预训练蛋白质嵌入的 PPI 预测模型（简化版 TUnA 架构）。

架构：
1. 共享投影层：protein_dim → hidden_dim
2. 交互特征融合：|proj1 - proj2|, proj1 * proj2, concat(proj1, proj2)
3. 多层感知机分类器

支持：
- PPI 二分类（预测两个蛋白质是否存在相互作用）
- PPI 多标签分类（预测两个蛋白质在多个类别上的相互作用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPITUnA(nn.Module):
    """
    PPI_TUnA 模型

    特点：
    - 共享编码器：两个蛋白质使用同一套投影权重（参数共享）
    - 交互建模：元素乘积 + 绝对差值 + 拼接，多角度捕捉相互作用
    - 支持二分类和多标签分类
    """

    def __init__(self, protein_dim=1024, hidden_dim=128, dropout=0.3,
                 num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        proj_dim = hidden_dim * 2

        # 共享投影层：将 protein_dim 投影到统一维度
        self.projection = nn.Sequential(
            nn.Linear(protein_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 交互特征融合层
        # 输入: [|p1-p2|, p1*p2, p1, p2] -> concat -> 4 * hidden_dim
        fusion_input_dim = hidden_dim * 4
        self.interaction_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 分类头
        classifier_input_dim = hidden_dim * 2
        if task_type == 'multilabel':
            # 多标签：输出 num_classes 维
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # 二分类：输出 num_classes 维（logits）
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
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

        if isinstance(p1_idx, torch.Tensor):
            device = next(self.parameters()).device
            p1_idx = p1_idx.to(device)
            p2_idx = p2_idx.to(device)
        else:
            device = next(self.parameters()).device
            p1_idx = torch.as_tensor(p1_idx, dtype=torch.long, device=device)
            p2_idx = torch.as_tensor(p2_idx, dtype=torch.long, device=device)

        protein_x = graph_or_none.protein_x
        p1_emb = protein_x[p1_idx]
        p2_emb = protein_x[p2_idx]

        # 共享投影
        p1_proj = self.projection(p1_emb)
        p2_proj = self.projection(p2_emb)

        # 交互特征建模
        diff = torch.abs(p1_proj - p2_proj)
        prod = p1_proj * p2_proj
        fusion = torch.cat([diff, prod, p1_proj, p2_proj], dim=1)
        fusion = self.interaction_fc(fusion)

        # 分类
        output = self.classifier(fusion)
        return output


def PPI_TUnA_Model(protein_dim=1024, hidden_dim=128, dropout=0.3,
                   num_classes=2, task_type='binary', **kwargs):
    """
    PPI_TUnA 模型工厂函数（用于 model_manager）
    """
    return PPITUnA(
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
