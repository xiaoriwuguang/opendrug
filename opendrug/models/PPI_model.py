"""
PPI (Protein-Protein Interaction) 预测模型

支持:
- PPI 二分类: 预测两个蛋白质是否存在相互作用
- PPI 多标签分类: 预测两个蛋白质在多个类别上的相互作用

特点:
- 对称性建模: 两端都是蛋白质，使用共享编码器
- 支持多标签输出 (multi-hot)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPIPredictor(nn.Module):
    """
    PPI 预测器

    两个蛋白质分别经编码器后，通过交互模块预测相互作用
    """

    def __init__(self, protein_dim, hidden_dim=256, dropout=0.3,
                 num_classes=2, task_type='binary'):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.interaction_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
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
            output: [B, num_classes]
        """
        p1_idx = idx_batch[0]
        p2_idx = idx_batch[1]

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
        p1_emb = protein_x[p1_idx]
        p2_emb = protein_x[p2_idx]

        p1_hidden = self.protein_encoder(p1_emb)
        p2_hidden = self.protein_encoder(p2_emb)

        interaction = torch.cat([
            p1_hidden,
            p2_hidden,
            torch.abs(p1_hidden - p2_hidden),
            p1_hidden * p2_hidden,
        ], dim=1)

        output = self.interaction_predictor(interaction)
        return output


def PPI_Model(protein_dim=512, hidden_dim=256, dropout=0.3,
              num_classes=2, task_type='binary', **kwargs):
    """
    PPI 模型工厂函数（用于 model_manager）
    """
    return PPIPredictor(
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
