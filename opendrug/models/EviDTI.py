"""
EviDTI (Evidential Deep Learning for Drug-Target Interaction) 模型

基于 EviDTI 论文思想，使用预训练嵌入进行药物-靶点相互作用预测。

模型特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 证据感知学习（Evidential Learning）
3. 支持不确定性估计
4. 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EviDTI_Model(nn.Module):
    """
    EviDTI 模型

    使用双分支编码器处理药物和蛋白质嵌入，通过证据感知学习预测药物-靶点相互作用
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
                 task_type='regression', num_classes=2, evidential=True):
        super(EviDTI_Model, self).__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.task_type = task_type
        self.num_classes = num_classes
        self.evidential = evidential

        # 药物特征处理分支
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )

        # 蛋白质特征处理分支 - LightAttention 风格
        self.protein_seq_len = 64
        self.protein_channels = 64
        assert protein_dim % self.protein_channels == 0, f"protein_dim ({protein_dim}) must be divisible by {self.protein_channels}"
        
        self.feature_convolution = nn.Conv1d(self.protein_channels, hidden_dim, kernel_size=3, padding=1)
        self.attention_convolution = nn.Conv1d(self.protein_channels, hidden_dim, kernel_size=3, padding=1)
        
        self.protein_linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim)
        )

        # 交互层
        interaction_dim = hidden_dim * 3  # drug, protein, abs_diff
        self.fc1 = nn.Linear(interaction_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)

        # 证据层 (Evidential Layer)
        if evidential and task_type == 'classification':
            # 输出 num_classes 个参数 (用于 Dirichlet 分布)
            self.evidence_head = nn.Linear(hidden_dim // 4, num_classes)
        else:
            # 标准输出
            self.evidence_head = None

        # 回归输出头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 8, 1)
        )

        # 分类输出头
        if task_type == 'classification':
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_dim // 4, hidden_dim // 8),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 8, num_classes)
            )

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象 (包含 drug_x 和 protein_x)
            idx_batch: 批次数据 (drug_indices, protein_indices, labels)

        Returns:
            logits: 预测结果 (回归: 亲和力分数, 分类: 类别 logits)
        """
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        # 获取药物和蛋白质特征
        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        # 获取批次对应的特征
        drug_emb_raw = drug_x[drug_idx]
        protein_emb_raw = protein_x[protein_idx]

        # 药物分支
        drug_emb = self.drug_encoder(drug_emb_raw)

        # 蛋白质分支 - LightAttention 风格
        seq_length = self.protein_dim // self.protein_channels
        protein_seq = protein_emb_raw.view(-1, self.protein_channels, seq_length)
        
        # 特征卷积
        feature_out = self.feature_convolution(protein_seq)
        feature_out = self.dropout(feature_out)
        
        # 注意力卷积
        attention = self.attention_convolution(protein_seq)
        
        # Softmax 注意力加权
        attention_weight = F.softmax(attention, dim=-1)
        
        # 加权求和 + 最大值池化
        protein_emb1 = torch.sum(feature_out * attention_weight, dim=-1)
        protein_emb2, _ = torch.max(feature_out, dim=-1)
        protein_emb = torch.cat([protein_emb1, protein_emb2], dim=-1)
        protein_emb = self.protein_linear(protein_emb)

        # 交互特征
        abs_diff = torch.abs(drug_emb - protein_emb)
        combined = torch.cat([drug_emb, protein_emb, abs_diff], dim=1)

        # 交互层
        fully1 = self.leaky_relu(self.fc1(combined))
        fully1 = self.dropout(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))

        # 输出
        if self.evidential and self.task_type == 'classification' and self.evidence_head is not None:
            # 证据分类输出: alpha 参数
            evidence_out = self.evidence_head(fully3)
            # 使用 softplus + 1 确保 evidence > 0
            alphas = F.softplus(evidence_out) + 1
            return alphas
        elif self.task_type == 'classification':
            logits = self.classification_head(fully3)
            return logits
        else:
            logits = self.regression_head(fully3)
            return logits

    def get_config(self):
        """返回模型配置"""
        return {
            'drug_dim': self.drug_dim,
            'protein_dim': self.protein_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'task_type': self.task_type,
            'num_classes': self.num_classes,
            'evidential': self.evidential
        }


def EviDTI(drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
           task_type='regression', num_classes=2, **kwargs):
    """
    EviDTI 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比率
        task_type: 任务类型 ('regression' 或 'classification')
        num_classes: 分类类别数

    Returns:
        EviDTI_Model 实例
    """
    return EviDTI_Model(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes
    )
