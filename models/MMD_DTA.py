"""
MMD_DTA (Multi-Modal Deep Neural Network for Drug-Target Affinity) 模型

基于 MMD-DTA 论文思想，使用预训练文本嵌入进行药物-靶点亲和力预测。

模型特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 多层感知机交互: 使用 MLP 学习药物和蛋白质嵌入的交互
3. 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMD_DTA_Model(nn.Module):
    """
    MMD_DTA 模型

    使用多层感知机对药物和蛋白质嵌入进行交互建模
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
                 task_type='regression', num_classes=2):
        super(MMD_DTA_Model, self).__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.task_type = task_type
        self.num_classes = num_classes

        # 药物特征处理分支
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        # 蛋白质特征处理分支
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        # 注意力机制层
        self.drug_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.protein_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 交互层 - 使用多种交互方式
        # 交互特征: [d, p, |d-p|, d*p, d-p, (d+p)/2]
        interaction_dim = hidden_dim * 5

        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 回归输出头
        self.regression_head = nn.Linear(hidden_dim // 2, 1)

        # 分类输出头
        if task_type == 'classification':
            self.classification_head = nn.Linear(hidden_dim // 2, num_classes)

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

        # 编码
        drug_emb = self.drug_encoder(drug_emb_raw)
        protein_emb = self.protein_encoder(protein_emb_raw)

        # 注意力加权
        drug_attn_weights = F.softmax(self.drug_attention(drug_emb), dim=1)
        protein_attn_weights = F.softmax(self.protein_attention(protein_emb), dim=1)

        drug_emb = drug_emb * drug_attn_weights
        protein_emb = protein_emb * protein_attn_weights

        # 构建交互特征
        # |d-p|: 绝对差值
        abs_diff = torch.abs(drug_emb - protein_emb)
        # d*p: 元素乘积
        product = drug_emb * protein_emb
        # d-p: 差值
        diff = drug_emb - protein_emb
        # (d+p)/2: 平均
        avg = (drug_emb + protein_emb) / 2

        # 拼接所有交互特征
        combined = torch.cat([drug_emb, protein_emb, abs_diff, product, avg], dim=1)

        # 交互层
        interaction_feat = self.interaction_mlp(combined)

        # 输出
        if self.task_type == 'classification':
            logits = self.classification_head(interaction_feat)
            return logits
        else:
            logits = self.regression_head(interaction_feat)
            return logits

    def get_config(self):
        """返回模型配置"""
        return {
            'drug_dim': self.drug_dim,
            'protein_dim': self.protein_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'task_type': self.task_type,
            'num_classes': self.num_classes
        }


def MMD_DTA(drug_dim, protein_dim, hidden_dim=256, dropout=0.2, 
            task_type='regression', num_classes=2, **kwargs):
    """
    MMD_DTA 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比率
        task_type: 任务类型 ('regression' 或 'classification')
        num_classes: 分类类别数

    Returns:
        MMD_DTA_Model 实例
    """
    return MMD_DTA_Model(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes
    )
