"""
RSGCL_DTI (Representation Learning with Self-supervised Graph Contrastive Learning for Drug-Target Interaction) 模型

基于 RSGCL-DTI 论文思想，使用预训练嵌入进行药物-靶点相互作用预测。

模型特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 对比学习风格的表示学习
3. 多层感知机交互层
4. 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSGCL_DTI_Model(nn.Module):
    """
    RSGCL_DTI 模型

    使用双分支编码器处理药物和蛋白质嵌入，通过多层交互预测药物-靶点相互作用
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=512, dropout=0.2,
                 task_type='regression', num_classes=2):
        super(RSGCL_DTI_Model, self).__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.task_type = task_type
        self.num_classes = num_classes

        # 药物特征处理分支
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 蛋白质特征处理分支
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 对比学习风格的投影层
        self.drug_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.protein_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        # 交叉注意力层
        self.cross_attention_drug = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.cross_attention_protein = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 交互层
        interaction_dim = hidden_dim * 3  # drug, protein, abs_diff
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 回归输出头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 分类输出头
        if task_type == 'classification':
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )

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

        # 交叉注意力
        drug_query = drug_emb.unsqueeze(1)
        protein_key = protein_emb.unsqueeze(1)
        
        drug_attn, _ = self.cross_attention_drug(drug_query, protein_key, protein_key)
        protein_attn, _ = self.cross_attention_protein(protein_key, drug_query, drug_query)
        
        drug_emb = self.norm1(drug_emb + drug_attn.squeeze(1))
        protein_emb = self.norm2(protein_emb + protein_attn.squeeze(1))

        # 交互特征
        abs_diff = torch.abs(drug_emb - protein_emb)
        combined = torch.cat([drug_emb, protein_emb, abs_diff], dim=1)

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


def RSGCL_DTI(drug_dim, protein_dim, hidden_dim=512, dropout=0.2,
               task_type='regression', num_classes=2, **kwargs):
    """
    RSGCL_DTI 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比率
        task_type: 任务类型 ('regression' 或 'classification')
        num_classes: 分类类别数

    Returns:
        RSGCL_DTI_Model 实例
    """
    return RSGCL_DTI_Model(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes
    )
