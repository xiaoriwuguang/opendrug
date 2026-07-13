"""
DTIAM (Drug-Target Interaction via Attentive Multimodal) 模型

基于 DTIAM 论文思想，使用注意力机制融合多模态特征进行药物-靶点相互作用预测。

模型特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 注意力机制融合多模态特征
3. 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionFusion(nn.Module):
    """跨模态注意力融合模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Drug-to-Protein attention
        self.drug_to_protein_query = nn.Linear(hidden_dim, hidden_dim)
        self.drug_to_protein_key = nn.Linear(hidden_dim, hidden_dim)
        self.drug_to_protein_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Protein-to-Drug attention
        self.protein_to_drug_query = nn.Linear(hidden_dim, hidden_dim)
        self.protein_to_drug_key = nn.Linear(hidden_dim, hidden_dim)
        self.protein_to_drug_value = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出融合 - 所有特征拼接后投影
        # [drug_emb, protein_emb, drug_context, protein_context] = 4 * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, drug_emb, protein_emb):
        """
        Args:
            drug_emb: [batch, hidden_dim]
            protein_emb: [batch, hidden_dim]
        Returns:
            fused: [batch, hidden_dim]
        """
        # Drug-to-Protein attention
        q = self.drug_to_protein_query(drug_emb).unsqueeze(1)  # [batch, 1, hidden]
        k = self.drug_to_protein_key(protein_emb).unsqueeze(1)  # [batch, 1, hidden]
        v = self.drug_to_protein_value(protein_emb).unsqueeze(1)  # [batch, 1, hidden]
        
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        drug_context = torch.matmul(attn_probs, v).squeeze(1)  # [batch, hidden]
        
        # Protein-to-Drug attention
        q2 = self.protein_to_drug_query(protein_emb).unsqueeze(1)
        k2 = self.protein_to_drug_key(drug_emb).unsqueeze(1)
        v2 = self.protein_to_drug_value(drug_emb).unsqueeze(1)
        
        attn_scores2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        attn_probs2 = F.softmax(attn_scores2, dim=-1)
        protein_context = torch.matmul(attn_probs2, v2).squeeze(1)  # [batch, hidden]
        
        # 融合所有特征
        combined = torch.cat([drug_emb, protein_emb, drug_context, protein_context], dim=1)
        fused = self.fusion(combined)
        
        return fused


class DTIAM_Model(nn.Module):
    """
    DTIAM 模型

    使用双分支编码器和跨模态注意力融合机制进行药物-靶点相互作用预测
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
                 task_type='regression', num_classes=2):
        super(DTIAM_Model, self).__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.task_type = task_type
        self.num_classes = num_classes

        # 药物编码器分支 - 将 drug_dim 投影到 hidden_dim
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

        # 蛋白质编码器分支 - 将 protein_dim 投影到 hidden_dim
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

        # 跨模态注意力融合
        self.attention_fusion = AttentionFusion(hidden_dim)

        # 交互层
        # 维度: fusion_feat(hidden) + abs_diff(hidden) + product(hidden) = 3 * hidden
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
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

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象 (包含 drug_x 和 protein_x)
            idx_batch: 批次数据 (drug_indices, protein_indices, labels)

        Returns:
            logits: 预测结果
        """
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        # 获取药物和蛋白质特征
        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        # 获取批次对应的特征
        drug_emb_raw = drug_x[drug_idx]
        protein_emb_raw = protein_x[protein_idx]

        # 药物分支编码 -> [batch, hidden_dim]
        drug_encoded = self.drug_encoder(drug_emb_raw)

        # 蛋白质分支编码 -> [batch, hidden_dim]
        protein_encoded = self.protein_encoder(protein_emb_raw)

        # 跨模态注意力融合 -> [batch, hidden_dim]
        fusion_feat = self.attention_fusion(drug_encoded, protein_encoded)

        # 交互特征
        abs_diff = torch.abs(drug_encoded - protein_encoded)  # [batch, hidden_dim]
        product = drug_encoded * protein_encoded  # [batch, hidden_dim]
        combined = torch.cat([fusion_feat, abs_diff, product], dim=1)  # [batch, 3*hidden_dim]

        # 交互层 -> [batch, hidden_dim]
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


def DTIAM(drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
          task_type='regression', num_classes=2, **kwargs):
    """
    DTIAM 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比率
        task_type: 任务类型 ('regression' 或 'classification')
        num_classes: 分类类别数

    Returns:
        DTIAM_Model 实例
    """
    return DTIAM_Model(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes
    )
