"""
DTI (Drug-Target Interaction) 分类模型

用于药物-蛋白质相互作用分类预测的神经网络模型

特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 多种特征交互方式: 拼接、元素乘积、绝对差值
3. 支持多模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DTIPredictor(nn.Module):
    """
    DTI 预测器模型

    药物和蛋白质分别编码后，通过分类器预测相互作用概率
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
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
            graph_or_none: 图数据对象 (包含 drug_x 和 protein_x)
            idx_batch: 批次数据 (drug_indices, protein_indices, labels)

        Returns:
            logits: 分类 logits
        """
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb = self.drug_encoder(drug_x[drug_idx])
        protein_emb = self.protein_encoder(protein_x[protein_idx])

        combined = torch.cat([
            drug_emb,
            protein_emb,
            torch.abs(drug_emb - protein_emb),
            drug_emb * protein_emb
        ], dim=-1)

        logits = self.classifier(combined)
        return logits


class ModalAttentionFusion(nn.Module):
    """
    多模态注意力融合模块

    使用注意力机制融合多个模态的特征
    """

    def __init__(self, modal_dims, out_dim, dropout=0.0):
        super().__init__()
        self.modal_dims = list(map(int, modal_dims))
        self.out_dim = int(out_dim)

        self.proj = nn.ModuleList([nn.Linear(d, self.out_dim) for d in self.modal_dims])
        self.scor = nn.ModuleList([nn.Linear(d, 1) for d in self.modal_dims])

    def forward(self, x, splits):
        """
        Args:
            x: [N, sum(d_m)] - 拼接后的多模态特征
            splits: [0, d1, d1+d2, ..., sum] - 各模态边界

        Returns:
            fused: [N, out_dim] - 融合后的特征
            attention: [N, M] - 注意力权重
        """
        feats = []
        scores = []

        for l, r, pj, sc in zip(splits[:-1], splits[1:], self.proj, self.scor):
            xm = x[:, l:r]
            hm = F.relu(pj(xm))
            sm = sc(xm)
            feats.append(hm)
            scores.append(sm)

        S = torch.cat(scores, dim=1)
        A = torch.softmax(S, dim=1)
        H = torch.stack(feats, dim=1)
        fused = torch.sum(A.unsqueeze(-1) * H, dim=1)

        return fused, A


class DTIMultiModalModel(nn.Module):
    """
    多模态 DTI 模型

    融合药物的多个模态嵌入和蛋白质的多个模态嵌入
    """

    def __init__(self, drug_dims, protein_dims, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()

        self.drug_dims = list(drug_dims) if isinstance(drug_dims, (list, tuple)) else [drug_dims]
        self.protein_dims = list(protein_dims) if isinstance(protein_dims, (list, tuple)) else [protein_dims]
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.drug_fusion = ModalAttentionFusion(self.drug_dims, hidden_dim, dropout)
        self.protein_fusion = ModalAttentionFusion(self.protein_dims, hidden_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, graph_or_none, idx_batch):
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_splits = torch.tensor([0] + list(torch.cumsum(torch.tensor(self.drug_dims), dim=0)))
        protein_splits = torch.tensor([0] + list(torch.cumsum(torch.tensor(self.protein_dims), dim=0)))

        drug_emb, _ = self.drug_fusion(drug_x[drug_idx], drug_splits)
        protein_emb, _ = self.protein_fusion(protein_x[protein_idx], protein_splits)

        combined = torch.cat([
            drug_emb,
            protein_emb,
            torch.abs(drug_emb - protein_emb),
            drug_emb * protein_emb
        ], dim=-1)

        logits = self.classifier(combined)
        return logits


# 为了兼容性，提供默认的 DTI 模型
def DTI(feature=None, hidden1=256, hidden2=256, num_classes=2, **kwargs):
    """
    DTI 模型工厂函数
    """
    drug_dim = kwargs.get('drug_dim', 512)
    protein_dim = kwargs.get('protein_dim', 512)

    return DTIPredictor(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden1,
        dropout=kwargs.get('dropout', 0.3),
        num_classes=num_classes
    )


# 导出模型别名
DTI_Bilinear = DTIPredictor
DTI_MultiModal = DTIMultiModalModel
