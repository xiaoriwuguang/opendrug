

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGENFMLinear(nn.Module):
    """
    KGE-NFM 线性融合模块

    融合头尾实体的嵌入特征
    """

    def __init__(self, embedding_dim, output_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        out_dim = output_dim or embedding_dim

        self.linear = nn.Linear(embedding_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, head_emb, tail_emb):
        """
        Args:
            head_emb: [N, embedding_dim] 头实体嵌入
            tail_emb: [N, embedding_dim] 尾实体嵌入

        Returns:
            fused_emb: [N, output_dim]
        """
        combined = torch.cat([head_emb, tail_emb], dim=-1)
        out = self.linear(combined)
        out = self.bn(out)
        out = F.relu(out)
        return out


class NFMInteraction(nn.Module):
    """
    Neural Factorization Machine 交互层

    使用 Bi-Interaction 池化捕获特征交互
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.bi_interaction = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: [N, input_dim] 交互特征

        Returns:
            output: [N, hidden_dim]
        """
        linear_out = self.linear(x).squeeze(-1)
        bi_out = self.bi_interaction(x)
        bi_out = F.relu(bi_out)
        return bi_out


class KGE_NFM(nn.Module):
    """
    KGE-NFM 模型

    结合知识图谱嵌入特征和 NFM 进行药物-靶点预测

    特点:
    1. 融合头尾实体的 KGE 嵌入
    2. Bi-Interaction 池化捕获二阶特征交互
    3. DNN 层进行深层特征提取
    4. 支持分类和回归任务
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, nfm_hidden=128,
                 dnn_layers=2, dropout=0.3, num_classes=2, task_type='classification'):
        """
        Args:
            drug_dim: 药物嵌入维度
            protein_dim: 蛋白质嵌入维度
            hidden_dim: 隐藏层维度
            nfm_hidden: NFM 交互层隐藏维度
            dnn_layers: DNN 层数
            dropout: Dropout 比例
            num_classes: 类别数 (分类任务)
            task_type: 'classification' 或 'regression'
        """
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        total_dim = drug_dim + protein_dim

        # 药物编码器
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 蛋白质编码器
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # NFM 交互层
        self.nfm = NFMInteraction(hidden_dim * 4, nfm_hidden)

        # DNN 层
        dnn_modules = []
        dnn_input_dim = nfm_hidden
        for i in range(dnn_layers):
            dnn_modules.append(nn.Linear(dnn_input_dim, hidden_dim))
            dnn_modules.append(nn.BatchNorm1d(hidden_dim))
            dnn_modules.append(nn.ReLU())
            dnn_modules.append(nn.Dropout(dropout))
            dnn_input_dim = hidden_dim
        self.dnn = nn.Sequential(*dnn_modules)

        # 输出层
        if task_type == 'classification':
            self.output_layer = nn.Linear(hidden_dim, num_classes)
        else:
            self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象 (包含 drug_x 和 protein_x)
            idx_batch: 批次数据 (drug_indices, protein_indices, labels)

        Returns:
            output: 预测结果 (logits 或 scores)
        """
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb = self.drug_encoder(drug_x[drug_idx])
        protein_emb = self.protein_encoder(protein_x[protein_idx])

        interaction_feats = torch.cat([
            drug_emb,
            protein_emb,
            drug_emb * protein_emb,
            torch.abs(drug_emb - protein_emb)
        ], dim=-1)

        nfm_out = self.nfm(interaction_feats)
        dnn_out = self.dnn(nfm_out)
        output = self.output_layer(dnn_out)
        return output


class KGE_NFM_Bilinear(nn.Module):
    """
    KGE-NFM 双线性版本

    使用双线性变换建模药物-蛋白质交互
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.3,
                 num_classes=2, task_type='classification'):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes if task_type == 'classification' else 1)
        )

    def forward(self, graph_or_none, idx_batch):
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb = self.drug_encoder(drug_x[drug_idx])
        protein_emb = self.protein_encoder(protein_x[protein_idx])

        bilinear_out = self.bilinear(drug_emb, protein_emb)

        combined = torch.cat([
            drug_emb,
            protein_emb,
            bilinear_out
        ], dim=-1)

        output = self.classifier(combined)
        return output


class KGE_AttentionFusion(nn.Module):
    """
    KGE 注意力融合模块

    使用注意力机制融合多种嵌入表示
    """

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, head_emb, tail_emb):
        """
        Args:
            head_emb: [N, input_dim]
            tail_emb: [N, input_dim]

        Returns:
            fused: [N, hidden_dim]
            attention: [N, 2] 注意力权重
        """
        combined = torch.stack([head_emb, tail_emb], dim=1)

        q = self.query(combined.mean(dim=1, keepdim=True))
        k = self.key(combined)
        v = self.value(combined)

        scores = torch.sum(q * k, dim=-1, keepdim=True) / (q.size(-1) ** 0.5)
        attention = F.softmax(scores, dim=1)

        fused = torch.sum(attention * v, dim=1)

        return fused, attention.squeeze(-1)


class KGE_NFM_Attention(nn.Module):
    """
    KGE-NFM with Attention 注意力版本

    使用注意力机制融合药物和蛋白质嵌入
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, nfm_hidden=128,
                 dropout=0.3, num_classes=2, task_type='classification'):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.task_type = task_type

        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention_fusion = KGE_AttentionFusion(hidden_dim, hidden_dim)

        self.nfm = NFMInteraction(hidden_dim * 3, nfm_hidden)

        self.dnn = nn.Sequential(
            nn.Linear(nfm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        if task_type == 'classification':
            self.output_layer = nn.Linear(hidden_dim // 2, num_classes)
        else:
            self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, graph_or_none, idx_batch):
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb = self.drug_encoder(drug_x[drug_idx])
        protein_emb = self.protein_encoder(protein_x[protein_idx])

        fused_emb, attention = self.attention_fusion(drug_emb, protein_emb)

        interaction_feats = torch.cat([
            drug_emb,
            protein_emb,
            fused_emb
        ], dim=-1)

        nfm_out = self.nfm(interaction_feats)
        dnn_out = self.dnn(nfm_out)
        output = self.output_layer(dnn_out)
        return output


def KGE_NFM_Model(drug_dim=512, protein_dim=512, hidden_dim=256, nfm_hidden=128,
                  dnn_layers=2, dropout=0.3, num_classes=2, task_type='classification',
                  model_variant='standard', **kwargs):
    """
    KGE_NFM 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        nfm_hidden: NFM 交互层维度
        dnn_layers: DNN 层数
        dropout: Dropout 比例
        num_classes: 类别数 (分类任务)
        task_type: 'classification' 或 'regression'
        model_variant: 'standard', 'bilinear', 或 'attention'

    Returns:
        model: KGE-NFM 模型实例
    """
    if model_variant == 'bilinear':
        return KGE_NFM_Bilinear(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            task_type=task_type
        )
    elif model_variant == 'attention':
        return KGE_NFM_Attention(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            nfm_hidden=nfm_hidden,
            dropout=dropout,
            num_classes=num_classes,
            task_type=task_type
        )
    else:
        return KGE_NFM(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            nfm_hidden=nfm_hidden,
            dnn_layers=dnn_layers,
            dropout=dropout,
            num_classes=num_classes,
            task_type=task_type
        )


# 导出模型别名
KGE_NFM_Standard = KGE_NFM
KGE_NFM_Standard_Model = KGE_NFM_Model
