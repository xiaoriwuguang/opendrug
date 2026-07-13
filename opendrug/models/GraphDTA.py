"""
GraphDTA (Graph-based Drug-Target Affinity) 模型

基于 GraphDTA 论文思想，使用预训练嵌入进行药物-靶点亲和力预测。

模型特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 图卷积风格的交互建模
3. 1D卷积用于蛋白质序列特征提取
4. 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDTA_Model(nn.Module):
    """
    GraphDTA 模型

    使用双分支编码器处理药物和蛋白质嵌入，通过多层交互预测药物-靶点相互作用
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
                 task_type='regression', num_classes=2, gnn_layers=3):
        super(GraphDTA_Model, self).__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.task_type = task_type
        self.num_classes = num_classes
        self.gnn_layers = gnn_layers

        # 药物特征处理分支 - 图卷积风格
        self.drug_convs = nn.ModuleList()
        self.drug_norms = nn.ModuleList()
        in_dim = drug_dim
        for i in range(gnn_layers):
            out_dim = hidden_dim if i < gnn_layers - 1 else hidden_dim
            self.drug_convs.append(nn.Linear(in_dim, out_dim))
            self.drug_norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        # 蛋白质特征处理分支 - 1D卷积风格
        # 将蛋白质嵌入 reshape 为 (batch, channels, length) 格式
        self.protein_seq_len = 64  # 将 protein_dim 视为 64 通道，长度为 protein_dim // 64
        self.protein_channels = 64
        assert protein_dim % self.protein_channels == 0, f"protein_dim ({protein_dim}) must be divisible by {self.protein_channels}"
        
        self.protein_conv1 = nn.Conv1d(self.protein_channels, hidden_dim, kernel_size=3, padding=1)
        self.protein_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.protein_pool = nn.AdaptiveMaxPool1d(1)

        # 注意力池化层
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

        # 交互层
        interaction_dim = hidden_dim * 4  # drug, protein, abs_diff, product
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 回归输出头
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 分类输出头
        if task_type == 'classification':
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
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

        # 药物分支 - 图卷积风格处理
        drug_emb = drug_emb_raw
        for i, (conv, norm) in enumerate(zip(self.drug_convs, self.drug_norms)):
            drug_emb = conv(drug_emb)
            drug_emb = norm(drug_emb)
            drug_emb = F.relu(drug_emb)
            if self.training:
                drug_emb = F.dropout(drug_emb, p=self.dropout_rate, training=self.training)

        # 蛋白质分支 - 1D卷积风格处理
        # 将 (batch, protein_dim) reshape 为 (batch, channels, length)
        seq_length = self.protein_dim // self.protein_channels
        protein_seq = protein_emb_raw.view(-1, self.protein_channels, seq_length)
        
        # 卷积 + 池化
        protein_emb = F.relu(self.protein_conv1(protein_seq))
        protein_emb = F.relu(self.protein_conv2(protein_emb))
        protein_emb = self.protein_pool(protein_emb).squeeze(-1)  # (batch, hidden_dim)

        # 注意力加权
        drug_attn_weights = F.softmax(self.drug_attention(drug_emb), dim=1)
        protein_attn_weights = F.softmax(self.protein_attention(protein_emb), dim=1)
        drug_emb = drug_emb * drug_attn_weights
        protein_emb = protein_emb * protein_attn_weights

        # 交互特征
        abs_diff = torch.abs(drug_emb - protein_emb)
        product = drug_emb * protein_emb
        combined = torch.cat([drug_emb, protein_emb, abs_diff, product], dim=1)

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
            'num_classes': self.num_classes,
            'gnn_layers': self.gnn_layers
        }


def GraphDTA(drug_dim, protein_dim, hidden_dim=256, dropout=0.2,
               task_type='regression', num_classes=2, **kwargs):
    """
    GraphDTA 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比率
        task_type: 任务类型 ('regression' 或 'classification')
        num_classes: 分类类别数

    Returns:
        GraphDTA_Model 实例
    """
    return GraphDTA_Model(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes
    )
