"""
DrugBAN (Dual Attention Network) 模型

基于 Bilinear Attention Network 的药物-靶点相互作用预测模型。
适配 OpenDrug 框架，使用预训练嵌入代替原始 SMILES/序列编码。

核心特点:
1. 双线性注意力机制 (Bilinear Attention) 捕获药物-蛋白质交互
2. 基于嵌入的特征编码 (适配 OpenDrug 预训练嵌入)
3. 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class BANLayer(nn.Module):
    """
    Bilinear Attention Network Layer

    实现双线性注意力机制，用于药物-蛋白质特征交互。
    核心思想是通过爱因斯坦求和计算注意力权重，实现细粒度的模态交互。
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super().__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)

        if 1 < k:
            self.p_net = nn.AvgPool1d(k, stride=k)

        hk = h_dim * k
        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, hk).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(hk, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        """使用双线性注意力的加权求和进行特征池化"""
        v_flat = v.reshape(v.size(0), -1)   # [B, -1]
        q_flat = q.reshape(q.size(0), -1)   # [B, -1]
        a_flat = att_map.reshape(att_map.size(0), -1)  # [B, -1]
        # 双线性加权: v_flat[b,k] * a_flat[b,k'] * q_flat[b,k'']
        # 先让 a 的最后一维 broadcast 到 hk
        a_exp = a_flat.unsqueeze(1)  # [B, 1, -1]
        q_exp = q_flat.unsqueeze(2)  # [B, 1, hk]
        vq = v_flat.unsqueeze(1) * a_exp * q_exp  # [B, -1, hk]
        fusion_logits = vq.sum(dim=1)  # [B, hk]
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits.transpose(1, 2)).transpose(1, 2).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        """
        Args:
            v: 药物特征 [batch, v_dim]
            q: 蛋白质特征 [batch, q_dim]
            softmax: 是否返回 softmax 后的注意力图

        Returns:
            logits: 融合后的特征 [batch, h_dim]
            att_maps: 注意力图 [batch, h_out, v_num, q_num]
        """
        v_num = v.size(1)
        q_num = q.size(1)

        if self.h_out <= self.c:
            v_ = self.v_net(v).unsqueeze(1)
            q_ = self.q_net(q).unsqueeze(1)
            att_maps = torch.einsum('xhyk,bmk,bmk->bhm', self.h_mat, v_, q_) + self.h_bias.squeeze(0).squeeze(-1)  # [B, h_out]
            att_maps = att_maps.unsqueeze(-1).unsqueeze(-1)  # [B, h_out, 1, 1]
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        if softmax:
            p = nn.functional.softmax(att_maps.reshape(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.reshape(-1, self.h_out, v_num, q_num)

        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i

        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """全连接网络，用于特征变换"""

    def __init__(self, dims, act='ReLU', dropout=0):
        super().__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())

        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class DrugBAN_Embedding(nn.Module):
    """
    DrugBAN 基于嵌入的版本

    使用 OpenDrug 预训练嵌入代替原始 SMILES/序列编码。
    保留双线性注意力机制的核心设计。
    """

    def __init__(self, drug_dim=512, protein_dim=512, hidden_dim=256,
                 ban_heads=2, dropout=0.2, task_type='classification',
                 num_classes=2, ban_variant='standard'):
        """
        Args:
            drug_dim: 药物嵌入维度
            protein_dim: 蛋白质嵌入维度
            hidden_dim: 隐藏层维度
            ban_heads: BAN 层注意力头数
            dropout: Dropout 比率
            task_type: 'classification' 或 'regression'
            num_classes: 分类类别数
            ban_variant: BAN 变体 ('standard', 'enhanced')
        """
        super().__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.task_type = task_type
        self.num_classes = num_classes

        # 药物编码器: 投影到统一维度
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 蛋白质编码器: 投影到统一维度
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BAN 层配置
        ban_in_dim = hidden_dim  # 编码后的维度
        ban_h_dim = hidden_dim // 2  # BAN 内部维度
        ban_out = ban_heads  # 注意力头数

        # 双线性注意力网络
        self.ban = weight_norm(
            BANLayer(
                v_dim=ban_in_dim,
                q_dim=ban_in_dim,
                h_dim=ban_h_dim,
                h_out=ban_out,
                act='ReLU',
                dropout=dropout,
                k=1
            ),
            name='h_mat', dim=None
        )

        # 交互特征融合
        if ban_variant == 'enhanced':
            # 增强版: 使用拼接 + 交互
            self.interaction_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 标准版: 直接使用 BAN 输出
            self.interaction_fusion = nn.Sequential(
                nn.Linear(ban_h_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # 输出层
        if task_type == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

        self.dropout_layer = nn.Dropout(dropout)

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

        # 获取药物和蛋白质特征
        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        # 获取批次对应的特征
        drug_emb_raw = drug_x[drug_idx]
        protein_emb_raw = protein_x[protein_idx]

        # 编码药物和蛋白质特征
        drug_emb = self.drug_encoder(drug_emb_raw)
        protein_emb = self.protein_encoder(protein_emb_raw)

        # BAN 层进行双线性注意力交互
        ban_out, att_maps = self.ban(drug_emb, protein_emb)

        # 与原始特征拼接
        if hasattr(self.interaction_fusion, 'main') and len(self.interaction_fusion.main) > 2:
            # 增强版融合
            combined = torch.cat([ban_out, drug_emb, protein_emb], dim=1)
        else:
            # 标准版
            combined = ban_out

        # 交互融合
        interaction_feat = self.interaction_fusion(combined)

        # 输出预测
        output = self.output_layer(interaction_feat)

        return output

    def get_attention_maps(self, graph_or_none, idx_batch):
        """获取注意力图，用于可解释性分析"""
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb_raw = drug_x[drug_idx]
        protein_emb_raw = protein_x[protein_idx]

        drug_emb = self.drug_encoder(drug_emb_raw)
        protein_emb = self.protein_encoder(protein_emb_raw)

        ban_out, att_maps = self.ban(drug_emb, protein_emb, softmax=True)

        return att_maps


class DrugBAN_Model(nn.Module):
    """
    DrugBAN 模型主类

    完整的 DrugBAN 模型，包含:
    1. 药物编码器
    2. 蛋白质编码器
    3. 双线性注意力层
    4. 分类/回归输出头
    """

    def __init__(self, drug_dim=512, protein_dim=512, hidden_dim=256,
                 ban_heads=2, dropout=0.2, task_type='classification',
                 num_classes=2, ban_variant='standard'):
        super().__init__()

        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.task_type = task_type
        self.num_classes = num_classes

        # 药物编码器
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

        # 蛋白质编码器
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

        # BAN 层
        ban_in_dim = hidden_dim
        ban_h_dim = hidden_dim // 2
        ban_out = ban_heads

        self.ban = weight_norm(
            BANLayer(
                v_dim=ban_in_dim,
                q_dim=ban_in_dim,
                h_dim=ban_h_dim,
                h_out=ban_out,
                act='ReLU',
                dropout=dropout,
                k=1
            ),
            name='h_mat', dim=None
        )

        # 交互特征处理
        self.fusion = nn.Sequential(
            nn.Linear(ban_h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 额外交互特征
        self.extra_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 输出头
        if task_type == 'classification':
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, graph_or_none, idx_batch):
        """前向传播"""
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb_raw = drug_x[drug_idx]
        protein_emb_raw = protein_x[protein_idx]

        # 编码
        drug_emb = self.drug_encoder(drug_emb_raw)
        protein_emb = self.protein_encoder(protein_emb_raw)

        # BAN 交互
        ban_out, _ = self.ban(drug_emb, protein_emb)

        # 融合 BAN 输出
        interaction = self.fusion(ban_out)

        # 额外交互特征
        abs_diff = torch.abs(drug_emb - protein_emb)
        product = drug_emb * protein_emb
        extra_feat = torch.cat([drug_emb, protein_emb, abs_diff, product], dim=1)
        extra_interaction = self.extra_interaction(extra_feat)

        # 组合交互特征
        final_feat = interaction + extra_interaction

        # 输出
        output = self.output_head(final_feat)

        return output


def DrugBAN(drug_dim=512, protein_dim=512, hidden_dim=256,
            ban_heads=2, dropout=0.2, task_type='classification',
            num_classes=2, ban_variant='standard', **kwargs):
    """
    DrugBAN 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        ban_heads: BAN 层注意力头数
        dropout: Dropout 比率
        task_type: 'classification' 或 'regression'
        num_classes: 分类类别数
        ban_variant: BAN 变体 ('standard', 'enhanced')

    Returns:
        DrugBAN 模型实例
    """
    return DrugBAN_Model(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        ban_heads=ban_heads,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes
    )


def DrugBAN_Embedding_Model(drug_dim=512, protein_dim=512, hidden_dim=256,
                            ban_heads=2, dropout=0.2, task_type='classification',
                            num_classes=2, ban_variant='standard', **kwargs):
    """
    DrugBAN 嵌入版本工厂函数

    使用更简单的架构，基于嵌入编码。
    """
    return DrugBAN_Embedding(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        ban_heads=ban_heads,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes,
        ban_variant=ban_variant
    )
