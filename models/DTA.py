"""
DTA (Drug-Target Affinity) 模型

用于药物-蛋白质亲和力预测的神经网络模型

特点:
1. 双分支编码器: 药物分支 + 蛋白质分支
2. 多种特征交互方式: 拼接、元素乘积、绝对差值
3. 支持图神经网络编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv


class DTABilinearPredictor(nn.Module):
    """
    简单的双线性预测器模型

    药物和蛋白质分别编码后，通过双线性变换预测亲和力
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim

        # 药物编码器
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 蛋白质编码器
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象 (包含 drug_x 和 protein_x)
            idx_batch: 批次数据 (drug_indices, protein_indices, labels)

        Returns:
            logits: 预测的亲和力分数
        """
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        device = drug_idx.device

        # 获取药物和蛋白质特征
        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        # 获取批次对应的特征
        drug_emb = self.drug_encoder(drug_x[drug_idx])
        protein_emb = self.protein_encoder(protein_x[protein_idx])

        # 特征交互: [d, p, |d-p|, d*p]
        combined = torch.cat([
            drug_emb,
            protein_emb,
            torch.abs(drug_emb - protein_emb),
            drug_emb * protein_emb
        ], dim=-1)

        # 预测
        logits = self.predictor(combined)
        return logits


class DTAGNNModel(nn.Module):
    """
    基于图神经网络的 DTA 模型

    在药物-蛋白质异构图上进行消息传递
    """

    def __init__(self, drug_dim, protein_dim, hidden_dim=256, num_gnn_layers=2,
                 gnn_type='sage', dropout=0.3):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type

        # 初始化嵌入层
        self.drug_embedding = nn.Linear(drug_dim, hidden_dim)
        self.protein_embedding = nn.Linear(protein_dim, hidden_dim)

        # 图神经网络层
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            if gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
            else:
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = dropout

    def forward(self, graph_or_none, idx_batch):
        """前向传播"""
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        # 获取节点特征
        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x
        edge_index = graph_or_none.edge_index

        # 初始化节点嵌入
        num_drugs = drug_x.shape[0]
        num_proteins = protein_x.shape[0]

        # 药物节点嵌入
        drug_emb = F.relu(self.drug_embedding(drug_x))
        # 蛋白质节点嵌入
        protein_emb = F.relu(self.protein_embedding(protein_x))

        # 拼接所有节点特征
        all_node_feats = torch.cat([drug_emb, protein_emb], dim=0)

        # 图神经网络消息传递
        for i, gnn in enumerate(self.gnn_layers):
            all_node_feats = gnn(all_node_feats, edge_index)
            all_node_feats = F.relu(all_node_feats)
            all_node_feats = F.dropout(all_node_feats, p=self.dropout, training=self.training)

        # 分离药物和蛋白质嵌入
        updated_drug_emb = all_node_feats[:num_drugs]
        updated_protein_emb = all_node_feats[num_drugs:]

        # 获取批次对应的嵌入
        batch_drug_emb = updated_drug_emb[drug_idx]
        batch_protein_emb = updated_protein_emb[protein_idx]

        # 特征交互
        combined = torch.cat([
            batch_drug_emb,
            batch_protein_emb,
            torch.abs(batch_drug_emb - batch_protein_emb),
            batch_drug_emb * batch_protein_emb
        ], dim=-1)

        # 预测
        logits = self.predictor(combined)
        return logits


class DTAMultiModalModel(nn.Module):
    """
    多模态 DTA 模型

    融合药物的多个模态嵌入和蛋白质的多个模态嵌入
    """

    def __init__(self, drug_dims, protein_dims, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.drug_dims = list(drug_dims) if isinstance(drug_dims, (list, tuple)) else [drug_dims]
        self.protein_dims = list(protein_dims) if isinstance(protein_dims, (list, tuple)) else [protein_dims]
        self.hidden_dim = hidden_dim

        # 药物多模态注意力融合
        self.drug_fusion = ModalAttentionFusion(self.drug_dims, hidden_dim, dropout)

        # 蛋白质多模态注意力融合
        self.protein_fusion = ModalAttentionFusion(self.protein_dims, hidden_dim, dropout)

        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_or_none, idx_batch):
        """前向传播"""
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        # 获取特征
        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        # 药物特征分割
        drug_splits = [0] + list(self.drug_dims)
        drug_splits = torch.tensor([0] + list(torch.cumsum(torch.tensor(self.drug_dims), dim=0)))

        # 蛋白质特征分割
        protein_splits = torch.tensor([0] + list(torch.cumsum(torch.tensor(self.protein_dims), dim=0)))

        # 融合药物多模态特征
        drug_emb, _ = self.drug_fusion(drug_x[drug_idx], drug_splits)

        # 融合蛋白质多模态特征
        protein_emb, _ = self.protein_fusion(protein_x[protein_idx], protein_splits)

        # 特征交互
        combined = torch.cat([
            drug_emb,
            protein_emb,
            torch.abs(drug_emb - protein_emb),
            drug_emb * protein_emb
        ], dim=-1)

        # 预测
        logits = self.predictor(combined)
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


# 为了兼容性，提供默认的 DTA 模型
def DTA(feature=None, hidden1=256, hidden2=256, num_classes=None, **kwargs):
    """
    DTA 模型工厂函数

    Args:
        feature: 特征维度（如果只有单模态）
        hidden1: 第一隐藏层维度
        hidden2: 第二隐藏层维度
        num_classes: 类别数（DTA 回归任务中不使用）

    Returns:
        model: DTA 模型实例
    """
    drug_dim = kwargs.get('drug_dim', 512)
    protein_dim = kwargs.get('protein_dim', 512)

    return DTABilinearPredictor(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden1,
        dropout=kwargs.get('dropout', 0.3)
    )


# 导出模型
DTA_Bilinear = DTABilinearPredictor
DTA_GNN = DTAGNNModel
DTA_MultiModal = DTAMultiModalModel
