"""
GraphPPIS (Graph Convolutional Network for Protein-Protein Interaction Sites) 模型

基于论文: "GraphPPIS: a graph neural network for predicting protein-protein interaction sites"
https://academic.oup.com/bib/article/22/6/bbab259/6322794

原始 GraphPPIS 设计:
- 使用 3D 结构信息构建蛋白质接触图 (contact map / distance map)
- BLOSUM62 / PSSM + HMM 特征 + DSSP 二级结构特征
- GCNII (Graph Convolutional Network with Initial residual) 层叠
- 每层使用 Initial residual + Identity mapping 缓解过平滑

OpenDrug 适配:
- 输入为蛋白质级嵌入 (protein_x: [N, protein_dim])
- 在 PPI 网络图结构上进行图卷积编码
- 双通道 (shared encoder) 分别编码两个蛋白质的图嵌入
- 交互特征: concat + |emb1 - emb2| + emb1 * emb2
- 使用 AdaptiveAvgPool1d 将不同大小蛋白质图池化到统一维度

支持:
- PPI 二分类 (CrossEntropyLoss)
- PPI 多标签分类 (BCEWithLogitsLoss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class GraphConvolution(nn.Module):
    """
    GCNII 图卷积层

    改进的图卷积层，结合了：
    1. 初始残差 (Initial Residual): h_l = (1 - alpha) * A @ h_l + alpha * h_0
       - 保留初始特征，防止信息在多层传播后丢失
    2. 导向残差 (Guided Residual): h_l = theta * W @ h_l + (1 - theta) * h_{l-1}
       - theta = min(1, log(lambda / l + 1)) 控制残差强度
    3. 变体模式 (Variant): 使用 2*in_features 输入，模拟双向信息流

    原始论文: "Simple and Deep Graph Convolutional Networks"
    """

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super().__init__()
        self.variant = variant
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        """
        Args:
            input: 图节点特征 [N, in_features]
            adj: 归一化邻接矩阵 [N, N] (稀疏或密集)
            h0: 初始节点特征 [N, in_features_original]
            lamda: 控制 theta 的参数
            alpha: 初始残差系数
            l: 当前层编号
        Returns:
            output: [N, out_features]
        """
        theta = min(1, math.log(lamda / l + 1))
        hi = torch.spmm(adj, input) if adj.is_sparse else torch.mm(adj, input)

        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        output = theta * torch.mm(support, self.weight) + (1 - theta) * r

        if self.residual and input.size(-1) == self.out_features:
            output = output + input

        return output


class GCNIIBlock(nn.Module):
    """
    GCNII 块：单层图卷积 + 激活 + Dropout

    包含多个 GCNII 图卷积层，模拟原始论文中的深度 GCNII 架构。
    """

    def __init__(self, nlayers, nfeat, nhidden, dropout, lamda, alpha, variant):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(
                GraphConvolution(nhidden, nhidden, residual=True, variant=variant)
            )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nhidden))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class ProteinGraphEncoder(nn.Module):
    """
    蛋白质图编码器

    使用 GCNII 对 PPI 网络中的蛋白质节点进行编码。
    每个蛋白质作为一个图节点，其特征由预训练嵌入初始化，
    通过 GCNII 层在 PPI 网络上进行信息传播。
    """

    def __init__(self, protein_dim, hidden_dim=256, nlayers=4, dropout=0.3,
                 lamda=1.5, alpha=0.7, variant=True, pooling='mean'):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.input_proj = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gcnii = GCNIIBlock(
            nlayers=nlayers,
            nfeat=hidden_dim,
            nhidden=hidden_dim,
            dropout=dropout,
            lamda=lamda,
            alpha=alpha,
            variant=variant,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: 节点特征 [N, protein_dim]
            edge_index: 边索引 [2, E]
            batch: 批索引 (可选，用于图池化)
        Returns:
            out: [N, hidden_dim] 节点嵌入
        """
        h = self.input_proj(x)

        num_nodes = x.size(0)
        adj = self._build_adj(edge_index, num_nodes, x.device)

        h = self.gcnii(h, adj)

        h = self.output_proj(h)

        return h

    def _build_adj(self, edge_index, num_nodes, device):
        """
        从边索引构建归一化邻接矩阵

        归一化: A_norm = D^{-0.5} @ A @ D^{-0.5}
        """
        edge_weight = torch.ones(edge_index.size(1), device=device)

        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes, fill_value=1.0
        )

        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, row, edge_weight)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        adj = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))
        return adj.coalesce()


class InteractionModule(nn.Module):
    """
    交互特征构建模块

    将两个蛋白质的图嵌入构建为丰富的交互特征向量。
    特征构成:
    - concat: [hidden_dim * 2]
    - abs_diff: [hidden_dim]
    - product: [hidden_dim]
    总输出维度: hidden_dim * 4
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 4

    def forward(self, emb1, emb2):
        """
        Args:
            emb1: [B, hidden_dim]
            emb2: [B, hidden_dim]
        Returns:
            interaction_feat: [B, hidden_dim * 4]
        """
        concat_feat = torch.cat([emb1, emb2], dim=1)
        abs_diff_feat = torch.abs(emb1 - emb2)
        product_feat = emb1 * emb2

        interaction_feat = torch.cat(
            [concat_feat, abs_diff_feat, product_feat], dim=1
        )
        return interaction_feat


class GraphPPIS(nn.Module):
    """
    GraphPPIS 模型

    核心架构:
    1. GCNII 图编码器: 在 PPI 网络上进行图卷积，学习蛋白质的结构化嵌入
    2. 双通道编码: 两个蛋白质分别通过共享 GCNII 编码器得到图嵌入
    3. 交互特征构建: concat + abs_diff + product
    4. 分类头: 深度 MLP，输出二分类或多标签分类 logits

    关键设计:
    - GCNII: 初始残差 + 导向残差，有效缓解图神经网络过平滑问题
    - Shared encoder: 两个蛋白质共享权重，增强配对表示的泛化能力
    - 自环 + 对称归一化: 稳定图卷积过程

    Args:
        protein_dim: 蛋白质嵌入维度
        hidden_dim: GCNII 隐藏层维度 (默认 256)
        nlayers: GCNII 层数 (默认 4)
        dropout: Dropout 概率 (默认 0.3)
        num_classes: 类别数 (二分类=2, 多标签=标签数)
        task_type: 'binary' 或 'multilabel'
    """

    def __init__(self, protein_dim=1024, hidden_dim=256, nlayers=4,
                 dropout=0.3, lamda=1.5, alpha=0.7, variant=True,
                 pooling='mean', num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.encoder = ProteinGraphEncoder(
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            nlayers=nlayers,
            dropout=dropout,
            lamda=lamda,
            alpha=alpha,
            variant=variant,
            pooling=pooling,
        )

        self.interaction_module = InteractionModule(hidden_dim)
        interaction_output_dim = self.interaction_module.output_dim

        if task_type == 'multilabel':
            self.classifier = nn.Sequential(
                nn.Linear(interaction_output_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(interaction_output_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象 (PyG Data)，包含:
                - protein_x: [N, protein_dim] 蛋白质嵌入
                - edge_index: [2, E] 蛋白质相互作用网络边
            idx_batch: 批次数据 (p1_idx, p2_idx, labels)
                - p1_idx: [B] 蛋白质1索引
                - p2_idx: [B] 蛋白质2索引
                - labels: [B] 或 [B, num_classes]

        Returns:
            output: [B, num_classes] logits
        """
        p1_idx = idx_batch[0]
        p2_idx = idx_batch[1]

        device = next(self.parameters()).device
        if isinstance(p1_idx, torch.Tensor):
            p1_idx = p1_idx.to(device)
            p2_idx = p2_idx.to(device)
        else:
            p1_idx = torch.as_tensor(p1_idx, dtype=torch.long, device=device)
            p2_idx = torch.as_tensor(p2_idx, dtype=torch.long, device=device)

        protein_x = graph_or_none.protein_x
        edge_index = graph_or_none.edge_index

        protein_embeddings = self.encoder(protein_x, edge_index)

        p1_emb = protein_embeddings[p1_idx]
        p2_emb = protein_embeddings[p2_idx]

        interaction_feat = self.interaction_module(p1_emb, p2_emb)

        output = self.classifier(interaction_feat)
        return output


def GraphPPIS_Model(protein_dim=1024, hidden_dim=256, nlayers=4,
                    dropout=0.3, lamda=1.5, alpha=0.7, variant=True,
                    pooling='mean', num_classes=2, task_type='binary', **kwargs):
    """
    GraphPPIS 模型工厂函数（用于 model_manager）
    """
    return GraphPPIS(
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        nlayers=nlayers,
        dropout=dropout,
        lamda=lamda,
        alpha=alpha,
        variant=variant,
        pooling=pooling,
        num_classes=num_classes,
        task_type=task_type,
    )
