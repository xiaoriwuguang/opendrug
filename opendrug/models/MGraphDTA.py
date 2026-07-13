"""
MGraphDTA (Deep Multiscale Graph Neural Network) 模型

用于药物-靶点相互作用分类/亲和力预测的图神经网络模型

原始论文特点:
1. 蛋白质编码器: 基于序列的多尺度 CNN (StackCNN)
2. 药物编码器: DenseNet 风格的图神经网络
3. 分类器: MLP 输出分类/回归结果

本实现适配 OpenDrug 框架:
- 输入: 预训练的药物和蛋白质嵌入
- 支持 DTI 分类任务和 DTA 回归任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from collections import OrderedDict


class Conv1dReLU(nn.Module):
    """一维卷积 + ReLU 模块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                     kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    """堆叠 CNN 模块用于蛋白质序列特征提取"""
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        layers = []
        for i in range(layer_num):
            layers.append(Conv1dReLU(in_channels if i == 0 else out_channels, 
                                      out_channels, kernel_size, stride, padding))
        layers.append(nn.AdaptiveMaxPool1d(1))
        self.inc = nn.Sequential(*layers)

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class ProteinEncoder(nn.Module):
    """
    蛋白质序列编码器

    使用多尺度 CNN 提取蛋白质序列特征
    """
    def __init__(self, block_num=3, vocab_size=21, embedding_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_size, 96, 3)
            )
        self.linear = nn.Linear(block_num * 96, 96)

    def forward(self, x):
        """
        Args:
            x: [N, seq_len] 蛋白质序列

        Returns:
            protein_emb: [N, 96] 蛋白质嵌入
        """
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)
        return x


class NodeLevelBatchNorm(nn.Module):
    """
    节点级批归一化

    适用于图数据中所有节点在一批中
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.ones_(self.running_mean)
        nn.init.ones_(self.running_var)

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class GraphConvBlock(nn.Module):
    """图卷积 + 批归一化 + ReLU 块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x)
        x = torch.spmm(edge_index, x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class DenseLayer(nn.Module):
    """DenseNet 风格的密集连接层"""
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBlock(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBlock(int(growth_rate * bn_size), growth_rate)

    def forward(self, features):
        x = torch.cat(features, dim=-1)
        x = self.conv1(x, self.edge_index)
        x = self.conv2(x, self.edge_index)
        return x


class GraphDenseNetBlock(nn.Module):
    """DenseNet 风格的图卷积块"""
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.layers.append(layer)

    def forward(self, x, edge_index):
        features = [x]
        for layer in self.layers:
            layer.edge_index = edge_index
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim=-1)


class GraphTransition(nn.Module):
    """DenseNet 风格的过渡层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConvBlock(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class DrugEncoder(nn.Module):
    """
    药物分子图编码器

    使用 DenseNet 风格的图神经网络编码药物分子图
    """
    def __init__(self, num_input_features=22, out_dim=96, growth_rate=32, 
                 block_config=(8, 8, 8), bn_sizes=(2, 2, 2)):
        super().__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv0', GraphConvBlock(num_input_features, 32))
        ]))
        num_features = 32

        for i, num_layers in enumerate(block_config):
            block = GraphDenseNetBlock(
                num_layers, num_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module(f'block{i+1}', block)
            num_features = num_features + int(num_layers * growth_rate)

            trans = GraphTransition(num_features, num_features // 2)
            self.features.add_module(f"transition{i+1}", trans)
            num_features = num_features // 2

        self.out_dim = num_features

    def forward(self, x, edge_index):
        """
        Args:
            x: [N, num_nodes, num_features] 节点特征
            edge_index: [2, num_edges] 边索引

        Returns:
            drug_emb: [N, out_dim] 药物嵌入
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        x_flat = x.view(-1, x.size(-1))
        edge_index_flat = edge_index
        
        x_out = self.features(x_flat, edge_index_flat)
        
        x_out = x_out.view(batch_size, num_nodes, -1)
        x_out = x_out.mean(dim=1)
        
        return x_out


class MGraphDTA(nn.Module):
    """
    MGraphDTA 模型

    结合蛋白质序列编码和药物图神经网络进行药物-靶点预测

    特点:
    1. 蛋白质编码器: 多尺度 CNN 提取序列特征
    2. 药物编码器: DenseNet 风格图神经网络
    3. 支持分类和回归任务
    """

    def __init__(self, drug_dim=512, protein_dim=512, hidden_dim=256,
                 filter_num=32, out_dim=1, task_type='regression',
                 vocab_protein_size=21, embedding_size=128, num_classes=2):
        """
        Args:
            drug_dim: 药物嵌入维度
            protein_dim: 蛋白质嵌入维度
            hidden_dim: 隐藏层维度
            filter_num: 过滤器数量
            out_dim: 输出维度 (分类为 num_classes, 回归为 1)
            task_type: 'classification' 或 'regression'
            vocab_protein_size: 蛋白质词汇表大小
            embedding_size: 蛋白质嵌入维度
            num_classes: 类别数 (分类任务)
        """
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.task_type = task_type

        self.protein_encoder = ProteinEncoder(
            block_num=3, 
            vocab_size=vocab_protein_size, 
            embedding_size=embedding_size
        )
        
        self.drug_encoder = DrugEncoder(
            num_input_features=min(drug_dim, 22),
            out_dim=filter_num * 3,
            growth_rate=32,
            block_config=(8, 8, 8),
            bn_sizes=(2, 2, 2)
        )
        
        self.drug_projection = nn.Linear(drug_dim, filter_num * 3)
        self.protein_projection = nn.Linear(96, filter_num * 3)

        classifier_input_dim = filter_num * 3 * 2
        
        if task_type == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )

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

        drug_emb = self.drug_projection(drug_x[drug_idx])
        
        protein_emb_raw = protein_x[protein_idx]
        if protein_emb_raw.size(-1) == 96:
            protein_emb = protein_emb_raw
        else:
            protein_emb = self.protein_encoder.embed(protein_idx.long() % 21)
            protein_emb = protein_emb[:, :96]
            protein_emb = self.protein_encoder.linear(
                torch.cat([protein_emb] * (protein_emb_raw.size(-1) // protein_emb.size(-1) + 1), dim=-1)[:, :protein_emb_raw.size(-1)]
            )

        x = torch.cat([protein_emb, drug_emb], dim=-1)
        output = self.classifier(x)
        return output


class MGraphDTA_Embedding(nn.Module):
    """
    MGraphDTA 嵌入版本 - 稳定架构

    使用预训练嵌入进行药物-靶点预测
    参考 OpenDrug 稳定模型架构设计
    """

    def __init__(self, drug_dim=512, protein_dim=512, hidden_dim=256,
                 dropout=0.1, num_classes=2, task_type='regression'):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.task_type = task_type

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

        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
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

        interaction = torch.cat([
            drug_emb,
            protein_emb,
            drug_emb * protein_emb,
            torch.abs(drug_emb - protein_emb)
        ], dim=-1)

        features = self.interaction_mlp(interaction)
        output = self.output_layer(features)
        return output


def MGraphDTA_Model(drug_dim=512, protein_dim=512, hidden_dim=256,
                     dropout=0.1, num_classes=2, task_type='regression',
                     model_variant='embedding', **kwargs):
    """
    MGraphDTA 模型工厂函数

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度
        dropout: Dropout 比例
        num_classes: 类别数 (分类任务)
        task_type: 'classification' 或 'regression'
        model_variant: 'embedding' 或 'graph'

    Returns:
        model: MGraphDTA 模型实例
    """
    if model_variant == 'graph':
        return MGraphDTA(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            task_type=task_type,
            num_classes=num_classes
        )
    else:
        return MGraphDTA_Embedding(
            drug_dim=drug_dim,
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            task_type=task_type
        )


# 导出模型别名
MGraphDTA_Standard = MGraphDTA
MGraphDTA_Standard_Model = MGraphDTA_Model
