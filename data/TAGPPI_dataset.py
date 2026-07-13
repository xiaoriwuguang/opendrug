"""
TAGPPI 数据集类

复用 OpenDrug 的 PPIDataset 数据加载逻辑（蛋白质嵌入 + PPI 对），
TAGPPI 数据集在此基础上做最小适配：
- 节点特征: protein_x [num_proteins, protein_dim]
- 边索引: edge_index (PPI 网络)
- DataLoader 返回 (p1_idx, p2_idx, label)

支持 PPI 二分类和多标签分类。
"""

from data.PPI_dataset import PPIDataset
from data.PPI_dataset import (
    DataLoadingModule,
    FeatureProcessingModule,
    DataSplittingModule,
    BasePPIDataset,
    DataLoaderCreationModule,
    EdgeNoiseModule,
)
import numpy as np
import torch
from torch_geometric.data import Data
import os


class TAGPPI_Dataset:
    """
    TAGPPI 数据集

    复用 PPIDataset 的嵌入加载、数据划分逻辑，
    仅在图构建部分保持与 TAGPPI 模型兼容（PyG Data）。

    DataLoader 返回:
        (p1_idx [B], p2_idx [B], label [B] or [B, num_classes])
    """

    def __init__(self, args):
        self.args = args
        self.data_o = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.protein_dim = 0
        self.num_proteins = 0
        self.num_classes = 2
        self.task_type = 'binary'

    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        """加载 PPI 数据"""
        ppi_type = getattr(self.args, 'ppi_type', 'binary')
        print("=== TAGPPI Dataset Loading ===")
        print(f"PPI 类型: {ppi_type}")
        print(f"数据文件: {self.args.matrix_path}")
        print(f"蛋白质嵌入: {self.args.protein_embedding_paths}")

        protein_id2vec, self.protein_dim = DataLoadingModule.read_embedding_pt(
            self.args.protein_embedding_paths
        )
        print(f"蛋白质嵌入维度: {self.protein_dim}, 蛋白质数量: {len(protein_id2vec)}")

        if ppi_type == 'multilabel':
            pairs_df, labels_matrix, self.num_classes = DataLoadingModule.read_ppi_multilabel_pairs(
                self.args.matrix_path
            )
            self.task_type = 'multilabel'
            print(f"PPI 配对数量: {len(pairs_df)}, 标签类别数: {self.num_classes}")
            print(f"多标签分布: {np.sum(labels_matrix, axis=0).tolist()}")
        else:
            pairs_df = DataLoadingModule.read_ppi_binary_pairs(self.args.matrix_path)
            self.task_type = 'binary'
            self.num_classes = 2
            print(f"PPI 配对数量: {len(pairs_df)}")

        protein_list = sorted(set(pairs_df['p1_id']) | set(pairs_df['p2_id']))
        self.num_proteins = len(protein_list)
        print(f"数据集中蛋白质数: {self.num_proteins}")

        protein_id_to_index = {p: i for i, p in enumerate(protein_list)}

        protein_x = FeatureProcessingModule.build_feature_matrix(
            protein_list, protein_id2vec, self.protein_dim, self.args
        )

        triples = np.asarray(
            [(protein_id_to_index[str(p1)], protein_id_to_index[str(p2)])
             for p1, p2 in zip(pairs_df['p1_id'], pairs_df['p2_id'])],
            dtype=np.int64
        )

        if self.task_type == 'multilabel':
            labels = labels_matrix
        else:
            labels = pairs_df['label'].to_numpy().astype(np.int64)
            unique, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique, counts))
            print(f"标签分布: {label_dist}")

        if getattr(self.args, 'general', False):
            (train_triples, train_labels,
             val_triples, val_labels,
             test_triples, test_labels) = DataSplittingModule.split_data_by_protein(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )
        else:
            (train_triples, train_labels,
             val_triples, val_labels,
             test_triples, test_labels) = DataSplittingModule.split_data(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )

        print(f"训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")

        # 边噪声注入（同时处理标签）
        train_triples, train_labels = EdgeNoiseModule.add_edge_noise(
            train_triples, train_labels, self.args.noise_edge,
            random_seed=getattr(self.args, 'seed', 1)
        )

        params = DataLoaderCreationModule.create_dataloader_config(self.args)

        train_loader = DataLoaderCreationModule.create_ppi_dataloaders(
            train_triples, train_labels,
            val_triples, val_labels,
            test_triples, test_labels,
            self.args,
        )
        self.train_loader, self.val_loader, self.test_loader = train_loader

        # 构建蛋白质关系图（用训练集中的蛋白质对构建）
        edge_list = []
        for p1_idx, p2_idx in train_triples:
            edge_list.append([int(p1_idx), int(p2_idx)])
            edge_list.append([int(p2_idx), int(p1_idx)])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        self.data_o = Data(
            x=protein_x,
            edge_index=edge_index,
            protein_x=protein_x,
            num_proteins=self.num_proteins,
            protein_dim=self.protein_dim,
        )

        self.args.protein_dim = self.protein_dim
        self.args.num_proteins = self.num_proteins
        self.args.num_classes = self.num_classes
        self.args.task_type = self.task_type

        print(f"蛋白质特征维度: {self.protein_dim}")
        print(f"标签类别数: {self.num_classes}")
        print(f"任务类型: {self.task_type}")
        print("=== TAGPPI 数据加载完成 ===")

    def get_data_stats(self):
        """获取数据集统计信息"""
        stats = {
            'num_proteins': self.num_proteins,
            'protein_dim': self.protein_dim,
            'num_classes': self.num_classes,
            'task_type': self.task_type,
            'train_size': len(self.train_loader.dataset) if self.train_loader else 0,
            'val_size': len(self.val_loader.dataset) if self.val_loader else 0,
            'test_size': len(self.test_loader.dataset) if self.test_loader else 0,
        }
        return stats
