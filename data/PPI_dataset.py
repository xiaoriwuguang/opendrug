"""
PPI (Protein-Protein Interaction) 数据集类

支持两种任务模式:
1. PPI 二分类: 预测两个蛋白质是否存在相互作用 (ppi1.tsv)
2. PPI 多标签分类: 预测两个蛋白质在多个类别上的相互作用 (ppi_multilabel.tsv)

与 DTI 的主要区别:
1. 配对类型: 蛋白质-蛋白质 对, 而非 药物-蛋白质 对
2. 实体类型: 两端都是蛋白质, 共用同一个嵌入表
3. 对称性: interaction(p1, p2) == interaction(p2, p1)
4. 多标签: 输出为 multi-hot 向量
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class DataLoadingModule:
    """PPI 数据加载模块"""

    @staticmethod
    def read_embedding_pt(embedding_paths):
        """读取嵌入文件"""
        if isinstance(embedding_paths, str):
            embedding_paths = [embedding_paths]

        id2vec = {}
        total_dim = 0

        for path in embedding_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"未找到嵌入文件: {path}")

            data = torch.load(path, map_location='cpu', weights_only=False)

            if not isinstance(data, dict):
                raise ValueError(f"期望 pt 文件中是 dict，但得到的是 {type(data)}")

            def _to_numpy(v):
                if hasattr(v, 'detach'):
                    arr = v.detach().cpu().numpy().astype(np.float32)
                else:
                    arr = np.asarray(v, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                return arr

            current_id2vec = {str(k): _to_numpy(v) for k, v in data.items()}

            example_key = next(iter(current_id2vec))
            current_dim = current_id2vec[example_key].shape[0]
            total_dim += current_dim

            if not id2vec:
                id2vec = current_id2vec
            else:
                if set(id2vec.keys()) != set(current_id2vec.keys()):
                    raise ValueError(f"嵌入文件 {path} 的 ID 集与之前的嵌入不一致")
                for id_ in id2vec:
                    id2vec[id_] = np.concatenate([id2vec[id_], current_id2vec[id_]], axis=0)

        return id2vec, total_dim

    @staticmethod
    def read_ppi_binary_pairs(matrix_path):
        """
        读取 PPI 二分类数据文件

        格式: protein1_id  protein2_id  pred_label
        """
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 PPI 数据文件: {matrix_path}")

        df = pd.read_csv(matrix_path, sep='\t', dtype=str)
        cols_lower = {c.lower(): c for c in df.columns}

        p1_col = None
        p2_col = None
        for c in df.columns:
            cl = c.lower()
            if 'protein1' in cl or 'p1' in cl:
                p1_col = c
            elif 'protein2' in cl or 'p2' in cl:
                p2_col = c

        if p1_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein1 列")
        if p2_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein2 列")

        label_col = None
        for c in df.columns:
            cl = c.lower()
            if 'label' in cl or 'pred_label' in cl or 'interaction' in cl:
                label_col = c
                break

        if label_col is None:
            df['label'] = 1
        else:
            df['label'] = pd.to_numeric(df[label_col], errors='coerce').fillna(1).astype(int)

        df['p1_id'] = df[p1_col].astype(str)
        df['p2_id'] = df[p2_col].astype(str)

        return df[['p1_id', 'p2_id', 'label']]

    @staticmethod
    def read_ppi_multilabel_pairs(matrix_path):
        """
        读取 PPI 多标签数据文件

        格式: protein1_id  protein2_id  label (逗号分隔的多标签)
        """
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 PPI 多标签数据文件: {matrix_path}")

        df = pd.read_csv(matrix_path, sep='\t', dtype=str)
        cols_lower = {c.lower(): c for c in df.columns}

        p1_col = None
        p2_col = None
        for c in df.columns:
            cl = c.lower()
            if 'protein1' in cl or 'p1' in cl:
                p1_col = c
            elif 'protein2' in cl or 'p2' in cl:
                p2_col = c

        if p1_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein1 列")
        if p2_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein2 列")

        label_col = None
        for c in df.columns:
            cl = c.lower()
            if 'label' in cl:
                label_col = c
                break

        if label_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 label 列")

        df['p1_id'] = df[p1_col].astype(str)
        df['p2_id'] = df[p2_col].astype(str)

        def parse_multilabel(val):
            parts = str(val).split(',')
            return [int(p.strip()) for p in parts]

        labels = df[label_col].apply(parse_multilabel)
        num_labels = len(labels.iloc[0])
        labels_matrix = np.array(labels.tolist(), dtype=np.int64)

        return df[['p1_id', 'p2_id']], labels_matrix, num_labels


class FeatureProcessingModule:
    """PPI 特征处理模块"""

    @staticmethod
    def build_feature_matrix(entity_list, id2vec, emb_dim, args=None):
        """构建特征矩阵"""
        feats, miss = [], 0
        for e in entity_list:
            if e in id2vec:
                v = id2vec[e]
            elif e.upper() in id2vec:
                v = id2vec[e.upper()]
            elif e.lower() in id2vec:
                v = id2vec[e.lower()]
            else:
                miss += 1
                v = np.zeros(emb_dim, dtype=np.float32)
            feats.append(v)

        feats = np.asarray(feats, dtype=np.float32)
        feats = FeatureProcessingModule.normalize_features(feats)

        if args is not None and getattr(args, 'noise_std', 0.0) > 0:
            noise = np.random.normal(0, args.noise_std, feats.shape).astype(np.float32)
            feats = feats + noise

        return torch.tensor(feats, dtype=torch.float32)

    @staticmethod
    def normalize_features(feats):
        """特征标准化"""
        std = feats.std(axis=0)
        std[std == 0] = 1.0
        return (feats - feats.mean(axis=0)) / std


class EdgeNoiseModule:
    """PPI 图结构边噪声注入模块"""

    @staticmethod
    def add_edge_noise(triples, labels, noise_ratio, random_seed=1):
        """
        边噪声注入：随机删除和添加边（PPI 任务：两端都是蛋白质）

        Args:
            triples: 训练三元组 (p1_idx, p2_idx, _)
            labels: 训练标签（可以是1D数组或2D多标签矩阵）
            noise_ratio: 噪声比例，用于计算删除/添加的边数
            random_seed: 随机种子

        Returns:
            添加噪声后的三元组和标签
        """
        if noise_ratio <= 0:
            return triples, labels

        triples = triples.copy()
        rng = np.random.RandomState(random_seed)
        n_edges = len(triples)

        # 计算要删除和添加的边数
        num_modify = int(n_edges * float(noise_ratio))

        # 1. 随机删除边（同时删除对应的标签）
        if num_modify > 0:
            del_indices = rng.choice(n_edges, size=min(num_modify, n_edges), replace=False)
            keep_mask = np.ones(n_edges, dtype=bool)
            keep_mask[del_indices] = False
            triples = triples[keep_mask]
            labels = labels[keep_mask]
            n_edges = len(triples)

        # 2. 随机添加边（使用真实标签）
        num_add = num_modify
        num_proteins = max(int(triples[:, 0].max()), int(triples[:, 1].max())) + 1

        # 判断是否为多标签模式
        is_multilabel = len(labels.shape) > 1 and labels.shape[1] > 1

        new_edges = []
        new_labels = []
        existing_edges = set(zip(triples[:, 0].tolist(), triples[:, 1].tolist()))

        attempts = 0
        max_attempts = num_add * 10
        while len(new_edges) < num_add and attempts < max_attempts:
            attempts += 1
            p1_idx = rng.randint(0, num_proteins)
            p2_idx = rng.randint(0, num_proteins)
            if p1_idx == p2_idx:  # 排除自环
                continue
            if (p1_idx, p2_idx) in existing_edges:  # 跳过已存在的边
                continue

            # 生成新标签
            if is_multilabel:
                # 多标签：按各标签类别的原始比例生成
                new_label = (rng.random(labels.shape[1]) < labels.mean(axis=0)).astype(int)
            else:
                # 二分类：按原始正样本比例生成
                pos_ratio = labels.mean() if len(labels) > 0 else 0.5
                new_label = 1 if rng.random() < pos_ratio else 0

            new_edges.append([p1_idx, p2_idx, 0])
            new_labels.append(new_label)

        if new_edges:
            new_edges = np.array(new_edges, dtype=np.int64)
            new_labels = np.array(new_labels, dtype=labels.dtype)
            triples = np.concatenate([triples, new_edges], axis=0)
            labels = np.concatenate([labels, new_labels], axis=0)

        print(f"[Edge Noise] 边噪声比例: {noise_ratio}, 删除边数: {num_modify}, 添加边数: {len(new_edges)}, 最终边数: {len(triples)}")
        return triples, labels


class DataSplittingModule:
    """PPI 数据划分模块"""

    @staticmethod
    def split_data(triples, labels, val_ratio=0.1, test_ratio=0.2, random_seed=1):
        """随机划分数据集"""
        rng = np.random.RandomState(random_seed)
        n_total = len(triples)
        perm = rng.permutation(n_total)

        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

        train_triples = triples[train_idx]
        val_triples = triples[val_idx]
        test_triples = triples[test_idx]

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        return (train_triples, train_labels,
                val_triples, val_labels,
                test_triples, test_labels)

    @staticmethod
    def split_data_by_protein(triples, labels, val_ratio=0.1, test_ratio=0.2, random_seed=1):
        """基于蛋白质实体划分（确保测试集中的蛋白质不出现在训练/验证集中）"""
        rng = np.random.RandomState(random_seed)

        p1_set = set(triples[:, 0])
        p2_set = set(triples[:, 1])
        all_proteins = sorted(p1_set | p2_set)
        rng.shuffle(all_proteins)

        n_test = int(len(all_proteins) * test_ratio)
        n_val = int(len(all_proteins) * val_ratio)

        test_proteins = set(all_proteins[:n_test])
        val_proteins = set(all_proteins[n_test:n_test + n_val])
        train_proteins = set(all_proteins[n_test + n_val:])

        train_mask = np.array([
            (p1 in train_proteins and p2 in train_proteins)
            for p1, p2 in zip(triples[:, 0], triples[:, 1])
        ])
        val_mask = np.array([
            (p1 in val_proteins and p2 in val_proteins) or
            (p1 in train_proteins and p2 in val_proteins) or
            (p1 in val_proteins and p2 in train_proteins)
            for p1, p2 in zip(triples[:, 0], triples[:, 1])
        ])
        test_mask = np.array([
            (p1 in test_proteins or p2 in test_proteins)
            for p1, p2 in zip(triples[:, 0], triples[:, 1])
        ])

        train_triples = triples[train_mask]
        val_triples = triples[val_mask]
        test_triples = triples[test_mask]

        train_labels = labels[train_mask]
        val_labels = labels[val_mask]
        test_labels = labels[test_mask]

        print(f"[PPI generalization] 训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")
        print(f"[PPI generalization] 训练蛋白质: {len(train_proteins)}, 验证蛋白质: {len(val_proteins)}, 测试蛋白质: {len(test_proteins)}")

        return (train_triples, train_labels,
                val_triples, val_labels,
                test_triples, test_labels)


class BasePPIDataset(Dataset):
    """PPI 基础数据集类"""
    def __init__(self, triples, labels):
        self.p1 = triples[:, 0]
        self.p2 = triples[:, 1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.p1[index], self.p2[index], self.labels[index])


class DataLoaderCreationModule:
    """DataLoader 创建模块"""

    @staticmethod
    def create_dataloader_config(args):
        return {
            'batch_size': args.batch,
            'shuffle': False,
            'num_workers': int(getattr(args, 'workers', 0)),
            'drop_last': False,
            'pin_memory': False,
            'persistent_workers': False,
        }

    @staticmethod
    def create_ppi_dataloaders(train_triples, train_labels,
                                val_triples, val_labels,
                                test_triples, test_labels, args):
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BasePPIDataset(train_triples, train_labels),
                                  **{**params, 'shuffle': True})
        val_loader = DataLoader(BasePPIDataset(val_triples, val_labels), **params)
        test_loader = DataLoader(BasePPIDataset(test_triples, test_labels), **params)

        return train_loader, val_loader, test_loader


class PPIDataset:
    """
    PPI 数据集类

    支持:
    - PPI 二分类: ppi_binary 矩阵, 标签为 0/1
    - PPI 多标签分类: ppi_multilabel 矩阵, 标签为 multi-hot 向量
    - 蛋白质嵌入（支持多种模态拼接）
    - 按蛋白质实体划分（泛化实验）
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
        print("=== PPI Dataset Loading ===")
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

        if getattr(self.args, 'general', True):
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

        self.train_loader, self.val_loader, self.test_loader = DataLoaderCreationModule.create_ppi_dataloaders(
            train_triples, train_labels,
            val_triples, val_labels,
            test_triples, test_labels,
            self.args
        )

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
        print("=== PPI 数据加载完成 ===")

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
