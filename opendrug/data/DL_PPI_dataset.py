"""
DL-PPI 数据集类

适配 OpenDrug Pipeline:
- 输入: OpenDrug 蛋白质嵌入 (protein_sequence_embeddings.pt 等)
- 适配: 将 1D 嵌入 reshape 为 [seq_len, 1]，供 Conv1d 使用
- 构建: 蛋白质图 (PPI 网络) + 序列特征

DL-PPI 原始设计使用氨基酸序列的向量表示.
适配策略:
- 嵌入维度 640: reshape -> [640, 1], 即 seq_len=640, in_channel=1
- 适配后的 in_feature=1 传入模型

支持 PPI 二分类和多标签分类。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class UnionFindSet:
    def __init__(self, m):
        self.roots = list(range(m))
        self.rank = [0] * m
        self.count = m

    def find(self, member):
        tmp = []
        while member != self.roots[member]:
            tmp.append(member)
            member = self.roots[member]
        for root in tmp:
            self.roots[root] = member
        return member

    def union(self, p, q):
        parentP = self.find(p)
        parentQ = self.find(q)
        if parentP != parentQ:
            if self.rank[parentP] > self.rank[parentQ]:
                self.roots[parentQ] = parentP
            elif self.rank[parentP] < self.rank[parentQ]:
                self.roots[parentP] = parentQ
            else:
                self.roots[parentQ] = parentP
                self.rank[parentP] -= 1
            self.count -= 1


class DL_PPI_DataLoader:
    """DL-PPI 数据加载器（处理嵌入文件）"""

    @staticmethod
    def read_embedding_pt(embedding_paths):
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
    def normalize_features(feats):
        std = feats.std(axis=0)
        std[std == 0] = 1.0
        return (feats - feats.mean(axis=0)) / std


class DataSplittingModule:
    """DL-PPI 数据划分（随机 + 图结构感知划分）"""

    @staticmethod
    def split_data_random(triples, labels, val_ratio=0.1, test_ratio=0.2, random_seed=1):
        rng = np.random.RandomState(random_seed)
        n_total = len(triples)
        perm = rng.permutation(n_total)

        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

        return (triples[train_idx], labels[train_idx],
                triples[val_idx], labels[val_idx],
                triples[test_idx], labels[test_idx])

    @staticmethod
    def split_by_protein(triples, labels, val_ratio=0.1, test_ratio=0.2, random_seed=1):
        """按蛋白质实体划分（蛋白质冷启动泛化）"""
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
            (p1 in val_proteins or p2 in val_proteins) or
            (p1 in train_proteins and p2 in val_proteins) or
            (p1 in val_proteins and p2 in train_proteins)
            for p1, p2 in zip(triples[:, 0], triples[:, 1])
        ])
        test_mask = np.array([
            (p1 in test_proteins or p2 in test_proteins)
            for p1, p2 in zip(triples[:, 0], triples[:, 1])
        ])

        return (triples[train_mask], labels[train_mask],
                triples[val_mask], labels[val_mask],
                triples[test_mask], labels[test_mask])


class EdgeNoiseModule:
    """DL-PPI 图结构边噪声注入模块"""

    @staticmethod
    def add_edge_noise(triples, labels, noise_ratio, random_seed=1):
        """
        边噪声注入：随机删除和添加边（PPI 任务：两端都是蛋白质）

        Args:
            triples: 训练三元组 (p1_idx, p2_idx)
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

            new_edges.append([p1_idx, p2_idx])
            new_labels.append(new_label)

        if new_edges:
            new_edges = np.array(new_edges, dtype=np.int64)
            new_labels = np.array(new_labels, dtype=labels.dtype)
            triples = np.concatenate([triples, new_edges], axis=0)
            labels = np.concatenate([labels, new_labels], axis=0)

        print(f"[Edge Noise] 边噪声比例: {noise_ratio}, 删除边数: {num_modify}, 添加边数: {len(new_edges)}, 最终边数: {len(triples)}")
        return triples, labels


class BaseDLPPIEdgeDataset(Dataset):
    """DL-PPI 边数据集"""
    def __init__(self, triples, labels):
        self.p1 = triples[:, 0]
        self.p2 = triples[:, 1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.p1[index], self.p2[index], self.labels[index])


class DL_PPIDataset:
    """
    DL-PPI 数据集

    适配 OpenDrug Pipeline:
    1. 从 protein_sequence_embeddings.pt 加载蛋白质嵌入
    2. 从 ppi1.tsv / ppi_multilabel.tsv 加载蛋白质对数据
    3. 构建 PyG Data 对象 (节点特征 + 边索引 + 边标签)
    4. 数据划分 (随机 / 按蛋白质实体)
    5. 将 1D 嵌入 reshape 为 [seq_len, 1] 供 Conv1d 使用

    图构建: 基于所有训练数据构建无向 PPI 图
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
        ppi_type = getattr(self.args, 'ppi_type', 'binary')
        print("=== DL-PPI Dataset Loading ===")
        print(f"PPI 类型: {ppi_type}")
        print(f"数据文件: {self.args.matrix_path}")
        print(f"蛋白质嵌入: {self.args.protein_embedding_paths}")

        protein_id2vec, self.protein_dim = DL_PPI_DataLoader.read_embedding_pt(
            self.args.protein_embedding_paths
        )
        print(f"蛋白质嵌入维度: {self.protein_dim}, 蛋白质数量: {len(protein_id2vec)}")

        if ppi_type == 'multilabel':
            pairs_df, labels_matrix, self.num_classes = self._read_ppi_multilabel_pairs(
                self.args.matrix_path
            )
            self.task_type = 'multilabel'
            print(f"PPI 配对数量: {len(pairs_df)}, 标签类别数: {self.num_classes}")
            print(f"多标签分布: {np.sum(labels_matrix, axis=0).tolist()}")
        else:
            pairs_df = self._read_ppi_binary_pairs(self.args.matrix_path)
            self.task_type = 'binary'
            self.num_classes = 2
            print(f"PPI 配对数量: {len(pairs_df)}")

        protein_list = sorted(set(pairs_df['p1_id']) | set(pairs_df['p2_id']))
        self.num_proteins = len(protein_list)
        print(f"数据集中蛋白质数: {self.num_proteins}")

        protein_id_to_index = {p: i for i, p in enumerate(protein_list)}

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
            print(f"标签分布: {dict(zip(unique, counts))}")

        if getattr(self.args, 'general', True):
            (train_triples, train_labels,
             val_triples, val_labels,
             test_triples, test_labels) = DataSplittingModule.split_by_protein(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )
        else:
            (train_triples, train_labels,
             val_triples, val_labels,
             test_triples, test_labels) = DataSplittingModule.split_data_random(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )

        print(f"训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")

        # 边噪声注入（同时处理标签）
        train_triples, train_labels = EdgeNoiseModule.add_edge_noise(
            train_triples, train_labels,
            float(getattr(self.args, 'noise_edge', 0.0)),
            random_seed=getattr(self.args, 'seed', 1)
        )

        self._build_data_objects(
            protein_list, protein_id2vec,
            triples, labels,
            train_triples, train_labels,
            val_triples, val_labels,
            test_triples, test_labels,
        )

        self.args.protein_dim = self.protein_dim
        self.args.num_proteins = self.num_proteins
        self.args.num_classes = self.num_classes
        self.args.task_type = self.task_type

        print(f"DL-PPI in_feature: 1 (由嵌入 reshape 而来)")
        print(f"标签类别数: {self.num_classes}")
        print(f"任务类型: {self.task_type}")
        print("=== DL-PPI 数据加载完成 ===")

    def _read_ppi_binary_pairs(self, matrix_path):
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 PPI 数据文件: {matrix_path}")

        df = pd.read_csv(matrix_path, sep='\t', dtype=str)
        p1_col = next((c for c in df.columns if 'protein1' in c.lower() or c.lower() == 'protein1_id'), None)
        p2_col = next((c for c in df.columns if 'protein2' in c.lower() or c.lower() == 'protein2_id'), None)

        if p1_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein1 列")
        if p2_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein2 列")

        label_col = next((c for c in df.columns if 'label' in c.lower()), None)
        if label_col is None:
            df['label'] = 1
        else:
            df['label'] = pd.to_numeric(df[label_col], errors='coerce').fillna(1).astype(int)

        df['p1_id'] = df[p1_col].astype(str)
        df['p2_id'] = df[p2_col].astype(str)
        return df[['p1_id', 'p2_id', 'label']]

    def _read_ppi_multilabel_pairs(self, matrix_path):
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 PPI 多标签数据文件: {matrix_path}")

        df = pd.read_csv(matrix_path, sep='\t', dtype=str)
        p1_col = next((c for c in df.columns if 'protein1' in c.lower() or c.lower() == 'protein1_id'), None)
        p2_col = next((c for c in df.columns if 'protein2' in c.lower() or c.lower() == 'protein2_id'), None)
        label_col = next((c for c in df.columns if c.lower() == 'label'), None)

        if label_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 label 列")

        df['p1_id'] = df[p1_col].astype(str)
        df['p2_id'] = df[p2_col].astype(str)

        def parse_multilabel(val):
            return [int(p.strip()) for p in str(val).split(',')]

        labels = df[label_col].apply(parse_multilabel)
        num_labels = len(labels.iloc[0])
        labels_matrix = np.array(labels.tolist(), dtype=np.int64)

        return df[['p1_id', 'p2_id']], labels_matrix, num_labels

    def _build_data_objects(self, protein_list, protein_id2vec,
                           all_triples, all_labels,
                           train_triples, train_labels,
                           val_triples, val_labels,
                           test_triples, test_labels):
        """
        构建 PyG Data 对象

        核心: 将 1D 蛋白质嵌入 reshape 为 [protein_dim, 1] -> [seq_len, in_channel=1]
        供 Conv1d + biGRU 处理
        """
        feats = []
        for p in protein_list:
            if p in protein_id2vec:
                v = protein_id2vec[p]
            elif p.upper() in protein_id2vec:
                v = protein_id2vec[p.upper()]
            elif p.lower() in protein_id2vec:
                v = protein_id2vec[p.lower()]
            else:
                v = np.zeros(self.protein_dim, dtype=np.float32)
            feats.append(v)

        feats = np.asarray(feats, dtype=np.float32)
        feats = DL_PPI_DataLoader.normalize_features(feats)
        feats = torch.tensor(feats, dtype=torch.float32)

        noise = getattr(self.args, 'noise_std', 0.0)
        if noise > 0:
            feats = feats + torch.randn_like(feats) * noise

        seq_len = self.protein_dim
        in_channel = 1
        feats_3d = feats.view(self.num_proteins, seq_len, in_channel)

        ppi_num = len(all_triples)
        if self.task_type == 'multilabel':
            edge_labels = torch.tensor(all_labels, dtype=torch.float)
        else:
            edge_labels = torch.tensor(all_labels, dtype=torch.long)

        train_edge_labels = torch.tensor(train_labels, dtype=torch.float)
        val_edge_labels = torch.tensor(val_labels, dtype=torch.float)
        test_edge_labels = torch.tensor(test_labels, dtype=torch.float)

        self.data_o = Data()
        self.data_o.x = feats_3d
        self.data_o.protein_x = feats
        self.data_o.num_proteins = self.num_proteins
        self.data_o.protein_dim = self.protein_dim
        self.data_o.num_classes = self.num_classes
        self.data_o.task_type = self.task_type

        self.data_o.train_mask = list(range(len(train_triples)))
        self.data_o.val_mask = list(range(len(train_triples), len(train_triples) + len(val_triples)))
        self.data_o.test_mask = list(range(len(train_triples) + len(val_triples),
                                           len(train_triples) + len(val_triples) + len(test_triples)))

        train_edge_idx = torch.tensor(train_triples, dtype=torch.long).t().contiguous()
        all_edge_idx = torch.tensor(all_triples, dtype=torch.long).t().contiguous()
        self.data_o.edge_index = all_edge_idx
        self.data_o.edge_index_train = train_edge_idx
        self.data_o.edge_attr_1 = edge_labels

        params = {
            'batch_size': self.args.batch,
            'shuffle': False,
            'num_workers': int(getattr(self.args, 'workers', 0)),
            'drop_last': False,
            'pin_memory': False,
            'persistent_workers': False,
        }

        self.train_loader = DataLoader(
            BaseDLPPIEdgeDataset(train_triples, train_edge_labels), **{**params, 'shuffle': True}
        )
        self.val_loader = DataLoader(
            BaseDLPPIEdgeDataset(val_triples, val_edge_labels), **params
        )
        self.test_loader = DataLoader(
            BaseDLPPIEdgeDataset(test_triples, test_edge_labels), **params
        )

    def get_data_stats(self):
        return {
            'num_proteins': self.num_proteins,
            'protein_dim': self.protein_dim,
            'num_classes': self.num_classes,
            'task_type': self.task_type,
            'train_size': len(self.train_loader.dataset) if self.train_loader else 0,
            'val_size': len(self.val_loader.dataset) if self.val_loader else 0,
            'test_size': len(self.test_loader.dataset) if self.test_loader else 0,
        }
