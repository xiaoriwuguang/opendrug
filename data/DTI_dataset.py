"""
DTI (Drug-Target Interaction) 分类数据集类
用于药物-蛋白质相互作用分类预测任务

与 DDI 的主要区别:
1. 配对数据: 药物-蛋白质 对, 而非 药物-药物 对
2. 标签类型: 二分类标签(0/1)
3. 双嵌入系统: 药物嵌入 + 蛋白质嵌入
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class DataLoadingModule:
    """DTI 数据加载模块"""

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
                # 替换 NaN/Inf 为 0
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
    def read_dti_pairs(matrix_path):
        """
        读取 DTI 数据文件

        Args:
            matrix_path: DTI 数据文件路径 (.tsv 格式)

        Returns:
            df: 处理后的数据框，包含 drug_id, protein_id, label 列
        """
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 DTI 数据文件: {matrix_path}")

        df = pd.read_csv(matrix_path, sep='\t', dtype=str)

        cols_lower = {c.lower(): c for c in df.columns}

        drug_col = None
        protein_col = None
        for c in df.columns:
            cl = c.lower()
            if 'drug' in cl or 'compound' in cl or 'molecule' in cl:
                drug_col = c
            if 'protein' in cl or 'target' in cl or 'gene' in cl:
                protein_col = c

        if drug_col is None or protein_col is None:
            if 'id1' in cols_lower and 'id2' in cols_lower:
                drug_col = cols_lower['id1']
                protein_col = cols_lower['id2']

        if drug_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 drug 列")
        if protein_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein 列")

        label_col = None
        for c in df.columns:
            cl = c.lower()
            if 'label' in cl or 'pred_label' in cl or 'interaction' in cl or 'class' in cl:
                label_col = c
                break

        if label_col is None:
            df['label'] = 1
        else:
            df['label'] = pd.to_numeric(df[label_col], errors='coerce').fillna(1).astype(int)

        df['drug_id'] = df[drug_col].astype(str)
        df['protein_id'] = df[protein_col].astype(str)

        return df[['drug_id', 'protein_id', 'label']]


class FeatureProcessingModule:
    """DTI 特征处理模块"""

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

        # 高斯噪声
        if args is not None and float(getattr(args, 'noise_std', 0.0)) > 0:
            feats = FeatureProcessingModule.add_gaussian_noise(feats, args.noise_std)

        # 稀疏丢弃
        if args is not None and float(getattr(args, 'sparse_drop_rate', 0.0)) > 0:
            feats = FeatureProcessingModule.sparse_dropout(feats, args.sparse_drop_rate)

        return torch.tensor(feats, dtype=torch.float32)

    @staticmethod
    def normalize_features(feats):
        """特征标准化"""
        std = feats.std(axis=0)
        std[std == 0] = 1.0
        return (feats - feats.mean(axis=0)) / std

    @staticmethod
    def add_gaussian_noise(feats, noise_std):
        """
        添加高斯噪声

        Args:
            feats: 原始特征矩阵
            noise_std: 噪声标准差

        Returns:
            添加噪声后的特征矩阵
        """
        noise = np.random.normal(0, float(noise_std), feats.shape).astype(np.float32)
        return feats + noise

    @staticmethod
    def sparse_dropout(feats, drop_rate):
        """
        稀疏丢弃

        Args:
            feats: 原始特征矩阵
            drop_rate: 丢弃比例

        Returns:
            稀疏化后的特征矩阵
        """
        mask = (np.random.rand(*feats.shape) > drop_rate).astype(np.float32)
        return feats * mask


class LabelNoiseModule:
    """DTI 标签噪声注入模块"""

    @staticmethod
    def add_label_noise_binary(labels, noise_ratio, random_seed=1):
        """
        二分类标签噪声注入

        Args:
            labels: 标签数组 (0/1)
            noise_ratio: 噪声比例
            random_seed: 随机种子

        Returns:
            添加噪声后的标签
        """
        if noise_ratio <= 0:
            return labels

        labels = labels.copy()
        rng = np.random.RandomState(random_seed)
        n_train = len(labels)
        flip_n = int(n_train * float(noise_ratio))
        idx_sel = rng.choice(n_train, size=flip_n, replace=False)

        for ii in idx_sel:
            labels[ii] = 1 - labels[ii]

        print(f"[Label Noise] 训练集标签噪声比例: {noise_ratio}, 影响样本数: {len(idx_sel)}")
        return labels

    @staticmethod
    def add_label_noise_asymmetric(labels, noise_ratio, random_seed=1):
        """
        非对称二分类标签噪声注入
        将正类(1)以一定概率flip为负类(0)，模拟更真实的噪声场景

        Args:
            labels: 标签数组 (0/1)
            noise_ratio: 正类被翻转的比例
            random_seed: 随机种子

        Returns:
            添加噪声后的标签
        """
        if noise_ratio <= 0:
            return labels

        labels = labels.copy()
        rng = np.random.RandomState(random_seed)

        pos_indices = np.where(labels == 1)[0]
        flip_n = int(len(pos_indices) * float(noise_ratio))
        flip_idx = rng.choice(pos_indices, size=flip_n, replace=False)
        labels[flip_idx] = 0

        print(f"[Label Noise] 非对称噪声: 正类翻转比例 {noise_ratio}, 影响样本数: {len(flip_idx)}")
        return labels


class EdgeNoiseModule:
    """DTI 图结构边噪声注入模块"""

    @staticmethod
    def add_edge_noise(triples, labels, noise_ratio, random_seed=1):
        """
        边噪声注入：随机删除和添加边

        Args:
            triples: 训练三元组 (drug_idx, protein_idx, _)
            labels: 训练标签
            noise_ratio: 噪声比例，用于计算删除/添加的边数
            random_seed: 随机种子

        Returns:
            添加噪声后的三元组和标签
        """
        if noise_ratio <= 0:
            return triples, labels

        triples = triples.copy()
        labels = labels.copy()
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
        num_drugs = int(triples[:, 0].max()) + 1
        num_proteins = int(triples[:, 1].max()) + 1

        # 计算原始数据集中正样本比例
        pos_ratio = labels.mean() if len(labels) > 0 else 0.5
        pos_label = 1
        neg_label = 0

        new_edges = []
        new_labels = []
        existing_edges = set(zip(triples[:, 0].tolist(), triples[:, 1].tolist()))

        attempts = 0
        max_attempts = num_add * 10
        while len(new_edges) < num_add and attempts < max_attempts:
            attempts += 1
            drug_idx = rng.randint(0, num_drugs)
            protein_idx = rng.randint(0, num_proteins)

            # 跳过已存在的边
            if (drug_idx, protein_idx) in existing_edges:
                continue

            # 随机分配标签（按原始正样本比例）
            new_label = pos_label if rng.random() < pos_ratio else neg_label
            new_edges.append([drug_idx, protein_idx, 0])
            new_labels.append(new_label)

        if new_edges:
            new_edges = np.array(new_edges, dtype=np.int64)
            new_labels = np.array(new_labels, dtype=np.int64)
            triples = np.concatenate([triples, new_edges], axis=0)
            labels = np.concatenate([labels, new_labels], axis=0)

        print(f"[Edge Noise] 边噪声比例: {noise_ratio}, 删除边数: {num_modify}, 添加边数: {len(new_edges)}, 最终边数: {len(triples)}")
        return triples, labels


class DataSplittingModule:
    """DTI 数据划分模块"""

    @staticmethod
    def split_data(triples, labels, val_ratio=0.1, test_ratio=0.2, random_seed=1):
        """数据集划分"""
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
        """基于蛋白质划分，确保测试集中的蛋白质不出现在训练/验证集中"""
        rng = np.random.RandomState(random_seed)
        proteins = np.unique(triples[:, 1])
        rng.shuffle(proteins)

        n_test = int(len(proteins) * test_ratio)
        n_val = int(len(proteins) * val_ratio)

        test_proteins = set(proteins[:n_test])
        val_proteins = set(proteins[n_test:n_test + n_val])
        train_proteins = set(proteins[n_test + n_val:])

        train_mask = np.array([p in train_proteins for p in triples[:, 1]])
        val_mask = np.array([p in val_proteins for p in triples[:, 1]])
        test_mask = np.array([p in test_proteins for p in triples[:, 1]])

        train_triples = triples[train_mask]
        val_triples = triples[val_mask]
        test_triples = triples[test_mask]

        train_labels = labels[train_mask]
        val_labels = labels[val_mask]
        test_labels = labels[test_mask]

        print(f"[DTI generalization] 训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")
        print(f"[DTI generalization] 训练蛋白质: {len(train_proteins)}, 验证蛋白质: {len(val_proteins)}, 测试蛋白质: {len(test_proteins)}")

        return (train_triples, train_labels,
                val_triples, val_labels,
                test_triples, test_labels)


class BaseDTIDataset(Dataset):
    """DTI 基础数据集类"""
    def __init__(self, triples, labels):
        self.entity1 = triples[:, 0]  # drug indices
        self.entity2 = triples[:, 1]  # protein indices
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.entity1[index], self.entity2[index], self.labels[index])


class DataLoaderCreationModule:
    """DataLoader 创建模块"""

    @staticmethod
    def create_dataloader_config(args):
        """创建 DataLoader 配置"""
        return {
            'batch_size': args.batch,
            'shuffle': False,
            'num_workers': int(getattr(args, 'workers', 0)),
            'drop_last': False,
            'pin_memory': False,
            'persistent_workers': False,
        }

    @staticmethod
    def create_dti_dataloaders(train_triples, train_labels,
                               val_triples, val_labels,
                               test_triples, test_labels, args):
        """创建 DTI DataLoader"""
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BaseDTIDataset(train_triples, train_labels),
                                  **{**params, 'shuffle': True})
        val_loader = DataLoader(BaseDTIDataset(val_triples, val_labels), **params)
        test_loader = DataLoader(BaseDTIDataset(test_triples, test_labels), **params)

        return train_loader, val_loader, test_loader


class DTIDataset:
    """
    DTI 数据集类

    功能特点:
    - 支持药物-蛋白质相互作用分类预测
    - 支持多种药物嵌入模态
    - 支持多种蛋白质嵌入模态
    - 二分类任务：预测是否存在相互作用
    - 支持按蛋白质划分的数据集划分（泛化实验）
    """

    def __init__(self, args):
        self.args = args
        self.data_o = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.drug_dim = 0
        self.protein_dim = 0
        self.num_drugs = 0
        self.num_proteins = 0

    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        """加载 DTI 数据"""
        print("=== DTI Dataset Loading ===")
        print(f"数据文件: {self.args.matrix_path}")
        print(f"药物嵌入: {self.args.drug_embedding_paths}")
        print(f"蛋白质嵌入: {self.args.protein_embedding_paths}")

        drug_id2vec, self.drug_dim = DataLoadingModule.read_embedding_pt(
            self.args.drug_embedding_paths
        )
        print(f"药物嵌入维度: {self.drug_dim}, 药物数量: {len(drug_id2vec)}")

        if self.args.protein_embedding_paths:
            protein_id2vec, self.protein_dim = DataLoadingModule.read_embedding_pt(
                self.args.protein_embedding_paths
            )
            print(f"蛋白质嵌入维度: {self.protein_dim}, 蛋白质数量: {len(protein_id2vec)}")
        else:
            protein_id2vec = {}
            self.protein_dim = 0
            print("警告: 未提供蛋白质嵌入")

        pairs_df = DataLoadingModule.read_dti_pairs(self.args.matrix_path)
        print(f"DTI 配对数量: {len(pairs_df)}")

        drug_list = sorted(set(pairs_df['drug_id']))
        protein_list = sorted(set(pairs_df['protein_id']))

        self.num_drugs = len(drug_list)
        self.num_proteins = len(protein_list)

        print(f"数据集中药物数: {self.num_drugs}, 蛋白质数: {self.num_proteins}")

        drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
        protein_id_to_index = {p: i for i, p in enumerate(protein_list)}

        drug_x = FeatureProcessingModule.build_feature_matrix(
            drug_list, drug_id2vec, self.drug_dim, self.args
        )

        if self.protein_dim > 0 and protein_id2vec:
            protein_x = FeatureProcessingModule.build_feature_matrix(
                protein_list, protein_id2vec, self.protein_dim, self.args
            )
        else:
            protein_x = torch.zeros(self.num_proteins, self.drug_dim, dtype=torch.float32)

        triples = np.asarray(
            [(drug_id_to_index[d], protein_id_to_index[p], 0)
             for d, p in zip(pairs_df['drug_id'], pairs_df['protein_id'])],
            dtype=np.int64
        )
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

        # 标签噪声注入
        if float(getattr(self.args, 'noise_ratio', 0.0)) > 0:
            noise_type = getattr(self.args, 'noise_type', 'symmetric')
            if noise_type == 'asymmetric':
                train_labels = LabelNoiseModule.add_label_noise_asymmetric(
                    train_labels, self.args.noise_ratio,
                    random_seed=getattr(self.args, 'seed', 1)
                )
            else:
                train_labels = LabelNoiseModule.add_label_noise_binary(
                    train_labels, self.args.noise_ratio,
                    random_seed=getattr(self.args, 'seed', 1)
                )

        self.train_loader, self.val_loader, self.test_loader = DataLoaderCreationModule.create_dti_dataloaders(
            train_triples, train_labels,
            val_triples, val_labels,
            test_triples, test_labels,
            self.args
        )

        edge_index_drug_protein = []
        for i, p_idx, _ in train_triples:
            edge_index_drug_protein.append([int(i), int(p_idx) + self.num_drugs])
            edge_index_drug_protein.append([int(p_idx) + self.num_drugs, int(i)])

        if edge_index_drug_protein:
            edge_index = torch.tensor(edge_index_drug_protein, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        total_dim = self.drug_dim + self.protein_dim

        combined_x = torch.zeros(self.num_drugs + self.num_proteins, total_dim, dtype=torch.float32)
        combined_x[:self.num_drugs, :self.drug_dim] = drug_x
        if self.protein_dim > 0:
            combined_x[self.num_drugs:, self.drug_dim:self.drug_dim + self.protein_dim] = protein_x

        self.data_o = Data(
            x=combined_x,
            edge_index=edge_index,
            drug_x=drug_x,
            protein_x=protein_x,
            num_drugs=self.num_drugs,
            num_proteins=self.num_proteins,
            drug_dim=self.drug_dim,
            protein_dim=self.protein_dim
        )

        self.args.drug_dim = self.drug_dim
        self.args.protein_dim = self.protein_dim
        self.args.num_drugs = self.num_drugs
        self.args.num_proteins = self.num_proteins
        self.args.total_dim = total_dim
        self.args.num_classes = 2

        print(f"组合特征维度: {total_dim}")
        print("=== DTI 数据加载完成 ===")

    def get_data_stats(self):
        """获取数据集统计信息"""
        stats = {
            'num_drugs': self.num_drugs,
            'num_proteins': self.num_proteins,
            'drug_dim': self.drug_dim,
            'protein_dim': self.protein_dim,
            'train_size': len(self.train_loader.dataset) if self.train_loader else 0,
            'val_size': len(self.val_loader.dataset) if self.val_loader else 0,
            'test_size': len(self.test_loader.dataset) if self.test_loader else 0,
        }
        return stats


# 为了保持向后兼容性
DTI_dataset = DTIDataset
