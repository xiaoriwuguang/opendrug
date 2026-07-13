"""
DTA (Drug-Target Affinity) 数据集类
用于药物-蛋白质亲和力预测任务

与 DDI 的主要区别:
1. 配对数据: 药物-蛋白质 对, 而非 药物-药物 对
2. 标签类型: 亲和力分数(连续值) vs 关系类型(离散类别)
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
    """DTA 数据加载模块"""

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
    def read_dta_pairs(matrix_path):
        """
        读取 DTA 配对数据

        Args:
            matrix_path: DTA 数据文件路径 (.tsv 格式)

        Returns:
            df: 处理后的数据框，包含 drug_id, protein_id, affinity 列
        """
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 DTA 数据文件: {matrix_path}")

        df = pd.read_csv(matrix_path, sep='\t', dtype=str)

        # 标准化列名
        cols_lower = {c.lower(): c for c in df.columns}

        # 查找 drug/protein 相关列
        drug_col = None
        protein_col = None
        for c in df.columns:
            cl = c.lower()
            if 'drug' in cl or 'compound' in cl or 'molecule' in cl:
                drug_col = c
            if 'protein' in cl or 'target' in cl or 'gene' in cl:
                protein_col = c

        # 如果没找到，尝试使用 id1/id2
        if drug_col is None or protein_col is None:
            if 'id1' in cols_lower and 'id2' in cols_lower:
                drug_col = cols_lower['id1']
                protein_col = cols_lower['id2']

        if drug_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 drug 列")
        if protein_col is None:
            raise KeyError(f"文件 {matrix_path} 缺少 protein 列")

        # 查找 affinity 列
        affinity_col = None
        for c in df.columns:
            cl = c.lower()
            if 'affinity' in cl or 'score' in cl or 'label' in cl or 'value' in cl:
                affinity_col = c
                break

        # 如果没找到 affinity 列，默认为 1（二分类）
        if affinity_col is None:
            df['affinity'] = 1.0
        else:
            df['affinity'] = pd.to_numeric(df[affinity_col], errors='coerce')
            df = df.dropna(subset=['affinity'])

        df['drug_id'] = df[drug_col].astype(str)
        df['protein_id'] = df[protein_col].astype(str)

        return df[['drug_id', 'protein_id', 'affinity']]


class FeatureProcessingModule:
    """DTA 特征处理模块"""

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

        # 特征标准化
        feats = FeatureProcessingModule.normalize_features(feats)

        # 添加高斯噪声
        if args is not None and getattr(args, 'noise_std', 0.0) > 0:
            noise = np.random.normal(0, args.noise_std, feats.shape).astype(np.float32)
            feats = feats + noise

        return torch.tensor(feats, dtype=torch.float32)

    @staticmethod
    def normalize_features(feats):
        """特征标准化"""
        return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)


class DataSplittingModule:
    """DTA 数据划分模块"""

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

        # 获取所有蛋白质
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

        print(f"[DTA generalization] 训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")
        print(f"[DTA generalization] 训练蛋白质: {len(train_proteins)}, 验证蛋白质: {len(val_proteins)}, 测试蛋白质: {len(test_proteins)}")

        return (train_triples, train_labels,
                val_triples, val_labels,
                test_triples, test_labels)


class BaseDTADataset(Dataset):
    """DTA 基础数据集类"""
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
    def create_dta_dataloaders(train_triples, train_labels,
                               val_triples, val_labels,
                               test_triples, test_labels, args):
        """创建 DTA DataLoader"""
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BaseDTADataset(train_triples, train_labels),
                                  **{**params, 'shuffle': True})
        val_loader = DataLoader(BaseDTADataset(val_triples, val_labels), **params)
        test_loader = DataLoader(BaseDTADataset(test_triples, test_labels), **params)

        return train_loader, val_loader, test_loader


class DTADataset:
    """
    DTA 数据集类

    功能特点:
    - 支持药物-蛋白质亲和力预测
    - 支持多种药物嵌入模态
    - 支持多种蛋白质嵌入模态
    - 回归任务：预测亲和力分数
    - 支持按蛋白质划分的数据集划分（泛化实验）
    """

    def __init__(self, args):
        self.args = args
        self.data_o = None  # 包含药物特征、蛋白质特征、图结构
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # 存储嵌入维度信息
        self.drug_dim = 0
        self.protein_dim = 0
        self.num_drugs = 0
        self.num_proteins = 0

    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        """加载 DTA 数据"""
        print("=== DTA Dataset Loading ===")
        print(f"数据文件: {self.args.matrix_path}")
        print(f"药物嵌入: {self.args.drug_embedding_paths}")
        print(f"蛋白质嵌入: {self.args.protein_embedding_paths}")

        # 1. 读取药物嵌入
        drug_id2vec, self.drug_dim = DataLoadingModule.read_embedding_pt(
            self.args.drug_embedding_paths
        )
        print(f"药物嵌入维度: {self.drug_dim}, 药物数量: {len(drug_id2vec)}")

        # 2. 读取蛋白质嵌入
        if self.args.protein_embedding_paths:
            protein_id2vec, self.protein_dim = DataLoadingModule.read_embedding_pt(
                self.args.protein_embedding_paths
            )
            print(f"蛋白质嵌入维度: {self.protein_dim}, 蛋白质数量: {len(protein_id2vec)}")
        else:
            protein_id2vec = {}
            self.protein_dim = 0
            print("警告: 未提供蛋白质嵌入")

        # 3. 读取 DTA 配对数据
        pairs_df = DataLoadingModule.read_dta_pairs(self.args.matrix_path)
        print(f"DTA 配对数量: {len(pairs_df)}")

        # 4. 构建药物和蛋白质列表
        drug_list = sorted(set(pairs_df['drug_id']))
        protein_list = sorted(set(pairs_df['protein_id']))

        self.num_drugs = len(drug_list)
        self.num_proteins = len(protein_list)

        print(f"数据集中药物数: {self.num_drugs}, 蛋白质数: {self.num_proteins}")

        # 5. 构建 ID 到索引的映射
        drug_id_to_index = {d: i for i, d in enumerate(drug_list)}
        protein_id_to_index = {p: i for i, p in enumerate(protein_list)}

        # 6. 构建特征矩阵
        drug_x = FeatureProcessingModule.build_feature_matrix(
            drug_list, drug_id2vec, self.drug_dim, self.args
        )

        if self.protein_dim > 0 and protein_id2vec:
            protein_x = FeatureProcessingModule.build_feature_matrix(
                protein_list, protein_id2vec, self.protein_dim, self.args
            )
        else:
            protein_x = torch.zeros(self.num_proteins, self.drug_dim, dtype=torch.float32)

        # 7. 构建三元组和标签
        triples = np.asarray(
            [(drug_id_to_index[d], protein_id_to_index[p], 0)
             for d, p in zip(pairs_df['drug_id'], pairs_df['protein_id'])],
            dtype=np.int64
        )
        labels = pairs_df['affinity'].to_numpy().astype(np.float32)

        print(f"亲和力范围: [{labels.min():.4f}, {labels.max():.4f}], 均值: {labels.mean():.4f}")

        # 8. 数据划分
        if getattr(self.args, 'general', True):
            # 按蛋白质划分（泛化实验）
            (train_triples, train_labels,
             val_triples, val_labels,
             test_triples, test_labels) = DataSplittingModule.split_data_by_protein(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )
        else:
            # 随机划分
            (train_triples, train_labels,
             val_triples, val_labels,
             test_triples, test_labels) = DataSplittingModule.split_data(
                triples, labels, val_ratio, test_ratio, getattr(self.args, 'seed', 1)
            )

        print(f"训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")

        # 9. 创建 DataLoader
        self.train_loader, self.val_loader, self.test_loader = DataLoaderCreationModule.create_dta_dataloaders(
            train_triples, train_labels,
            val_triples, val_labels,
            test_triples, test_labels,
            self.args
        )

        # 10. 构建图结构（基于训练集中的药物-蛋白质交互）
        # 创建一个异构图：药物节点和蛋白质节点
        edge_index_drug_protein = []
        for i, p_idx, _ in train_triples:
            edge_index_drug_protein.append([int(i), int(p_idx) + self.num_drugs])
            edge_index_drug_protein.append([int(p_idx) + self.num_drugs, int(i)])

        if edge_index_drug_protein:
            edge_index = torch.tensor(edge_index_drug_protein, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 存储药物和蛋白质的总特征维度
        total_dim = self.drug_dim + self.protein_dim

        # 将药物和蛋白质特征拼接（用于 GNN）
        # 药物特征 padding 到与蛋白质相同的索引空间
        combined_x = torch.zeros(self.num_drugs + self.num_proteins, total_dim, dtype=torch.float32)
        combined_x[:self.num_drugs, :self.drug_dim] = drug_x
        if self.protein_dim > 0:
            combined_x[self.num_drugs:, self.drug_dim:self.drug_dim + self.protein_dim] = protein_x

        # 存储数据集信息
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

        # 更新 args 中的维度信息
        self.args.drug_dim = self.drug_dim
        self.args.protein_dim = self.protein_dim
        self.args.num_drugs = self.num_drugs
        self.args.num_proteins = self.num_proteins
        self.args.total_dim = total_dim

        print(f"组合特征维度: {total_dim}")
        print("=== DTA 数据加载完成 ===")

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
DTA_dataset = DTADataset
