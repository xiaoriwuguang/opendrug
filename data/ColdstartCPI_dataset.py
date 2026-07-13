"""
ColdstartCPI 数据集类

为 ColdstartCPI 模型提供四种特征:
1. drug_g: 药物全局特征（拼接所有指定模态的嵌入向量）
2. drug_m: 药物分子序列特征矩阵（如果有矩阵格式的 drug_sequence）
3. protein_g: 蛋白质全局特征（拼接所有指定模态的嵌入向量）
4. protein_m: 蛋白质氨基酸序列特征矩阵（如果有矩阵格式的 protein_sequence）

支持 DTI 分类和 DTA 回归任务。

嵌入映射（可按需扩展）:
- 所有 drug_* 模态 -> drug_g（拼接）
- drug_sequence (矩阵格式) -> drug_m
- 所有 protein_* 模态 -> protein_g（拼接）
- protein_sequence (矩阵格式) -> protein_m
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class DataLoadingModule:
    """ColdstartCPI 数据加载模块"""

    @staticmethod
    def read_embedding_pt(embedding_paths):
        """读取嵌入文件列表，返回 id2vec 字典和总维度"""
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
    def read_sequence_embedding_pt(embedding_path):
        """读取序列嵌入文件（每个 ID 对应一个矩阵）"""
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError(f"未找到嵌入文件: {embedding_path}")

        data = torch.load(embedding_path, map_location='cpu', weights_only=False)

        if not isinstance(data, dict):
            raise ValueError(f"期望 pt 文件中是 dict，但得到的是 {type(data)}")

        id2matrix = {}
        feat_dim = None

        def _to_numpy(v):
            if hasattr(v, 'detach'):
                arr = v.detach().cpu().numpy().astype(np.float32)
            else:
                arr = np.asarray(v, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr

        for k, v in data.items():
            mat = _to_numpy(v)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            id2matrix[str(k)] = mat
            if feat_dim is None:
                feat_dim = mat.shape[1]

        return id2matrix, feat_dim

    @staticmethod
    def read_dti_pairs(matrix_path):
        """读取 DTI 数据文件"""
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

    @staticmethod
    def read_dta_pairs(matrix_path):
        """读取 DTA 数据文件"""
        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"未找到 DTA 数据文件: {matrix_path}")

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

        affinity_col = None
        for c in df.columns:
            cl = c.lower()
            if 'affinity' in cl or 'score' in cl or 'label' in cl or 'value' in cl:
                affinity_col = c
                break

        if affinity_col is None:
            df['affinity'] = 1.0
        else:
            df['affinity'] = pd.to_numeric(df[affinity_col], errors='coerce')
            df = df.dropna(subset=['affinity'])

        df['drug_id'] = df[drug_col].astype(str)
        df['protein_id'] = df[protein_col].astype(str)

        return df[['drug_id', 'protein_id', 'affinity']]


class DataSplittingModule:
    """数据划分模块"""

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

        return (triples[train_idx], labels[train_idx],
                triples[val_idx], labels[val_idx],
                triples[test_idx], labels[test_idx])

    @staticmethod
    def split_data_by_protein(triples, labels, val_ratio=0.1, test_ratio=0.2, random_seed=1):
        """基于蛋白质划分"""
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

        print(f"[ColdstartCPI] 训练集: {len(train_triples)}, 验证集: {len(val_triples)}, 测试集: {len(test_triples)}")
        print(f"[ColdstartCPI] 训练蛋白质: {len(train_proteins)}, 验证蛋白质: {len(val_proteins)}, 测试蛋白质: {len(test_proteins)}")

        return (train_triples, train_labels,
                val_triples, val_labels,
                test_triples, test_labels)


class EdgeNoiseModule:
    """ColdstartCPI 图结构边噪声注入模块"""

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
            new_label = 1 if rng.random() < pos_ratio else 0
            new_edges.append([drug_idx, protein_idx, 0])
            new_labels.append(new_label)

        if new_edges:
            new_edges = np.array(new_edges, dtype=np.int64)
            new_labels = np.array(new_labels, dtype=np.int64)
            triples = np.concatenate([triples, new_edges], axis=0)
            labels = np.concatenate([labels, new_labels], axis=0)

        print(f"[Edge Noise] 边噪声比例: {noise_ratio}, 删除边数: {num_modify}, 添加边数: {len(new_edges)}, 最终边数: {len(triples)}")
        return triples, labels


class BaseColdstartCPIDataset(Dataset):
    """ColdstartCPI 基础数据集类"""
    def __init__(self, triples, labels):
        self.entity1 = triples[:, 0]
        self.entity2 = triples[:, 1]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.entity1[index], self.entity2[index], self.labels[index])


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
    def create_dataloaders(train_triples, train_labels,
                          val_triples, val_labels,
                          test_triples, test_labels, args):
        params = DataLoaderCreationModule.create_dataloader_config(args)

        train_loader = DataLoader(BaseColdstartCPIDataset(train_triples, train_labels),
                                  **{**params, 'shuffle': True})
        val_loader = DataLoader(BaseColdstartCPIDataset(val_triples, val_labels), **params)
        test_loader = DataLoader(BaseColdstartCPIDataset(test_triples, test_labels), **params)

        return train_loader, val_loader, test_loader


class ColdstartCPI_Dataset:
    """
    ColdstartCPI 数据集类

    为 ColdstartCPI 模型提供四种特征:
    - drug_g: 药物全局特征向量（拼接所有 drug_* 模态嵌入）
    - drug_m: 药物序列特征矩阵（来自 drug_sequence 矩阵格式）
    - protein_g: 蛋白质全局特征向量（拼接所有 protein_* 模态嵌入）
    - protein_m: 蛋白质序列特征矩阵（来自 protein_sequence 矩阵格式）

    如果某个模态的嵌入为向量格式（而非矩阵），则只作为 drug_g/protein_g 拼接，
    不再额外作为 drug_m/protein_m 使用。
    """

    def __init__(self, args):
        self.args = args
        self.data_o = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.drug_g_dim = 0
        self.drug_m_dim = 0
        self.protein_g_dim = 0
        self.protein_m_dim = 0
        self.num_drugs = 0
        self.num_proteins = 0
        self.max_drug_seq = 100
        self.max_protein_seq = 1000

    def load_data(self, val_ratio=0.1, test_ratio=0.2):
        """加载 ColdstartCPI 所需的数据"""
        print("=== ColdstartCPI Dataset Loading ===")
        print(f"数据文件: {self.args.matrix_path}")
        print(f"嵌入目录: {self.args.embedding_dir}")

        task_type = getattr(self.args, 'task', 'train_xxxx')

        # 从 args 获取模态列表
        modality = getattr(self.args, 'modality', [])
        drug_mods = [m for m in modality if m.startswith('drug_')]
        protein_mods = [m for m in modality if m.startswith('protein_')]
        print(f"[INFO] 药物模态: {drug_mods}")
        print(f"[INFO] 蛋白质模态: {protein_mods}")

        # --- 1. 读取 DTI/DTA 配对 ---
        if task_type == 'dta':
            pairs_df = DataLoadingModule.read_dta_pairs(self.args.matrix_path)
        else:
            pairs_df = DataLoadingModule.read_dti_pairs(self.args.matrix_path)

        print(f"配对数量: {len(pairs_df)}")

        # --- 2. 收集所有药物和蛋白质 ID ---
        drug_ids = sorted(set(pairs_df['drug_id']))
        protein_ids = sorted(set(pairs_df['protein_id']))
        self.num_drugs = len(drug_ids)
        self.num_proteins = len(protein_ids)
        print(f"药物数: {self.num_drugs}, 蛋白质数: {self.num_proteins}")

        drug_id_to_index = {d: i for i, d in enumerate(drug_ids)}
        protein_id_to_index = {p: i for i, p in enumerate(protein_ids)}

        emb_map = self.args.embedding_map

        # --- 3. 加载药物嵌入 ---
        # drug_g: 拼接所有药物模态（全局特征向量）
        drug_g_embs = []
        for m in drug_mods:
            path = emb_map.get(m)
            if path and os.path.isfile(path):
                try:
                    id2vec, dim = DataLoadingModule.read_embedding_pt([path])
                    drug_g_embs.append((m, id2vec, dim))
                    print(f"  drug {m} -> drug_g, dim={dim}")
                except Exception as e:
                    print(f"  drug {m} 加载失败: {e}")

        # drug_m: drug_sequence 的矩阵格式（ndim > 1）
        drug_m_embs = []
        drug_m_path = emb_map.get('drug_sequence')
        if drug_m_path and os.path.isfile(drug_m_path):
            try:
                drug_m_id2mat, drug_m_dim = DataLoadingModule.read_sequence_embedding_pt(drug_m_path)
                first_val = next(iter(drug_m_id2mat.values()))
                if first_val.ndim > 1:
                    drug_m_embs.append(('drug_sequence', drug_m_id2mat, drug_m_dim))
                    print(f"  drug_sequence -> drug_m (matrix), dim={drug_m_dim}")
                else:
                    print(f"  drug_sequence 为向量格式，不作为 drug_m 使用")
            except Exception as e:
                print(f"  drug_sequence 加载失败: {e}")

        # --- 4. 加载蛋白质嵌入 ---
        # protein_g: 拼接所有蛋白质模态（全局特征向量）
        protein_g_embs = []
        for m in protein_mods:
            path = emb_map.get(m)
            if path and os.path.isfile(path):
                try:
                    id2vec, dim = DataLoadingModule.read_embedding_pt([path])
                    protein_g_embs.append((m, id2vec, dim))
                    print(f"  protein {m} -> protein_g, dim={dim}")
                except Exception as e:
                    print(f"  protein {m} 加载失败: {e}")

        # protein_m: protein_sequence 的矩阵格式（ndim > 1）
        protein_m_embs = []
        for m_name in ['protein_sequence', 'protein_structure']:
            protein_m_path = emb_map.get(m_name)
            if protein_m_path and os.path.isfile(protein_m_path):
                try:
                    protein_m_id2mat, protein_m_dim = DataLoadingModule.read_sequence_embedding_pt(protein_m_path)
                    first_val = next(iter(protein_m_id2mat.values()))
                    if first_val.ndim > 1:
                        protein_m_embs.append((m_name, protein_m_id2mat, protein_m_dim))
                        print(f"  {m_name} -> protein_m (matrix), dim={protein_m_dim}")
                        break
                    else:
                        print(f"  {m_name} 为向量格式，不作为 protein_m 使用")
                except Exception as e:
                    print(f"  {m_name} 加载失败: {e}")

        # --- 5. 构建特征张量 ---
        # drug_g: [num_drugs, total_drug_g_dim]
        self.drug_g_dim = sum(d[2] for d in drug_g_embs)
        drug_g_tensors = {}
        for m, id2vec, dim in drug_g_embs:
            for did, vec in id2vec.items():
                if did not in drug_g_tensors:
                    drug_g_tensors[did] = []
                drug_g_tensors[did].append(vec)

        drug_g_tensor = torch.zeros(self.num_drugs, self.drug_g_dim, dtype=torch.float32)
        for i, did in enumerate(drug_ids):
            if did in drug_g_tensors:
                vecs = drug_g_tensors[did]
                offset = 0
                for vec in vecs:
                    end = offset + vec.shape[0]
                    drug_g_tensor[i, offset:end] = torch.from_numpy(vec)
                    offset = end

        # drug_m: [num_drugs, max_drug_seq, drug_m_dim]
        self.drug_m_dim = drug_m_embs[0][2] if drug_m_embs else 0
        drug_m_tensor = torch.zeros(self.num_drugs, self.max_drug_seq, self.drug_m_dim, dtype=torch.float32)
        if drug_m_embs:
            _, drug_m_id2mat, _ = drug_m_embs[0]
            for i, did in enumerate(drug_ids):
                if did in drug_m_id2mat:
                    mat = drug_m_id2mat[did]
                    seq_len = min(mat.shape[0], self.max_drug_seq)
                    drug_m_tensor[i, :seq_len] = torch.from_numpy(mat[:seq_len])

        # protein_g: [num_proteins, total_protein_g_dim]
        self.protein_g_dim = sum(d[2] for d in protein_g_embs)
        protein_g_tensors = {}
        for m, id2vec, dim in protein_g_embs:
            for pid, vec in id2vec.items():
                if pid not in protein_g_tensors:
                    protein_g_tensors[pid] = []
                protein_g_tensors[pid].append(vec)

        protein_g_tensor = torch.zeros(self.num_proteins, self.protein_g_dim, dtype=torch.float32)
        for i, pid in enumerate(protein_ids):
            if pid in protein_g_tensors:
                vecs = protein_g_tensors[pid]
                offset = 0
                for vec in vecs:
                    end = offset + vec.shape[0]
                    protein_g_tensor[i, offset:end] = torch.from_numpy(vec)
                    offset = end

        # protein_m: [num_proteins, max_protein_seq, protein_m_dim]
        self.protein_m_dim = protein_m_embs[0][2] if protein_m_embs else 0
        protein_m_tensor = torch.zeros(self.num_proteins, self.max_protein_seq, self.protein_m_dim, dtype=torch.float32)
        if protein_m_embs:
            _, protein_m_id2mat, _ = protein_m_embs[0]
            for i, pid in enumerate(protein_ids):
                if pid in protein_m_id2mat:
                    mat = protein_m_id2mat[pid]
                    seq_len = min(mat.shape[0], self.max_protein_seq)
                    protein_m_tensor[i, :seq_len] = torch.from_numpy(mat[:seq_len])

        # --- 6. 构建三元组 ---
        triples = np.asarray(
            [(drug_id_to_index[d], protein_id_to_index[p], 0)
             for d, p in zip(pairs_df['drug_id'], pairs_df['protein_id'])],
            dtype=np.int64
        )

        if task_type == 'dta':
            labels = pairs_df['affinity'].to_numpy().astype(np.float32)
            print(f"亲和力范围: [{labels.min():.4f}, {labels.max():.4f}], 均值: {labels.mean():.4f}")
        else:
            labels = pairs_df['label'].to_numpy().astype(np.int64)
            unique, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique, counts))
            print(f"标签分布: {label_dist}")

        # --- 7. 数据划分 ---
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

        # --- 8. 创建 DataLoader ---
        self.train_loader, self.val_loader, self.test_loader = DataLoaderCreationModule.create_dataloaders(
            train_triples, train_labels,
            val_triples, val_labels,
            test_triples, test_labels,
            self.args
        )

        # --- 9. 构建图数据对象 ---
        edge_index_drug_protein = []
        for i, p_idx, _ in train_triples:
            edge_index_drug_protein.append([int(i), int(p_idx) + self.num_drugs])
            edge_index_drug_protein.append([int(p_idx) + self.num_drugs, int(i)])

        if edge_index_drug_protein:
            edge_index = torch.tensor(edge_index_drug_protein, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # --- 9b. Fallback: 确保所有维度 >= 1，避免 Linear 层 dim=0 报错 ---
        if self.drug_g_dim == 0:
            self.drug_g_dim = max(self.args.unify_num, 1)
            drug_g_tensor = torch.zeros(self.num_drugs, self.drug_g_dim, dtype=torch.float32)
            print(f"  [fallback] drug_g_dim=0 -> {self.drug_g_dim}")
        if self.drug_m_dim == 0:
            self.drug_m_dim = max(self.args.unify_num, 1)
            drug_m_tensor = torch.zeros(self.num_drugs, self.max_drug_seq, self.drug_m_dim, dtype=torch.float32)
            print(f"  [fallback] drug_m_dim=0 -> {self.drug_m_dim}")
        if self.protein_g_dim == 0:
            self.protein_g_dim = max(self.args.unify_num, 1)
            protein_g_tensor = torch.zeros(self.num_proteins, self.protein_g_dim, dtype=torch.float32)
            print(f"  [fallback] protein_g_dim=0 -> {self.protein_g_dim}")
        if self.protein_m_dim == 0:
            self.protein_m_dim = max(self.args.unify_num, 1)
            protein_m_tensor = torch.zeros(self.num_proteins, self.max_protein_seq, self.protein_m_dim, dtype=torch.float32)
            print(f"  [fallback] protein_m_dim=0 -> {self.protein_m_dim}")

        self.data_o = Data(
            edge_index=edge_index,
            drug_g=drug_g_tensor,
            drug_m=drug_m_tensor,
            protein_g=protein_g_tensor,
            protein_m=protein_m_tensor,
            num_drugs=self.num_drugs,
            num_proteins=self.num_proteins,
            drug_g_dim=self.drug_g_dim,
            drug_m_dim=self.drug_m_dim,
            protein_g_dim=self.protein_g_dim,
            protein_m_dim=self.protein_m_dim,
            max_drug_seq=self.max_drug_seq,
            max_protein_seq=self.max_protein_seq,
        )

        # --- 10. 回写维度信息到 args ---
        self.args.drug_g_dim = self.drug_g_dim
        self.args.drug_m_dim = self.drug_m_dim
        self.args.protein_g_dim = self.protein_g_dim
        self.args.protein_m_dim = self.protein_m_dim
        self.args.num_drugs = self.num_drugs
        self.args.num_proteins = self.num_proteins
        self.args.max_drug_seq = self.max_drug_seq
        self.args.max_protein_seq = self.max_protein_seq

        # unify_num 默认为 drug_g_dim
        if not hasattr(self.args, 'unify_num') or getattr(self.args, 'unify_num', 0) == 0:
            dim_candidates = [d for d in [self.drug_g_dim, self.drug_m_dim, self.protein_g_dim, self.protein_m_dim] if d > 0]
            self.args.unify_num = min(dim_candidates) if dim_candidates else 256

        print(f"特征维度: drug_g={self.drug_g_dim}, drug_m={self.drug_m_dim}, "
              f"protein_g={self.protein_g_dim}, protein_m={self.protein_m_dim}")
        print(f"unify_num: {self.args.unify_num}")
        print("=== ColdstartCPI 数据加载完成 ===")
