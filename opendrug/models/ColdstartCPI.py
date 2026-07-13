"""
ColdstartCPI 模型

将 baseline/ColdstartCPI 的模型迁移到 opendrug pipeline。
核心设计：
1. 药物和蛋白质各有两个输入分支：全局特征向量 + 序列/图特征矩阵
2. 使用 Transformer Encoder 层建模药物-蛋白质交互
3. 支持 DTI 分类和 DTA 回归任务

与原始 ColdstartCPI 的区别：
- 使用 opendrug 的嵌入作为输入（替换原始 Mol2Vec/ProtTrans）
- 支持可配置的 unify_num（统一表示维度）
- 适配 opendrug 的 trainer 和 pipeline 接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColdstartCPI(nn.Module):
    """
    ColdstartCPI 主模型

    药物输入:
        drug_g: 全局特征向量 [B, drug_g_dim]
        drug_m: 分子序列/图特征矩阵 [B, drug_seq_len, drug_m_dim]

    蛋白质输入:
        protein_g: 全局特征向量 [B, protein_g_dim]
        protein_m: 氨基酸序列特征矩阵 [B, protein_seq_len, protein_m_dim]

    处理流程:
        1. c_g_unit / c_m_unit / p_g_unit / p_m_unit: 投影到 unify_num
        2. 拼接: [drug_g, drug_m, protein_g, protein_m] -> Transformer 输入
        3. Interacting_Layer: TransformerEncoder 建模交互
        4. 取 drug_g 和 protein_g 对应的输出向量
        5. predict_layer: 输出分类/回归预测
    """

    def __init__(self,
                 drug_g_dim=512,
                 drug_m_dim=512,
                 protein_g_dim=512,
                 protein_m_dim=512,
                 unify_num=256,
                 head_num=4,
                 dropout=0.1,
                 task_type='classification',
                 num_classes=2,
                 max_drug_seq=100,
                 max_protein_seq=1000):
        """
        Args:
            drug_g_dim: 药物全局特征维度（来自 opendrug 嵌入拼接）
            drug_m_dim: 药物序列特征维度
            protein_g_dim: 蛋白质全局特征维度
            protein_m_dim: 蛋白质序列特征维度
            unify_num: 统一表示维度（所有投影的目标维度）
            head_num: Transformer 层头数
            dropout: Dropout 比率
            task_type: 'classification' 或 'regression'
            num_classes: 分类类别数（回归时为 1）
            max_drug_seq: 药物序列最大长度
            max_protein_seq: 蛋白质序列最大长度
        """
        super().__init__()

        self.unify_num = unify_num
        self.task_type = task_type
        self.num_classes = num_classes
        self.max_drug_seq = max_drug_seq
        self.max_protein_seq = max_protein_seq

        # 药物全局特征投影
        self.c_g_unit = nn.Sequential(
            nn.Linear(drug_g_dim, unify_num),
            nn.PReLU(),
            nn.Linear(unify_num, unify_num),
            nn.PReLU()
        )

        # 药物序列特征投影
        self.c_m_unit = nn.Sequential(
            nn.Linear(drug_m_dim, unify_num),
            nn.PReLU(),
            nn.Linear(unify_num, unify_num),
            nn.PReLU()
        )

        # 蛋白质全局特征投影
        self.p_g_unit = nn.Sequential(
            nn.Linear(protein_g_dim, unify_num),
            nn.PReLU(),
            nn.Linear(unify_num, unify_num),
            nn.PReLU()
        )

        # 蛋白质序列特征投影
        self.p_m_unit = nn.Sequential(
            nn.Linear(protein_m_dim, unify_num),
            nn.PReLU(),
            nn.Linear(unify_num, unify_num),
            nn.PReLU()
        )

        # Transformer Encoder 层建模交互
        self.Interacting_Layer = nn.TransformerEncoderLayer(
            unify_num, head_num, batch_first=True, dropout=dropout
        )

        # 分类/回归输出层
        if task_type == 'classification':
            self.predict_layer = nn.Sequential(
                nn.Linear(unify_num * 2, unify_num * 2),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(unify_num * 2, num_classes),
            )
        else:
            self.predict_layer = nn.Sequential(
                nn.Linear(unify_num * 2, unify_num * 2),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(unify_num * 2, 1),
            )

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象，包含:
                - drug_g: 药物全局特征 [num_drugs, drug_g_dim]
                - drug_m: 药物序列特征 [num_drugs, max_drug_seq, drug_m_dim]
                - protein_g: 蛋白质全局特征 [num_proteins, protein_g_dim]
                - protein_m: 蛋白质序列特征 [num_proteins, max_protein_seq, protein_m_dim]
            idx_batch: 批次数据 (drug_idx, protein_idx, labels)
                - drug_idx: [B] 药物索引
                - protein_idx: [B] 蛋白质索引

        Returns:
            output: 预测结果 [B, num_classes] (分类) 或 [B] (回归)
        """
        device = next(self.parameters()).device
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        # 把索引转为 tensor（保留在 CPU，索引操作在 CPU 上进行）
        if not isinstance(drug_idx, torch.Tensor):
            drug_idx = torch.as_tensor(drug_idx, dtype=torch.long)
        if not isinstance(protein_idx, torch.Tensor):
            protein_idx = torch.as_tensor(protein_idx, dtype=torch.long)

        # 在 CPU 上先索引切片，再只把批次数据搬到 GPU
        # drug_g: [B, drug_g_dim], drug_m: [B, max_drug_seq, drug_m_dim]
        # protein_g: [B, protein_g_dim], protein_m: [B, max_protein_seq, protein_m_dim]
        drug_g = graph_or_none.drug_g[drug_idx].to(device)
        c_m = graph_or_none.drug_m[drug_idx].to(device)
        p_g_f = graph_or_none.protein_g[protein_idx].to(device)
        p_m = graph_or_none.protein_m[protein_idx].to(device)

        # 投影到统一维度
        c_g_f = self.c_g_unit(drug_g)      # [B, unify_num]
        c_m = self.c_m_unit(c_m)          # [B, max_drug_seq, unify_num]
        p_g_f = self.p_g_unit(p_g_f)      # [B, unify_num]
        p_m = self.p_m_unit(p_m)          # [B, max_protein_seq, unify_num]

        # 拼接: [drug_g, drug_m, protein_g, protein_m] -> [B, total_seq_len, unify_num]
        # 索引: [0] = drug_g, [1:1+max_drug_seq] = drug_m, [1+max_drug_seq] = protein_g,
        #       [2+max_drug_seq:] = protein_m
        c_size = c_m.size(1)
        p_size = p_m.size(1)

        unite_m = torch.cat([
            c_g_f.unsqueeze(1),    # [B, 1, unify_num]
            c_m,                   # [B, max_drug_seq, unify_num]
            p_g_f.unsqueeze(1),    # [B, 1, unify_num]
            p_m                    # [B, max_protein_seq, unify_num]
        ], dim=1)                   # [B, 1+c_size+1+p_size, unify_num]

        # Transformer Encoder 建模交互
        inter_m = self.Interacting_Layer(unite_m)   # [B, total_seq_len, unify_num]

        # 提取 drug_g 和 protein_g 对应的输出向量
        c_g_out = inter_m[:, 0]                    # [B, unify_num]
        p_g_out = inter_m[:, c_size + 1]           # [B, unify_num]

        # 拼接并预测
        inter_f = torch.cat([c_g_out, p_g_out], dim=1)   # [B, unify_num * 2]
        predict = self.predict_layer(inter_f)            # [B, num_classes] 或 [B, 1]

        return predict


def ColdstartCPI_Model(drug_g_dim=512, drug_m_dim=512, protein_g_dim=512, protein_m_dim=512,
                       unify_num=256, head_num=4, dropout=0.1, task_type='classification',
                       num_classes=2, max_drug_seq=100, max_protein_seq=1000, **kwargs):
    """
    ColdstartCPI 模型工厂函数（用于 model_manager）

    Args:
        drug_g_dim: 药物全局特征维度（默认 512，来自 opendrug 嵌入拼接）
        drug_m_dim: 药物序列特征维度（如果有 drug_sequence 嵌入）
        protein_g_dim: 蛋白质全局特征维度
        protein_m_dim: 蛋白质序列特征维度
        unify_num: 统一表示维度（默认 256）
        head_num: Transformer 头数（默认 4）
        dropout: Dropout（默认 0.1）
        task_type: 'classification' 或 'regression'
        num_classes: 类别数（默认 2）
        max_drug_seq: 药物序列最大长度（默认 100）
        max_protein_seq: 蛋白质序列最大长度（默认 1000）

    Returns:
        ColdstartCPI 模型实例
    """
    return ColdstartCPI(
        drug_g_dim=drug_g_dim,
        drug_m_dim=drug_m_dim,
        protein_g_dim=protein_g_dim,
        protein_m_dim=protein_m_dim,
        unify_num=unify_num,
        head_num=head_num,
        dropout=dropout,
        task_type=task_type,
        num_classes=num_classes,
        max_drug_seq=max_drug_seq,
        max_protein_seq=max_protein_seq,
    )
