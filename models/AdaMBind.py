"""
AdaMBind (Adaptive Molecule-Binding) 模型

基于论文: "AdaMBind: Drug-Target Affinity Prediction with Meta-Learning and Task Adaptation"
https://github.com/JackCai233/AdaMBind

原始 AdaMBind 设计:
- GAT+GCN 混合图神经网络作为药物分子图编码器
  - GAT Layer (Graph Attention): 10-head 注意力聚合邻居原子特征
  - GCN Layer (Graph Convolution): 对称归一化消息传递
  - concat[global_max_pool, global_mean_pool] → Linear → 128-d
- 1D CNN 蛋白质序列编码器
  - Embedding(26, 128) → Conv1D(32 filters, kernel=8) → Linear → 128-d
- 双线性融合层
  - concat[drug(128), protein(128)] → 256-d → MLP → 1 (DTA 亲和力分数)
- MAML 元学习框架 + 自适应任务调度器 (ATS)
  - ATS: LSTM + DeepSets 策略网络，对任务进行难度评分和采样
  - 内循环: 5 步 Adam 微调
  - 外循环: 梯度下降更新元参数

OpenDrug 适配:
- 输入为预计算的药物嵌入 (drug_x: [N_drug, drug_dim]) 和蛋白质嵌入 (protein_x: [N_protein, protein_dim])
- 适配策略: 将 AdaMBind 的双线性协同交互思想迁移到预训练嵌入融合
  - 药物分支: MLP 投影编码器 (无 O(n²) 节点间注意力)
  - 蛋白质分支: MLP 投影编码器
  - 双线性协同交互: 元素级交互特征 + 注意力权重
  - 交互预测: MLP 分类器/回归器
- 保留原始 AdaMBind 的核心: 双线性协同交互 + MLP 融合预测

支持:
- DTI 二分类 (CrossEntropyLoss)
- DTA 回归 (MSELoss)
- 评估指标: MSE, RMSE, MAE, R2, Pearson, Spearman, CI (DTA) / Accuracy, F1, AUC, AP (DTI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    高效的 MLP 编码器 (替代 O(n²) 的 GAT attention)

    由于 OpenDrug 中每个 drug/protein 已是单一嵌入向量（非分子图），
    用两层的 MLP 投影代替节点间注意力，达到 O(n) 复杂度。

    架构: Linear(input_dim, hidden) → ReLU → Linear(hidden, output_dim)
    """

    def __init__(self, input_dim, output_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BilinearInteraction(nn.Module):
    """
    双线性协同交互层

    模拟原始 AdaMBind 的核心交互机制:
    - 通过两条独立路径分别对 drug 和 protein 进行线性投影
    - 计算交互强度: alpha = sigmoid(W1 @ drug + W2 @ protein)
    - 生成交互特征: cross_feat = drug * protein * alpha
    - 融合: concat[drug, protein, cross_feat] → MLP
    """

    def __init__(self, drug_dim=128, protein_dim=128, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim

        self.W1 = nn.Linear(drug_dim, 1, bias=False)
        self.W2 = nn.Linear(protein_dim, 1, bias=False)

        self.fusion_proj = nn.Sequential(
            nn.Linear(drug_dim + protein_dim + drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, drug_emb, protein_emb):
        """
        Args:
            drug_emb: [B, drug_dim]
            protein_emb: [B, protein_dim]
        Returns:
            [B, hidden_dim]
        """
        alpha = torch.sigmoid(self.W1(drug_emb) + self.W2(protein_emb))
        cross_feat = drug_emb * protein_emb * alpha
        fused = torch.cat([drug_emb, protein_emb, cross_feat], dim=-1)
        fused = self.fusion_proj(fused)
        return fused


class InteractionPredictor(nn.Module):
    """
    交互预测头

    支持:
    - DTA 回归: MLP → 1 (MSELoss)
    - DTI 二分类: MLP → num_classes (CrossEntropyLoss)
    """

    def __init__(self, input_dim, hidden_dim=512, dropout=0.2,
                 num_classes=2, task_type='regression'):
        super().__init__()
        self.task_type = task_type

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if task_type == 'regression':
            self.output = nn.Linear(hidden_dim // 2, 1)
        else:
            self.output = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        h = self.mlp(x)
        return self.output(h)


class AdaMBind(nn.Module):
    """
    AdaMBind 模型

    核心架构:
    1. MLPEncoder (Drug): 高效投影编码药物嵌入 (O(n), 非 O(n²))
    2. MLPEncoder (Protein): 高效投影编码蛋白质嵌入
    3. BilinearInteraction: 双线性协同交互 (核心创新)
    4. InteractionPredictor: MLP 预测器

    关键设计:
    - 无 O(n²) 节点间注意力: 每个 drug/protein 嵌入独立编码为 128-d
    - 双线性交互: alpha = sigmoid(W1@drug + W2@protein) 加权协同
    - concat[drug, protein, cross_feat] → MLP → 预测

    Args:
        drug_dim: 药物嵌入维度
        protein_dim: 蛋白质嵌入维度
        hidden_dim: 隐藏层维度 (默认 512)
        dropout: Dropout 概率 (默认 0.2)
        num_classes: 类别数 (二分类=2, 多标签=标签数)
        task_type: 'regression' (DTA) 或 'binary'/'multilabel' (DTI)
    """

    def __init__(self, drug_dim=1024, protein_dim=1024, hidden_dim=512,
                 dropout=0.2, num_classes=2, task_type='regression',
                 **kwargs):
        super().__init__()
        self.drug_dim = drug_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.task_type = task_type

        proj_dim = 128

        self.drug_encoder = MLPEncoder(
            input_dim=drug_dim,
            output_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.protein_encoder = MLPEncoder(
            input_dim=protein_dim,
            output_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.bilinear_interaction = BilinearInteraction(
            drug_dim=proj_dim,
            protein_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.predictor = InteractionPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            task_type=task_type,
        )

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象，包含:
                - drug_x: [N_drug, drug_dim] 药物嵌入
                - protein_x: [N_protein, protein_dim] 蛋白质嵌入
                - edge_index: [2, E] 异构图边
            idx_batch: 批次数据 (drug_idx, protein_idx, labels)
                - drug_idx: [B] 药物索引
                - protein_idx: [B] 蛋白质索引
                - labels: [B] 或 [B, num_classes]

        Returns:
            output: [B, 1] (DTA) 或 [B, num_classes] (DTI)
        """
        drug_idx = idx_batch[0]
        protein_idx = idx_batch[1]

        device = next(self.parameters()).device
        if isinstance(drug_idx, torch.Tensor):
            drug_idx = drug_idx.to(device)
            protein_idx = protein_idx.to(device)
        else:
            drug_idx = torch.as_tensor(drug_idx, dtype=torch.long, device=device)
            protein_idx = torch.as_tensor(protein_idx, dtype=torch.long, device=device)

        drug_x = graph_or_none.drug_x
        protein_x = graph_or_none.protein_x

        drug_emb_raw = drug_x[drug_idx]
        protein_emb_raw = protein_x[protein_idx]

        drug_emb = self.drug_encoder(drug_emb_raw)
        protein_emb = self.protein_encoder(protein_emb_raw)

        fused = self.bilinear_interaction(drug_emb, protein_emb)

        output = self.predictor(fused)

        return output


def AdaMBind_Model(drug_dim=1024, protein_dim=1024, hidden_dim=512,
                   dropout=0.2, num_classes=2,
                   task_type='regression', **kwargs):
    """
    AdaMBind 模型工厂函数（用于 model_manager）
    """
    return AdaMBind(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
