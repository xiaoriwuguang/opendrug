"""
D-SCRIPT (Deep Statistical Recognition of Interaction Sites) 模型

基于论文: "A geometric learning approach for protein and drug-target interaction
signaling prediction applied to drug discovery"
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010104

原始 D-SCRIPT 设计:
- 使用预训练语言模型 (ESM/Bepler LSTM) 的 per-residue 嵌入
- Projection Module: 降维投影 [N, d0] -> [N, d]
- Contact Module: 2D CNN 从配对蛋白质的投影嵌入预测接触图
  - z_dif = |z1 - z2|, z_mul = z1 * z2 -> concat -> Conv2d -> contact map [N, M]
- Interaction Module: 从接触图聚合预测蛋白质对是否相互作用
  - 距离加权池化 (可学习 theta 和 lambda 参数)
  - 全局均值-方差池化 -> sigmoid -> p(interaction)

OpenDrug 适配:
- 输入为蛋白质级嵌入 (protein_x: [N, protein_dim])
- 适配策略: 将蛋白质嵌入视为"序列长度=1"的特殊 contact map
  - 投影模块将 protein_dim 降至 projection_dim
  - 交互模块直接计算两个蛋白质的配对特征，不依赖序列长度
  - concat + |emb1 - emb2| + emb1 * emb2 -> 深度 MLP -> 交互预测
  - D-SCRIPT 核心思想: 保留可学习的距离加权池化机制

支持:
- PPI 二分类 (CrossEntropyLoss)
- PPI 多标签分类 (BCEWithLogitsLoss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticActivation(nn.Module):
    """
    广义 Sigmoid 激活函数

    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`

    k 是可学习的参数 (默认 k=1, 不训练)
    """

    def __init__(self, x0=0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)
        return o

    def clip(self):
        self.k.data.clamp_(min=0)


class ProjectionModule(nn.Module):
    """
    投影模块 (对应原始 D-SCRIPT 的 FullyConnectedEmbed)

    将语言模型的高维嵌入投影到低维交互空间。
    在 OpenDrug 适配中: protein_dim -> projection_dim

    架构: Linear(protein_dim, projection_dim) -> ReLU -> Dropout
    """

    def __init__(self, protein_dim, projection_dim, dropout=0.5, activation=nn.ReLU()):
        super().__init__()
        self.projection_dim = projection_dim
        self.transform = nn.Linear(protein_dim, projection_dim)
        self.drop = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, x):
        """
        Args:
            x: [B, protein_dim]
        Returns:
            [B, projection_dim]
        """
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t


class ContactModule(nn.Module):
    """
    接触图模块 (对应原始 D-SCRIPT 的 ContactCNN)

    将两个蛋白质的投影嵌入进行配对交互建模，预测接触图。
    核心思想: z_dif = |z1 - z2|, z_mul = z1 * z2

    原始 D-SCRIPT 处理可变长序列 (N x M contact map)，
    OpenDrug 适配中处理蛋白质级嵌入 (1 x 1 特殊 case)，
    通过将 protein_x 展开为 [B, 1, proj_dim] 来兼容 2D 卷积接口。

    架构: FullyConnected 配对 -> Conv2d 1x1 -> BatchNorm -> Sigmoid
    """

    def __init__(self, embed_dim, hidden_dim=50, width=7, activation=nn.Sigmoid()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.width = width

        self.fc = nn.Sequential(
            nn.Conv2d(2 * embed_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.conv = nn.Conv2d(hidden_dim, 1, kernel_size=width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        self._clip()

    def _clip(self):
        """强制转置不变性 (transpose invariance)"""
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        """
        Args:
            z0: [B, proj_dim] 蛋白质0的投影嵌入
            z1: [B, proj_dim] 蛋白质1的投影嵌入
        Returns:
            contact_map: [B, 1, 1, 1] 预测接触图 (标量交互强度)
        """
        B = z0.size(0)
        proj_d = z0.size(1)

        z0_2d = z0.unsqueeze(1)
        z1_2d = z1.unsqueeze(1)

        z0_exp = z0_2d.unsqueeze(2).expand(B, 1, 1, proj_d)
        z1_exp = z1_2d.unsqueeze(1).expand(B, 1, 1, proj_d)

        z_dif = torch.abs(z0_exp - z1_exp)
        z_mul = z0_exp * z1_exp
        z_cat = torch.cat([z_dif, z_mul], dim=3)

        z_cat = z_cat.transpose(2, 3).transpose(1, 2)

        c = self.fc(z_cat)
        s = self.conv(c)
        s = self.batchnorm(s)
        s = self.activation(s)

        return s


class InteractionModule(nn.Module):
    """
    交互模块 (对应原始 D-SCRIPT 的 ModelInteraction)

    核心思想: 从接触图聚合预测蛋白质对是否相互作用。

    原始 D-SCRIPT 使用:
    1. 可学习的距离加权矩阵 W (theta, lambda 参数)
    2. 全局均值-方差池化: phat = mean(Q) where Q = relu(C - mu - gamma*sigma)
    3. 可学习的 sigmoid 激活 (LogisticActivation)

    OpenDrug 适配: 保留核心池化机制，但输出多标签 logits。
    """

    def __init__(self, use_w=True, do_sigmoid=True, do_pool=False,
                 pool_size=9, theta_init=1, lambda_init=0, gamma_init=0,
                 do_deep_classifier=False, hidden_dim=128, num_classes=2):
        super().__init__()
        self.use_w = use_w
        self.do_sigmoid = do_sigmoid
        self.do_pool = do_pool
        self.pool_size = pool_size

        if do_sigmoid:
            self.activation = LogisticActivation(x0=0.5, k=20)

        if use_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        if do_pool:
            self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)

        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))

        self.do_deep_classifier = do_deep_classifier
        self.num_classes = num_classes

        if do_deep_classifier:
            self.deep_classifier = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        self._clip()

        self.register_buffer('xx', torch.arange(2000), persistent=False)

    def _clip(self):
        if self.use_w:
            self.theta.data.clamp_(min=0, max=1)
            self.lambda_.data.clamp_(min=0)
        self.gamma.data.clamp_(min=0)

    def _build_weight_matrix(self, N, M, device):
        """构建 D-SCRIPT 风格的可学习距离加权矩阵 [1, 1, N, M]"""
        x1 = -1 * torch.square(
            (self.xx[:N].to(device) + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2))
        )
        x2 = -1 * torch.square(
            (self.xx[:M].to(device) + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2))
        )
        x1 = torch.exp(self.lambda_ * x1)
        x2 = torch.exp(self.lambda_ * x2)
        W = x1.unsqueeze(1) * x2  # [N, M]
        W = (1 - self.theta) * W + self.theta
        W = W.unsqueeze(0).unsqueeze(0)  # [1, 1, N, M]
        return W

    def _pool_interaction(self, contact_map):
        """
        D-SCRIPT 核心池化机制

        Args:
            contact_map: [B, 1, N, M] 每个样本的接触图
        Returns:
            phat: [B] 每个样本的交互强度 (sigmoid 或 raw)
        """
        B = contact_map.size(0)
        N, M = contact_map.shape[-2:]

        # 1. 距离加权
        if self.use_w:
            W = self._build_weight_matrix(N, M, contact_map.device)
            weighted = contact_map * W
        else:
            weighted = contact_map

        # 2. 最大池化
        if self.do_pool:
            weighted = self.maxPool(weighted)

        # 3. 展平 spatial -> [B, N*M]
        flat = weighted.view(B, -1)

        # 4. 全局均值-方差条件池化
        mu = flat.mean(dim=1)  # [B]
        sigma = flat.std(dim=1)  # [B]
        threshold = mu + self.gamma * sigma
        Q = torch.relu(flat - threshold.unsqueeze(1))
        phat = Q.sum(dim=1) / (Q.sign().sum(dim=1) + 1)  # [B]

        if self.do_sigmoid:
            phat = self.activation(phat)

        return phat

    def forward(self, contact_map):
        """
        Args:
            contact_map: [B, 1, 1, 1] 接触图 (D-SCRIPT contact score per pair)
        Returns:
            [B, num_classes] logits
        """
        B = contact_map.size(0)
        phat = self._pool_interaction(contact_map)  # [B]

        if self.do_deep_classifier:
            interaction_scalar = phat.unsqueeze(1)  # [B, 1]
            logits = self.deep_classifier(interaction_scalar)
        else:
            interaction_scalar = phat.unsqueeze(1)  # [B, 1]
            if self.num_classes == 2:
                logits = torch.cat([1 - interaction_scalar, interaction_scalar], dim=1)
            else:
                logits = interaction_scalar.expand(B, self.num_classes)

        return logits


class D_SCRIPT(nn.Module):
    """
    D-SCRIPT 模型

    核心架构:
    1. ProjectionModule: protein_dim -> projection_dim
       模拟原始 D-SCRIPT 将高维语言模型嵌入投影到交互空间的 FC 层
    2. ContactModule: 配对特征构建 + 2D CNN
       模拟原始 D-SCRIPT 从两个蛋白质的嵌入预测接触图
    3. InteractionModule: 距离加权池化 + 深度 MLP 分类头
       模拟原始 D-SCRIPT 从接触图聚合预测蛋白质对相互作用

    关键设计:
    - D-SCRIPT 核心思想: 分离的投影 + 配对交互 + 可学习的全局池化
    - 可学习的距离加权机制 (theta, lambda) 捕捉序列距离依赖
    - 均值-方差条件池化 (gamma) 筛选显著接触位点
    - 对于二分类: 保留原始 D-SCRIPT 的 sigmoid 概率化
    - 对于多标签: 深度 MLP 分类头支持多类别独立预测

    Args:
        protein_dim: 蛋白质嵌入维度
        projection_dim: 投影维度 (默认 100)
        dropout: Dropout 概率 (默认 0.5)
        hidden_dim: Contact CNN 隐藏维度 (默认 50)
        width: 2D 卷积核宽度 (默认 7)
        num_classes: 类别数 (二分类=2, 多标签=标签数)
        task_type: 'binary' 或 'multilabel'
        use_w: 是否使用距离加权矩阵 (默认 True)
        do_sigmoid: 是否使用 sigmoid 激活 (默认 True, 仅二分类)
        do_deep_classifier: 多标签时使用深度 MLP (默认 True)
    """

    def __init__(self, protein_dim=1024, projection_dim=100, dropout=0.5,
                 hidden_dim=50, width=7, num_classes=2, task_type='binary',
                 use_w=True, do_sigmoid=True, do_deep_classifier=True,
                 pool_size=9, theta_init=1, lambda_init=0, gamma_init=0,
                 **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.task_type = task_type

        self.projection = ProjectionModule(
            protein_dim=protein_dim,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        final_sigmoid = do_sigmoid if task_type == 'binary' else False

        self.contact = ContactModule(
            embed_dim=projection_dim,
            hidden_dim=hidden_dim,
            width=width,
            activation=nn.Sigmoid(),
        )

        self.interaction = InteractionModule(
            use_w=use_w,
            do_sigmoid=final_sigmoid,
            do_pool=False,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
            do_deep_classifier=do_deep_classifier and task_type == 'multilabel',
            hidden_dim=max(projection_dim, 64),
            num_classes=num_classes,
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

        p1_emb = protein_x[p1_idx]
        p2_emb = protein_x[p2_idx]

        p1_proj = self.projection(p1_emb)
        p2_proj = self.projection(p2_emb)

        contact_map = self.contact(p1_proj, p2_proj)

        output = self.interaction(contact_map)

        return output


def D_SCRIPT_Model(protein_dim=1024, projection_dim=100, dropout=0.5,
                   hidden_dim=50, width=7, num_classes=2, task_type='binary',
                   use_w=True, do_sigmoid=True, do_deep_classifier=True,
                   pool_size=9, theta_init=1, lambda_init=0, gamma_init=0, **kwargs):
    """
    D_SCRIPT 模型工厂函数（用于 model_manager）
    """
    return D_SCRIPT(
        protein_dim=protein_dim,
        projection_dim=projection_dim,
        dropout=dropout,
        hidden_dim=hidden_dim,
        width=width,
        num_classes=num_classes,
        task_type=task_type,
        use_w=use_w,
        do_sigmoid=do_sigmoid,
        do_deep_classifier=do_deep_classifier,
        pool_size=pool_size,
        theta_init=theta_init,
        lambda_init=lambda_init,
        gamma_init=gamma_init,
    )
