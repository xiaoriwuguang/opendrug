"""
MARPPI (Multi-Attention ResNet for Protein-Protein Interaction) 模型

基于双通道 1D ResNet 的 PPI 预测模型，适配 OpenDrug 预计算蛋白质嵌入输入。

架构设计参考自 baseline/ppi/MARPPI/sim.py：
1. 双通道架构：两个蛋白质分别经过独立的 ResNet 风格编码器
2. Multi-head 分支交互块（identify_blocknew）：将特征沿通道分割为多头，
   跨头建立交叉连接以建模氨基酸残基间的依赖关系
3. 交互特征融合：拼接 + 元素乘积 + 绝对差值
4. MLP 分类头

支持：
- PPI 二分类
- PPI 多标签分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSplitBlock(nn.Module):
    """
    Multi-head 分支交互块（对应 sim.py 中的 identify_blocknew）

    核心思想：将特征沿通道分为多头 (num_splits=4)，跨头建立交叉连接，
    捕获不同子空间特征间的依赖关系，最后再重新拼接融合。

    原始 Keras 实现利用 Add 层的广播特性处理维度不匹配，
    PyTorch 版本使用 1x1 卷积确保 shortcut 通道匹配。

    输入: [B, channels, seq_len]
    输出: [B, out_channels, seq_len]
    """

    def __init__(self, in_channels, out_channels, num_splits=4, kernel_size=3):
        super().__init__()
        self.num_splits = num_splits
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.split_channels = in_channels // num_splits

        self.bn_pre = nn.BatchNorm1d(in_channels)

        self.conv_1x1_a = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn_a = nn.BatchNorm1d(in_channels)

        self.conv_1x1_out = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm1d(out_channels)

        self.conv_branch = nn.ModuleList()
        for _ in range(num_splits - 1):
            self.conv_branch.append(
                nn.Conv1d(self.split_channels, self.split_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
            )

        self.conv_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x

        x = self.bn_pre(x)
        x = F.relu(x)

        x = self.conv_1x1_a(x)
        x = self.bn_a(x)
        x = F.relu(x)

        # 沿通道分割为 num_splits 个头
        heads = list(torch.split(x, self.split_channels, dim=1))

        # 跨头交叉连接：head[i] += conv(head[i+1])
        for i in range(self.num_splits - 1):
            out = self.conv_branch[i](heads[i + 1])
            heads[i] = heads[i] + out

        x = torch.cat(heads, dim=1)

        x = self.conv_1x1_out(x)
        x = self.bn_out(x)

        # 原始 Keras 代码中 Add() 自动广播 sequence 维度 (?, 1, C) + (?, 2, C)
        # PyTorch 中 shortcut 需要匹配通道和序列长度
        shortcut = self.conv_shortcut(identity)
        shortcut = self.bn_shortcut(shortcut)

        x = x + shortcut
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    标准 1D 残差块（对应 sim.py 中的 identify_block）

    输入: [B, in_channels, L]
    输出: [B, out_channels, L]
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(mid_channels)

        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = F.relu(out)
        return out


class ConvolutionalBlock(nn.Module):
    """
    带下采样的 1D 卷积残差块（对应 sim.py 中的 convolutional_block）

    stride > 1 时同时压缩序列长度和扩展通道数
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(mid_channels)

        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = F.relu(out)
        return out


class ProteinEncoder(nn.Module):
    """
    单通道蛋白质编码器（对应 sim.py 中的 Channel-1 / Channel-2）

    层级结构:
    1. 预投影: protein_dim -> encoder_dim (避免大维度直接进 Conv1D 显存爆炸)
    2. Reshape -> Conv1D 格式: [B, 1, protein_dim] -> reshape -> [B, seq_len, encoder_dim]
    3. Conv1D(encoder_dim, k=3) + MaxPool
    4. ResidualBlock: encoder_dim -> encoder_dim
    5. MultiHeadSplitBlock: encoder_dim -> encoder_dim (多头交叉连接)
    6. 平均池化: -> [B, encoder_dim]

    输入: [B, protein_dim]
    输出: [B, encoder_dim]
    """

    def __init__(self, protein_dim, encoder_dim=512, dropout=0.2):
        super().__init__()
        self.protein_dim = protein_dim
        self.encoder_dim = encoder_dim

        proj_dim = encoder_dim
        seq_len = 8

        # 预投影：protein_dim -> encoder_dim，避免大维度直接进 Conv1D
        self.pre_proj = nn.Sequential(
            nn.Linear(protein_dim, proj_dim * seq_len),
            nn.BatchNorm1d(proj_dim * seq_len),
            nn.ReLU(inplace=True),
        )

        self.init_conv = nn.Conv1d(proj_dim, encoder_dim, kernel_size=3, stride=1)
        self.init_bn = nn.BatchNorm1d(encoder_dim)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv_block = ConvolutionalBlock(encoder_dim, encoder_dim, stride=1)

        self.res_block = ResidualBlock(encoder_dim, encoder_dim)

        self.multihead_block = MultiHeadSplitBlock(encoder_dim, encoder_dim, num_splits=4)

        self.final_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, protein_dim]
        B = x.size(0)

        # 预投影: [B, protein_dim] -> [B, proj_dim * seq_len]
        x = self.pre_proj(x)
        # 重塑为 Conv1D 格式: [B, proj_dim * seq_len] -> [B, proj_dim, seq_len]
        x = x.view(B, self.encoder_dim, 8)

        x = self.init_conv(x)                   # -> [B, encoder_dim, 6]
        x = self.init_bn(x)
        x = F.relu(x)
        x = self.init_pool(x)                   # -> [B, encoder_dim, 2]

        x = self.conv_block(x)                   # -> [B, encoder_dim, 2]
        x = self.res_block(x)                   # -> [B, encoder_dim, 2]
        x = self.multihead_block(x)             # -> [B, encoder_dim, 2]
        x = self.final_pool(x)                  # -> [B, encoder_dim, 1]
        x = x.squeeze(-1)                       # -> [B, encoder_dim]

        return x


class InteractionModule(nn.Module):
    """
    交互模块：融合两个蛋白质的编码表示

    交互方式（对应 sim.py 中的 concatenate + 全连接分类头）:
    - 拼接: concat(p1, p2)
    - 元素乘积: p1 * p2
    - 绝对差值: |p1 - p2|

    总输入维度: encoder_dim * 4
    """

    def __init__(self, encoder_dim, dropout=0.2):
        super().__init__()
        self.encoder_dim = encoder_dim

        fusion_dim = encoder_dim * 4

        self.fc1 = nn.Linear(fusion_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 8)
        self.bn4 = nn.BatchNorm1d(8)

        self.dropout = nn.Dropout(dropout)

    def forward(self, p1_enc, p2_enc):
        # 交互特征融合
        diff = torch.abs(p1_enc - p2_enc)
        prod = p1_enc * p2_enc
        fusion = torch.cat([p1_enc, p2_enc, prod, diff], dim=1)  # [B, 4*encoder_dim]

        x = self.fc1(fusion)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class MARPPI(nn.Module):
    """
    MARPPI 模型

    双通道架构：
    1. Encoder A: protein_A -> [B, encoder_dim]
    2. Encoder B: protein_B -> [B, encoder_dim]
    3. Interaction: concat, prod, diff -> MLP classifier
    4. Output: [B, num_classes]

    特点（继承自 sim.py）：
    - 双通道对称编码器（参数共享）
    - Multi-head 分支交互块（identify_blocknew）
    - 标准残差连接
    - BatchNorm 稳定训练
    """

    def __init__(self, protein_dim=1024, encoder_dim=512, dropout=0.2,
                 num_classes=2, task_type='binary', **kwargs):
        super().__init__()
        self.protein_dim = protein_dim
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        self.task_type = task_type

        # 双通道编码器（参数共享）
        self.encoder = ProteinEncoder(protein_dim, encoder_dim, dropout)

        # 交互模块
        self.interaction = InteractionModule(encoder_dim, dropout)

        # 输出层
        if task_type == 'multilabel':
            self.output = nn.Linear(8, num_classes)
        else:
            self.output = nn.Linear(8, num_classes)

    def forward(self, graph_or_none, idx_batch):
        """
        前向传播

        Args:
            graph_or_none: 图数据对象，包含 protein_x
            idx_batch: 批次数据 (p1_idx, p2_idx, labels)
                - p1_idx: [B] 蛋白质1索引
                - p2_idx: [B] 蛋白质2索引

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

        p1_enc = self.encoder(p1_emb)
        p2_enc = self.encoder(p2_emb)

        interaction_out = self.interaction(p1_enc, p2_enc)

        output = self.output(interaction_out)
        return output


def MARPPI_Model(protein_dim=1024, hidden_dim=512, dropout=0.2,
                 num_classes=2, task_type='binary', **kwargs):
    """
    MARPPI 模型工厂函数（用于 model_manager）
    hidden_dim -> encoder_dim
    """
    return MARPPI(
        protein_dim=protein_dim,
        encoder_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        task_type=task_type,
    )
