import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import math

class SpatialCorrelationModule(nn.Module):
    """
    空间相关性模块 (SCM)
    """
    def __init__(self, M_x, M_y):
        super().__init__()
        self.M_x = M_x
        self.M_y = M_y

        # 使用卷积层提取空间相关性
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x形状: [batch_size, L_max, 2, M] -> 需要转换为 [batch_size, L_max, 2, M_x, M_y]
        batch_size, L_max, _, M = x.shape
        x = x.view(batch_size, L_max, 2, self.M_x, self.M_y)
        x = x.permute(0, 1, 3, 4, 2)  # [batch_size, L_max, M_x, M_y, 2]
        x = x.reshape(-1, self.M_x, self.M_y, 2)  # 合并前两个维度
        x = x.permute(0, 3, 1, 2)  # [batch_size*L_max, 2, M_x, M_y]

        # 应用卷积层
        x = self.conv_layers(x)

        # 恢复原始形状
        x = x.permute(0, 2, 3, 1)  # [batch_size*L_max, M_x, M_y, 2]
        x = x.reshape(batch_size, L_max, self.M_x, self.M_y, 2)
        x = x.permute(0, 1, 4, 2, 3)  # [batch_size, L_max, 2, M_x, M_y]
        x = x.reshape(batch_size, L_max, 2, self.M_x * self.M_y)  # [batch_size, L_max, 2, M]
        x = x.permute(0, 1, 3, 2)  # [batch_size, L_max, M, 2]

        return x


class PilotLengthAdaptiveModule(nn.Module):
    """
    导频长度自适应模块 (PLAM)
    """
    def __init__(self, d_model, L_p_min, L_p_max):
        super().__init__()
        self.d_model = d_model
        self.L_p_min = L_p_min
        self.L_p_max = L_p_max

        self.fc = nn.Sequential(
            nn.Linear(d_model + 1, d_model),  # 输入: d_model维特征 + 导频长度
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x, L_p):
        # x形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # 平均池化
        pooled = torch.mean(x, dim=1)  # [batch_size, d_model]

        # 归一化导频长度
        normalized_L_p = (L_p - self.L_p_min) / (self.L_p_max - self.L_p_min)
        normalized_L_p = normalized_L_p.view(-1, 1).float()  # [batch_size, 1]

        # 拼接特征和导频长度
        combined = torch.cat([pooled, normalized_L_p], dim=1)  # [batch_size, d_model+1]

        # 通过全连接层生成权重
        weights = self.fc(combined)  # [batch_size, d_model]
        weights = weights.unsqueeze(1)  # [batch_size, 1, d_model]

        # 应用权重
        x = x * weights  # [batch_size, seq_len, d_model]

        return x


class HeterogeneousTransformer(nn.Module):
    """
    异构Transformer编码器层
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src