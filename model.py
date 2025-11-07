import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import math
from model_components import *

class VPLAUDNet(nn.Module):
    """
    可变导频长度的AUD网络
    """
    def __init__(self, num_users, M_antennas, d_model=512, nhead=8,
                 num_layers=5, M_x=8, M_y=8, L_p_min=8, L_p_max=28):
        super().__init__()
        self.num_users = num_users
        self.L_p_max = L_p_max
        self.M_antennas = M_antennas
        self.d_model = d_model
        self.M_x = M_x
        self.M_y = M_y

        # 预处理线性层
        self.pilot_linear = nn.Linear(2 * L_p_max, d_model)  # 导频序列线性层
        self.signal_linear = nn.Linear(2 * L_p_max * L_p_max, d_model)  # 接收信号线性层

        # 空间相关性模块
        self.scm = SpatialCorrelationModule(M_x, M_y)

        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            HeterogeneousTransformer(d_model, nhead) for _ in range(num_layers)
        ])

        # PLAM模块 (每层之间插入)
        self.plam_layers = nn.ModuleList([
            PilotLengthAdaptiveModule(d_model, L_p_min, L_p_max) for _ in range(num_layers)
        ])

        # 活动解码器
        self.activity_decoder = nn.Linear(d_model, 1)
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, received_signal, pilot_sequence, pilot_length):
        # received_signal: [batch_size, L_p_max, M, 2]
        # pilot_sequence: [batch_size, L_p_max, num_users, 2]
        # pilot_length: [batch_size]

        batch_size = received_signal.size(0)

        # 预处理接收信号
        # 应用空间相关性模块
        received_signal = self.scm(received_signal)  # [batch_size, L_p_max, M, 2]

        # 计算协方差矩阵
        received_real = received_signal[..., 0]
        received_imag = received_signal[..., 1]
        received_complex = torch.complex(received_real, received_imag)

        # 计算协方差矩阵 C = Y Y^H / M
        covariance = torch.einsum('blm,bkm->blk', received_complex, received_complex.conj()) / self.M_antennas
        covariance = torch.view_as_real(covariance)  # [batch_size, L_p_max, L_p_max, 2]
        covariance = covariance.reshape(batch_size, -1)  # [batch_size, 2 * L_p_max * L_p_max]
        signal_feature = self.signal_linear(covariance)  # [batch_size, d_model]
        signal_feature = signal_feature.unsqueeze(1)  # [batch_size, 1, d_model]

        # 预处理导频序列
        pilot_sequence = pilot_sequence.permute(0, 2, 1, 3)  # [batch_size, num_users, L_p_max, 2]
        pilot_sequence = pilot_sequence.reshape(batch_size, self.num_users, -1)  # [batch_size, num_users, 2 * L_p_max]
        pilot_features = self.pilot_linear(pilot_sequence)  # [batch_size, num_users, d_model]

        # 拼接所有序列 (K个导频序列 + 1个接收信号序列)
        all_sequences = torch.cat([pilot_features, signal_feature], dim=1)  # [batch_size, num_users+1, d_model]

        # 通过Transformer层
        for i, (transformer_layer, plam_layer) in enumerate(zip(self.transformer_layers, self.plam_layers)):
            all_sequences = transformer_layer(all_sequences)
            all_sequences = plam_layer(all_sequences, pilot_length)

        # 活动解码器
        signal_output = all_sequences[:, -1, :]  # 接收信号的特征 [batch_size, d_model]
        pilot_outputs = all_sequences[:, :-1, :]  # 所有用户的特征 [batch_size, num_users, d_model]

        # 计算相关性
        signal_output = signal_output.unsqueeze(1)  # [batch_size, 1, d_model]
        correlation = torch.matmul(signal_output, self.W_o)  # [batch_size, 1, d_model]
        correlation = torch.matmul(correlation, pilot_outputs.transpose(1, 2))  # [batch_size, 1, num_users]
        correlation = correlation.squeeze(1)  # [batch_size, num_users]

        # 应用tanh和sigmoid
        correlation = 10 * torch.tanh(correlation / math.sqrt(self.d_model))
        activity_probs = torch.sigmoid(correlation)

        return activity_probs