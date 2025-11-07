import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import math

class UserActivityDataset(Dataset):
    """
    生成活跃用户检测数据集，按照论文要求修改
    """
    def __init__(self, num_samples, num_users, L_p_min, L_p_max, M_antennas, snr_db, active_prob=0.1):
        """
        参数:
            num_samples: 样本数量
            num_users: 用户数量 (K)
            L_p_min: 最小导频长度
            L_p_max: 最大导频长度
            M_antennas: 基站天线数量 (M)
            snr_db: 信噪比(dB)
            active_prob: 用户活跃概率
        """
        self.num_samples = num_samples
        self.num_users = num_users
        self.L_p_min = L_p_min
        self.L_p_max = L_p_max
        self.M_antennas = M_antennas
        self.active_prob = active_prob
        self.snr_db = snr_db

        # 生成正交导频序列池 (L_p_max × K)
        self.pilot_pool = self._generate_orthogonal_pilots()

        # 生成活跃用户状态和导频长度
        self.active_states, self.pilot_lengths = self._generate_active_states_and_lengths()

        # 生成信道和接收信号
        self.channels, self.received_signals = self._generate_channels_and_signals()

    def _generate_orthogonal_pilots(self):
        """生成正交导频序列池"""
        # 使用Hadamard矩阵生成正交序列
        max_size = 2 ** math.ceil(math.log2(self.L_p_max))
        H = torch.tensor(np.random.choice([1, -1], size=(max_size, max_size)))
        H = H.float()

        # 正交化处理
        U, S, V = torch.svd(H)
        orthogonal_pilots = U[:, :self.L_p_max].T  # [L_p_max, L_p_max]

        # 扩展到K个用户
        if self.num_users > self.L_p_max:
            # 如果需要更多用户，重复使用导频
            repeat_times = math.ceil(self.num_users / self.L_p_max)
            orthogonal_pilots = orthogonal_pilots.repeat(1, repeat_times)[:, :self.num_users]
        else:
            orthogonal_pilots = orthogonal_pilots[:, :self.num_users]

        # 转换为复数 (QPSK)
        real_part = orthogonal_pilots
        imag_part = torch.zeros_like(real_part)

        # 随机旋转相位增强多样性
        phases = torch.rand(self.num_users) * 2 * torch.pi
        for i in range(self.num_users):
            real_part[:, i] = torch.cos(phases[i]) * orthogonal_pilots[:, i]
            imag_part[:, i] = torch.sin(phases[i]) * orthogonal_pilots[:, i]

        # 合并实部和虚部
        complex_pilots = torch.complex(real_part, imag_part)

        # 归一化功率
        power = torch.mean(torch.abs(complex_pilots) ** 2, dim=0, keepdim=True)
        complex_pilots = complex_pilots / torch.sqrt(power)

        return complex_pilots  # [L_p_max, num_users]

    def _generate_active_states_and_lengths(self):
        """生成用户活跃状态和导频长度"""
        # 生成活跃状态
        active_states = torch.bernoulli(
            torch.full((self.num_samples, self.num_users), self.active_prob)
        )

        # 为每个样本随机生成导频长度
        pilot_lengths = torch.randint(self.L_p_min, self.L_p_max + 1, (self.num_samples,))

        return active_states, pilot_lengths

    def _generate_channels_and_signals(self):
        """生成信道和接收信号"""
        # 生成3GPP信道 (简化版)
        channels = torch.randn(self.num_samples, self.num_users, self.M_antennas, 2)
        channels = channels / math.sqrt(2)  # 归一化功率
        channels = torch.view_as_complex(channels)  # [num_samples, num_users, M_antennas]

        # 初始化接收信号
        received_signals = torch.zeros(self.num_samples, self.L_p_max, self.M_antennas, 2)
        received_signals = torch.view_as_complex(received_signals)

        for i in range(self.num_samples):
            L_p = self.pilot_lengths[i]
            active_mask = self.active_states[i]  # [num_users]

            # 获取当前导频序列 (只取前L_p个符号)
            current_pilots = self.pilot_pool[:L_p, :]  # [L_p, num_users]

            # 计算每个用户的贡献 (只考虑活跃用户)
            active_channels = channels[i] * active_mask.unsqueeze(-1)  # [num_users, M_antennas]
            user_contributions = torch.einsum('lp,pm->lpm', current_pilots, active_channels)  # [L_p, num_users, M_antennas]

            # 在用户维度求和得到理想接收信号
            ideal_signal = torch.sum(user_contributions, dim=1)  # [L_p, M_antennas]

            # 添加高斯噪声
            snr_linear = 10 ** (self.snr_db / 10)
            signal_power = torch.mean(torch.abs(ideal_signal) ** 2)
            noise_power = signal_power / snr_linear

            # 生成复高斯噪声
            noise_real = torch.randn(L_p, self.M_antennas) * math.sqrt(noise_power / 2)
            noise_imag = torch.randn(L_p, self.M_antennas) * math.sqrt(noise_power / 2)
            noise = torch.complex(noise_real, noise_imag)

            # 添加噪声并零填充到最大长度
            noisy_signal = ideal_signal + noise
            received_signals[i, :L_p, :] = noisy_signal

        # 转换为实部虚部分开的形式
        received_signals_real = torch.view_as_real(received_signals)  # [num_samples, L_p_max, M_antennas, 2]

        return channels, received_signals_real

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'received_signal': self.received_signals[idx],  # [L_p_max, M_antennas, 2]
            'active_state': self.active_states[idx],  # [num_users]
            'pilot_sequence': torch.view_as_real(self.pilot_pool),  # [L_p_max, num_users, 2]
            'pilot_length': self.pilot_lengths[idx]  # 标量
        }