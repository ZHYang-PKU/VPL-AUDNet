import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import math
from model import VPLAUDNet
from model_components import *
from train_evaluation import *
from data_generation import *


def main():
    # 设置同步错误报告
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 参数设置
    num_samples = 20000
    num_users = 20
    L_p_min = 256
    L_p_max = 256
    M_antennas = 256
    M_x = 256
    M_y = 1
    snr_db = 0
    active_prob = 0.2
    batch_size = 64
    d_model = 512
    nhead = 8
    num_layers = 4

    # 创建数据集
    dataset = UserActivityDataset(
        num_samples=num_samples,
        num_users=num_users,
        L_p_min=L_p_min,
        L_p_max=L_p_max,
        M_antennas=M_antennas,
        snr_db=snr_db,
        active_prob=active_prob
    )


    # 划分数据集
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Dataset loaded")

    # 创建模型
    model = VPLAUDNet(
        num_users=num_users,
        M_antennas=M_antennas,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        M_x=M_x,
        M_y=M_y,
        L_p_min=L_p_min,
        L_p_max=L_p_max
    )

    # 训练模型
    model = train_model(model, train_loader, val_loader)

    # 测试模型
    aep = validate_model(model, test_loader)


if __name__ == "__main__":
    main()