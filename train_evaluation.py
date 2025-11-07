import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import math

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            received_signal = batch['received_signal'].to(device)
            pilot_sequence = batch['pilot_sequence'].to(device)
            active_state = batch['active_state'].to(device)
            pilot_length = batch['pilot_length'].to(device)

            optimizer.zero_grad()

            # 前向传播
            activity_probs = model(received_signal, pilot_sequence, pilot_length)

            # 计算损失
            loss = criterion(activity_probs, active_state)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                received_signal = batch['received_signal'].to(device)
                pilot_sequence = batch['pilot_sequence'].to(device)
                active_state = batch['active_state'].to(device)
                pilot_length = batch['pilot_length'].to(device)

                activity_probs = model(received_signal, pilot_sequence, pilot_length)
                loss = criterion(activity_probs, active_state)
                val_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 打印统计信息
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch +1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model

def validate_model(model, test_loader):
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_errors = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            received_signal = batch['received_signal'].to(device)
            pilot_sequence = batch['pilot_sequence'].to(device)
            active_state = batch['active_state'].to(device)
            pilot_length = batch['pilot_length'].to(device)

            activity_probs = model(received_signal, pilot_sequence, pilot_length)
            predictions = (activity_probs > 0.5).float()

            errors = torch.sum(torch.abs(predictions - active_state))
            total_errors += errors.item()
            total_samples += active_state.numel()

    aep = total_errors / total_samples
    print(f'Activity Error Probability: {aep:.4f}')

    return aep