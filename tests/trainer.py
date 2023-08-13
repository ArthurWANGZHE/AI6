import torch
import numpy as np
from global_config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(model, data, num_epochs, batch_size):
    model.train()
    num_sequences = len(data)
    num_batches = num_sequences // batch_size

    for epoch in range(num_epochs):
        # 每个epoch开始时，打乱数据的顺序
        np.random.shuffle(data)
        hidden = None

        for batch_idx in range(num_batches):
            # 获取当前批次数据
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = data[start_idx:end_idx]

            # 将数据转换为Tensor
            inputs = torch.tensor(batch_data[:, :-1], dtype=torch.long)
            targets = torch.tensor(batch_data[:, 1:], dtype=torch.long)

            # 将输入和目标数据移到设备（GPU或CPU）
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            outputs, hidden = model(inputs, hidden)

            # 计算损失
            loss = criterion(outputs, targets.view(-1))

            # 反向传播和优化
            loss.backward(retain_graph=True)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')