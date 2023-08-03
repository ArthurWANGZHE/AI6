import torch
import torch.nn as nn
import numpy as np

# 1. 准备数据
# 将小说数据进行预处理，并将其转换成数字序列（例如，每个字符对应一个数字）
# 假设你已经有了以下数据
# train_data: 数字序列形状为 (num_sequences, sequence_length)
# char_to_idx: 字符到数字的映射字典
# idx_to_char: 数字到字符的映射字典
train_data = ...
char_to_idx = ...
idx_to_char = ...
vocab_size = len(char_to_idx)


# 2. 定义模型
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output.view(-1, self.hidden_size))
        return output, hidden


# 3. 初始化模型和优化器
input_size = vocab_size  # 字符数/词汇表大小
hidden_size = 128
output_size = vocab_size
model = CharRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs=100
batch_size=64

# 4. 训练模型
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
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 将数据移到设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = train_data.astype(np.int64)
train_data = torch.tensor(train_data, dtype=torch.long).to(device)

# 训练模型
train_model(model, train_data)


# 5. 使用模型生成文本
def generate_text(model, start_text, length=100):
    model.eval()
    with torch.no_grad():
        hidden = None
        inputs = torch.tensor([[char_to_idx[c] for c in start_text]], dtype=torch.long).to(device)
        generated_text = start_text

        for _ in range(length):
            outputs, hidden = model(inputs, hidden)
            prob = torch.softmax(outputs[-1], dim=0).cpu().numpy()
            char_idx = np.random.choice(vocab_size, p=prob)
            char = idx_to_char[char_idx]
            generated_text += char
            inputs = torch.tensor([[char_idx]], dtype=torch.long).to(device)

        return generated_text


# 使用模型生成文本
start_text = "Once upon a time"
generated_text = generate_text(model, start_text, length=200)
print(generated_text)