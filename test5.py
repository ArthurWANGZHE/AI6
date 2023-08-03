import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

seq_length = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_text(text, seq_length):
    # 构建字符到数字的映射字典和数字到字符的映射字典
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # 将文本转换为数字序列
    data = [char_to_idx[ch] for ch in text]

    # 构建训练样本
    num_sequences = len(data) - seq_length
    sequences = []
    for i in range(num_sequences):
        seq = data[i:i + seq_length + 1]
        sequences.append(seq)

    return sequences, char_to_idx, idx_to_char

with open('data/book1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

sequences, char_to_idx, idx_to_char = preprocess_text(text, seq_length)
vocab_size = len(char_to_idx)

# 保存字符到数字的映射字典和数字到字符的映射字典
np.save('data/char_to_idx.npy', char_to_idx)
np.save('data/idx_to_char.npy', idx_to_char)


# 将数据移到设备（GPU或CPU）
train_data = np.array(sequences, dtype=np.int64)
train_data = torch.tensor(train_data,dtype=torch.long).to(device)


# 2. Define the CharRNN model
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


# 3. Initialize the model and optimizer on the GPU
input_size = vocab_size
hidden_size = 128
output_size = vocab_size
model = CharRNN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Create DataLoader for batching and shuffling the data
batch_size = 4
train_dataset = TensorDataset(train_data[:, :-1], train_data[:, 1:])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 5. Train the model
def train_model(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        hidden = None
        for inputs, targets in train_loader:
            # Move inputs and targets to the GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear gradients before the backward pass
            optimizer.zero_grad()

            # Forward pass, loss computation, and backward pass
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            loss.backward(retain_graph=True)
            optimizer.step()





            total_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')


# Train the model using the DataLoader
train_model(model, train_loader)

# 保存模型
torch.save(model.state_dict(), 'data/model.pth')

def generate_text(model, start_text, char_to_idx,idx_to_char,length=100):
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
# 6. Generate text using the model (Assuming you have already implemented the generate_text function)
start_text = "从前有座山"
# 导入模型和字符到数字的映射字典
model=model.load_state_dict(torch.load('data/model.pth'))
char_to_idx=np.load('data/char_to_idx.npy',allow_pickle=True).item()
idx_to_char=np.load('data/idx_to_char.npy',allow_pickle=True).item()
generated_text = generate_text(model, start_text,char_to_idx=char_to_idx,idx_to_char=idx_to_char, length=200)
print(generated_text)