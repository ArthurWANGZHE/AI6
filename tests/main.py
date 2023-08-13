import torch
import numpy as np
from data_process import preprocess_text
from global_config import *
from model_rnn import CharRNN
from trainer import train_model
from generator_ import generate_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取文本
with open('../data/book1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 数据预处理
sequences, char_to_idx, idx_to_char = preprocess_text(text, seq_length)
vocab_size = len(char_to_idx)

# 将数据移到设备（GPU或CPU）
train_data = np.array(sequences, dtype=np.int64)
train_data = torch.tensor(train_data, dtype=torch.long).to(device)
model=CharRNN(vocab_size, hidden_size, vocab_size).to(device)
# 创建模型
train_model(model, train_data,num_epochs, batch_size)

start_text = "从前有座山"
generated_text = generate_text(model, start_text, length=200)
print(generated_text)