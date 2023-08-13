from model_rnn import CharRNN
import torch
import torch.nn as nn

train_data = ...
char_to_idx = ...
idx_to_char = ...
vocab_size = 5000
seq_length = 100
input_size = vocab_size  # 字符数/词汇表大小
hidden_size = 128
output_size = vocab_size
model = CharRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs=10
batch_size=4