import torch
import transformers
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
import numpy as np

BERT_PATH = '../bert-base-chinese'
transformers.logging.set_verbosity_error()
# Load pre-trained BERT and tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)


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

with open('../data/book1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

seq_length = 100
sequences, char_to_idx, idx_to_char = preprocess_text(text, seq_length)
vocab_size = len(char_to_idx)

# 保存字符到数字的映射字典和数字到字符的映射字典
np.save('../data/char_to_idx.npy', char_to_idx)
np.save('../data/idx_to_char.npy', idx_to_char)


# Define the RNN-based text generator
class RNNTextGenerator(nn.Module):
    def __init__(self, rnn_hidden_size, vocab_size):
        super(RNNTextGenerator, self).__init__()
        self.rnn = nn.LSTM(input_size=rnn_hidden_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, input_seq):
        rnn_output, _ = self.rnn(input_seq)
        logits = self.linear(rnn_output)
        return logits


# Define the BERT-based model
class BERTFeatureExtractor(nn.Module):
    def __init__(self, num_bert_layers_to_use=1):
        super(BERTFeatureExtractor, self).__init__()
        self.num_bert_layers_to_use = num_bert_layers_to_use
        self.bert = bert
        self.bert_layers_to_use = [self.bert.encoder.layer[i] for i in range(num_bert_layers_to_use)]

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        selected_bert_layers_output = torch.cat(
            [bert_output.hidden_states[i] for i in range(self.num_bert_layers_to_use)], dim=-1)
        return selected_bert_layers_output


# Combine the RNN and BERT models for text generation
class TextGenerator(nn.Module):
    def __init__(self, rnn_hidden_size, vocab_size, num_bert_layers_to_use=1):
        super(TextGenerator, self).__init__()
        self.bert_feature_extractor = BERTFeatureExtractor(num_bert_layers_to_use)
        self.rnn_text_generator = RNNTextGenerator(rnn_hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_feature_extractor(input_ids, attention_mask)
        rnn_output = self.rnn_text_generator(bert_output)
        return rnn_output


# Example usage
# Assuming you have your input text and corresponding input_ids and attention_mask
input_text = "Hello, how are you?"
input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)

# Initialize the TextGenerator
rnn_hidden_size = 768  # Use BERT's hidden size (768)
vocab_size = tokenizer.vocab_size
num_bert_layers_to_use = 4  # Choose how many BERT layers to use, between 1 and 12
generator = TextGenerator(rnn_hidden_size, vocab_size, num_bert_layers_to_use)


# Custom Dataset for loading text data from file
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx].strip()
        # Tokenize and convert the sentence to input tensors
        encoded_data = tokenizer.encode_plus(sentence, add_special_tokens=True, padding='max_length', max_length=128, return_tensors='pt')
        input_ids = encoded_data['input_ids']
        attention_mask = encoded_data['attention_mask']
        return input_ids, attention_mask

# Prepare your training dataset (replace 'novel.txt' with your own file path)
train_dataset = TextDataset('../data/book1.txt')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training data (example data, replace with your own dataset)
train_input_ids = torch.tensor([tokenizer.encode("Hello, how are", add_special_tokens=True)])
train_attention_mask = torch.ones_like(train_input_ids)
train_target_ids = torch.tensor([tokenizer.encode(" you?", add_special_tokens=True)])

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

# Training loop (example)
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = generator(train_input_ids, train_attention_mask)
    loss = criterion(logits[:, :-1].reshape(-1, vocab_size), train_target_ids.reshape(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Example text generation after training
generated_ids = generator(train_input_ids, train_attention_mask)
generated_token_id = torch.argmax(generated_ids[0], dim=-1)

# Convert the token ID back to text using the tokenizer
generated_text = tokenizer.decode(generated_token_id.tolist())

print("Original Input Text:", input_text)
print("Generated Text:", generated_text)