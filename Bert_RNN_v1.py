import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import transformers
from tqdm import tqdm

# 手动输入第几次训练
T = eval(input("请输入第几次训练："))

BERT_PATH = './bert-base-chinese'
transformers.logging.set_verbosity_error()
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)

# 超参数
seq_length = 100 # 序列长度

# 读取数据构建字典并保存
def preprocess_text(text, seq_length):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    data = [char_to_idx[ch] for ch in text]

    num_sequences = len(data) - seq_length
    sequences = []
    for i in range(num_sequences):
        seq = data[i:i + seq_length + 1]
        sequences.append(seq)
    # 保存第T个字典
    np.save(f'char_to_idx{T}.npy', char_to_idx)
    np.save(f'idx_to_char{T}.npy', idx_to_char)

    return sequences, char_to_idx, idx_to_char


# Rnn文本生成
class RNNTextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(RNNTextGenerator, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        rnn_output, _ = self.rnn(input_seq)
        logits = self.linear(rnn_output)
        return logits



# 读取数据
with open('data/book.txt', 'r', encoding='utf-8') as file:
    text = file.read()
sequences, char_to_idx, idx_to_char = preprocess_text(text, seq_length)
vocab_size = len(char_to_idx)
sequences = torch.tensor(sequences, dtype=torch.long)




# 定义模型
input_size = vocab_size
hidden_size = 256

# Bert特征提取
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)
        return bert_output.last_hidden_state


# 定义模型
class RNN_BERT(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(RNN_BERT, self).__init__()
        self.bert_feature_extractor = BERT()
        self.rnn_text_generator = RNNTextGenerator(input_size=768, hidden_size=hidden_size, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_feature_extractor(input_ids, attention_mask)
        rnn_output = self.rnn_text_generator(bert_output)
        return rnn_output


# 训练
generator = RNN_BERT(hidden_size, vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
num_epochs = 10
batch_size = 32
loss_list = []
epoch_count = 1
for epoch in range(num_epochs):
    generator.train()
    total_loss = 0

    with tqdm(total=len(sequences) // batch_size) as pbar:
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            input_ids = batch_sequences[:, :-1]  # Input sequence (exclude last character)
            attention_mask = (input_ids != 0).type(torch.long)  # Attention mask to ignore padding tokens
            target_ids = batch_sequences[:, 1:]  # Target sequence (exclude first character)

            optimizer.zero_grad()
            logits = generator(input_ids, attention_mask)
            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_list.append(loss.item())
            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item()),'Epoch':epoch_count})
            pbar.update(1)
    epoch_count += 1

    average_loss = total_loss / (len(sequences) // batch_size)

# 保存模型
torch.save(generator.state_dict(), f'model_{T}.pth')


# 读取模型
T_ = eval(input("请输入读取第几次训练："))
char_to_idx = np.load(f'char_to_idx{T_}.npy', allow_pickle=True).item()
idx_to_char = np.load(f'idx_to_char{T_}.npy', allow_pickle=True).item()

# 加载模型
hidden_size = 256
vocab_size = len(char_to_idx)
generator = RNN_BERT(hidden_size, vocab_size)

generator.load_state_dict(torch.load(f'model_{T_}.pth', map_location=torch.device('cpu')))
generator.eval()

# 生成文本
def generate_text(prompt, max_length=200, temperature=1.0):
    input_ids = torch.tensor([[char_to_idx[ch] for ch in prompt]], dtype=torch.long)
    attention_mask = (input_ids != 0).type(torch.long)

    with torch.no_grad():
        output_text = prompt

        for _ in range(max_length):
            logits = generator(input_ids, attention_mask)
            logits = logits[:, -1, :]  # Consider the last token in the output sequence
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = idx_to_char[next_token_idx.item()]

            if next_token == '<EOS>':
                break

            output_text += next_token
            input_ids = torch.cat([input_ids, next_token_idx], dim=-1)
            attention_mask = (input_ids != 0).type(torch.long)

        return output_text

#  "从前有座山"
initial_prompt = "从前有座山"
generated_text = generate_text(initial_prompt, max_length=200, temperature=0.7)
print(generated_text)


# 目前能跑
# 估计显存不够
# 参数都没调
