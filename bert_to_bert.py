import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

BERT_PATH = 'bert-base-chinese'


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class BertToBertModel(nn.Module):
    def __init__(self):
        super(BertToBertModel, self).__init__()
        self.encoder_bert = BertModel.from_pretrained(BERT_PATH)
        self.decoder_bert = BertModel.from_pretrained(BERT_PATH)
        self.linear = nn.Linear(self.encoder_bert.config.hidden_size, self.decoder_bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder_bert(input_ids=input_ids, attention_mask=attention_mask)
        encoded_hidden_states = encoder_outputs.last_hidden_state
        decoder_input = self.linear(encoded_hidden_states)

        # 使用 decoder_input 作为输入
        decoder_outputs = self.decoder_bert(input_ids=decoder_input, attention_mask=attention_mask)
        decoded_hidden_states = decoder_outputs.last_hidden_state
        return decoded_hidden_states


# 示例训练数据
train_data = ["这是一段中文小说。", "另一段小说内容。"]

# 数据预处理
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
input_ids = [tokenizer.encode(text, add_special_tokens=True) for text in train_data]
attention_masks = [[1] * len(ids) for ids in input_ids]

# 创建数据加载器
batch_size = 2
train_dataset = MyDataset(list(zip(input_ids, attention_masks)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型并将其移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertToBertModel().to(device)

# 训练循环
num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_input_ids, batch_attention_masks in train_loader:
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        loss = criterion(outputs.view(-1, outputs.size(-1)), batch_input_ids.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}")

print("Training finished.")
def generate_continuation(prompt_text, max_length=50):
    model.eval()

    # 将输入文本编码为input_ids
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    # 生成续写文本
    with torch.no_grad():
        for _ in range(max_length):
            attention_mask = torch.ones(input_ids.shape, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_ids = outputs.argmax(dim=-1)[:, -1]
            input_ids = torch.cat((input_ids, predicted_ids.unsqueeze(-1)), dim=-1)

    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    return generated_text


# 输入一个续写的起始句子，生成续写内容
prompt = "故事的开始"
generated_text = generate_continuation(prompt, max_length=100)
print(generated_text)