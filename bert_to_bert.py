import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

BERT_PATH = '../bert-base-chinese'
class BertToBertModel(nn.Module):
    def __init__(self):
        super(BertToBertModel, self).__init__()

        # 初始化编码器和解码器的Bert模型和分词器
        self.encoder_bert = BertModel.from_pretrained(BERT_PATH)
        self.decoder_bert = BertModel.from_pretrained(BERT_PATH)

        # 其他需要的层和参数
        self.linear = nn.Linear(self.encoder_bert.config.hidden_size, self.decoder_bert.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        # 编码输入文本
        encoder_outputs = self.encoder_bert(input_ids=input_ids, attention_mask=attention_mask)
        encoded_hidden_states = encoder_outputs.last_hidden_state

        # 使用线性层将编码后的表示转换为解码器的表示
        decoder_input = self.linear(encoded_hidden_states)

        # 解码
        decoder_outputs = self.decoder_bert(input_ids=decoder_input, attention_mask=attention_mask)
        decoded_hidden_states = decoder_outputs.last_hidden_state

        return decoded_hidden_states

# 初始化模型
model = BertToBertModel()

# 输入示例（需要进行中文分词）
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
text = "这是一段中文小说。"
input_ids = tokenizer.encode(text, add_special_tokens=True)
attention_mask = [1] * len(input_ids)
input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加batch维度
attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # 添加batch维度

# 前向传播
output = model(input_ids, attention_mask)

# 输出解码后的表示
print(output)