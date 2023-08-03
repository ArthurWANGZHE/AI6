import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')


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
rnn_hidden_size = 256  # You can adjust this as per your requirement
vocab_size = tokenizer.vocab_size
num_bert_layers_to_use = 4  # Choose how many BERT layers to use, between 1 and 12
generator = TextGenerator(rnn_hidden_size, vocab_size, num_bert_layers_to_use)

# Generate text
logits = generator(input_ids, attention_mask)
predicted_token_id = torch.argmax(logits, dim=-1)

# Convert the token ID back to text using the tokenizer
predicted_text = tokenizer.decode(predicted_token_id[0].tolist())

print("Original Input Text:", input_text)
print("Generated Text:", predicted_text)