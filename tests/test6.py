import torch
import transformers
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW


BERT_PATH = '../bert-base-chinese'
transformers.logging.set_verbosity_error()
# Load pre-trained BERT and tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)


# Define a simple RNN-based text generator with BERT embeddings
class RNNWithBERT(nn.Module):
    def __init__(self, bert, rnn_hidden_size, vocab_size, num_bert_layers_to_use=1):
        super(RNNWithBERT, self).__init__()
        self.num_bert_layers_to_use = num_bert_layers_to_use
        self.bert = bert
        self.rnn = nn.LSTM(input_size=bert.config.hidden_size * num_bert_layers_to_use,
                           hidden_size=rnn_hidden_size,
                           num_layers=1,
                           batch_first=True)
        self.linear = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        # Pass input through BERT and get all hidden layers
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[2]

        # Concatenate specified BERT layers' outputs
        selected_bert_layers_output = torch.cat([bert_output[i] for i in range(self.num_bert_layers_to_use)], dim=-1)

        # Use the concatenated BERT layers' outputs as input to the RNN
        rnn_output, _ = self.rnn(selected_bert_layers_output.unsqueeze(1))

        # Pass RNN output through the linear layer to generate logits
        logits = self.linear(rnn_output.squeeze(1))
        return logits


# Example usage
# Assuming you have your input text and corresponding input_ids and attention_mask
input_text = "从前有座山"
input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)
attention_mask = torch.ones_like(input_ids)

# Initialize the RNNWithBERT model
rnn_hidden_size = 256  # You can adjust this as per your requirement
vocab_size = tokenizer.vocab_size
num_bert_layers_to_use = 4  # Choose how many BERT layers to use, between 1 and 12
generator = RNNWithBERT(bert, rnn_hidden_size, vocab_size, num_bert_layers_to_use)

# Generate text
logits = generator(input_ids, attention_mask)
predicted_token_id = torch.argmax(logits, dim=-1)

# Convert the token ID back to text using the tokenizer
predicted_text = tokenizer.decode(predicted_token_id[0].tolist())

print("Original Input Text:", input_text)
print("Generated Text:", predicted_text)