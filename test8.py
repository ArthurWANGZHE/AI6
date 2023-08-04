import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import transformers

BERT_PATH = './bert-base-chinese'
transformers.logging.set_verbosity_error()
# Load pre-trained BERT and tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
bert = BertModel.from_pretrained(BERT_PATH)

# Define the RNN-based text generator
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


# Data preprocessing function
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

    return sequences, char_to_idx, idx_to_char


# Load and preprocess text data (replace 'data/book1.txt' with your own file path)
with open('data/book1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

seq_length = 100  # Length of the input sequence for each training sample
sequences, char_to_idx, idx_to_char = preprocess_text(text, seq_length)
vocab_size = len(char_to_idx)

# Convert sequences to torch tensors
sequences = torch.tensor(sequences, dtype=torch.long)

# Define the RNN text generator model
input_size = vocab_size
hidden_size = 256
generator = RNNTextGenerator(input_size, hidden_size, vocab_size)


# Define the BERT feature extractor model
class BERTFeatureExtractor(nn.Module):
    def __init__(self):
        super(BERTFeatureExtractor, self).__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return bert_output.last_hidden_state


# Combine the BERT feature extractor and RNN text generator
class TextGeneratorWithBERT(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(TextGeneratorWithBERT, self).__init__()
        self.bert_feature_extractor = BERTFeatureExtractor()
        self.rnn_text_generator = RNNTextGenerator(input_size=768, hidden_size=hidden_size, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_feature_extractor(input_ids, attention_mask)
        rnn_output = self.rnn_text_generator(bert_output)
        return rnn_output


# Initialize the combined TextGeneratorWithBERT
generator = TextGeneratorWithBERT(hidden_size, vocab_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    generator.train()  # Set the model in training mode
    total_loss = 0

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

    average_loss = total_loss / (len(sequences) // batch_size)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

# Save the trained model
torch.save(generator.state_dict(), 'text_generator_model.pth')

# Save character mapping dictionaries
np.save('char_to_idx.npy', char_to_idx)
np.save('idx_to_char.npy', idx_to_char)

# Load character mapping dictionaries
char_to_idx = np.load('char_to_idx.npy', allow_pickle=True).item()
idx_to_char = np.load('idx_to_char.npy', allow_pickle=True).item()

# Initialize the combined TextGeneratorWithBERT
hidden_size = 256
vocab_size = len(char_to_idx)
generator = TextGeneratorWithBERT(hidden_size, vocab_size)

# Load the trained model
generator.load_state_dict(torch.load('text_generator_model.pth', map_location=torch.device('cpu')))
generator.eval()  # Set the model in evaluation mode

# Function to generate text given an initial prompt
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

# Generate text with initial prompt "从前有座山"
initial_prompt = "从前有座山"
generated_text = generate_text(initial_prompt, max_length=200, temperature=0.7)
print(generated_text)