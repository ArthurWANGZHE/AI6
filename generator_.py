import numpy as np
from global_config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def generate_text(model, start_text, length=100):
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