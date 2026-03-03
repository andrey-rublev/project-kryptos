import torch
import torch.nn as nn

# model for predicting a Vigenere key (padded to fixed length)
MAX_KEY_LEN = 10
# vocab includes padding index 0 plus actual characters
VOCAB_SIZE = 33   # letters plus pad
EMBED_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 1

class VigenereKeyPredictor(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 max_len=MAX_KEY_LEN):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # remember vocab size for reshape
        self.vocab_size = vocab_size
        # output will be expanded to (batch, max_len, vocab_size)
        self.fc = nn.Linear(hidden_size, vocab_size * max_len)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        out = self.fc(h)
        # reshape for convenience using stored vocab_size
        return out.view(-1, self.max_len, self.vocab_size)
