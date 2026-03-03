import torch
import torch.nn as nn

EMBED_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 1
VOCAB_SIZE = 32  # match Caesar training vocab
NUM_CLASSES = 26  # shifts 0-25

class CaesarKeyPredictor(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
