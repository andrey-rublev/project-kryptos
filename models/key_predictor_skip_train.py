import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import csv
import string

# defaults
DEFAULT_DATA_PATH = r"C:\Users\nikhi\Downloads\Kryptos\data\output\skip.csv"
DEFAULT_MODEL_PATH = r"C:\Users\nikhi\Downloads\Kryptos\models\key_predictor_skip.pt"

ALL_CHARS = string.ascii_lowercase + " .,!?'"
CHAR2IDX = {c: i for i, c in enumerate(ALL_CHARS)}
VOCAB_SIZE = len(ALL_CHARS)
NUM_CLASSES = 11  # 0-10 skips


# make sure encode package is importable
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from encode.skip import skip_cipher_coprime


def encode_text(text, max_len):
    text = text.lower()[:max_len]
    idxs = [CHAR2IDX.get(c, 0) for c in text]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    return idxs


class SkipDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_len, augment=0):
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plaintext = row["original_sentence"]
                ciphertext = row["encrypted_sentence"]
                key = int(row["key"])
                self.samples.append((encode_text(ciphertext, max_len), key))
                for _ in range(augment):
                    new_key = random.randint(0, 10)
                    new_cipher = skip_cipher_coprime(plaintext, new_key)
                    self.samples.append((encode_text(new_cipher, max_len), new_key))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SkipKeyPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=128,
                 num_layers=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Skip key predictor")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--augment", type=int, default=0,
                        help="extra re-encryptions per sample")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    full_dataset = SkipDataset(args.data_path, args.max_len, augment=args.augment)
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = SkipKeyPredictor(VOCAB_SIZE,
                             embed_size=args.embed_size,
                             hidden_size=args.hidden_size,
                             num_layers=args.num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_val_acc = 0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                logits = model(Xv)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yv).sum().item()
                val_total += yv.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_acc)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {total_loss/len(dataloader):.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered")
                break

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    if best_state is not None:
        torch.save(best_state, args.model_path)
    else:
        torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}, best val_acc={best_val_acc:.4f}")
