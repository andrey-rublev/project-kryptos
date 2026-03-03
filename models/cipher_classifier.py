import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import string
import random

# utility for augmenting Caesar cases
letters = string.ascii_lowercase

def _caesar_shift(text, shift):
    out = []
    for ch in text.lower():
        if ch in letters:
            out.append(chr((ord(ch)-97+shift)%26 + 97))
        else:
            out.append(ch)
    return "".join(out)

DATA_DIR = r"C:\Users\nikhi\Downloads\Kryptos\data\output"
MODEL_PATH = r"C:\Users\nikhi\Downloads\Kryptos\models\cipher_classifier.pt"

CIPHERS = ["caesar", "vigenere", "skip", "columnar"]

ALL_CHARS = string.ascii_lowercase + string.digits + string.punctuation + " \n"
CHAR2IDX = {c: i+1 for i, c in enumerate(ALL_CHARS)}  # 0 = padding
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(CHAR2IDX) + 1  # +1 for padding

MAX_LEN = 200  # default truncate/pad ciphertexts

def encode_text(text, max_len=None):
    if max_len is None:
        max_len = MAX_LEN
    text = text.lower()
    indices = [CHAR2IDX.get(c, 0) for c in text][:max_len]
    # pad
    indices += [0] * (max_len - len(indices))
    return indices

class CipherDataset(Dataset):
    def __init__(self, files, augment_caesar=0, max_samples_per_class=0, max_len=None):
        """files: list of csv filenames; augment_caesar: number of extra random shifts to add for each caesar example."""
        self.data = []
        for file in files:
            path = os.path.join(DATA_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cipher_text = row["encrypted_sentence"]
                    label = CIPHERS.index(row["cipher"])
                    self.data.append((cipher_text, label))
                    # augmentation for Caesar
                    if label == 0 and augment_caesar > 0:
                        # apply additional random shifts
                        for _ in range(augment_caesar):
                            shift = random.randint(1,25)
                            shifted = _caesar_shift(cipher_text, shift)
                            self.data.append((shifted, label))

        # limit samples per class if requested (keep balanced subset)
        if max_samples_per_class and max_samples_per_class > 0:
            by_class = {i: [] for i in range(len(CIPHERS))}
            for text, lab in self.data:
                by_class[lab].append((text, lab))
            new_data = []
            for lab, examples in by_class.items():
                if len(examples) <= max_samples_per_class:
                    new_data.extend(examples)
                else:
                    new_data.extend(random.sample(examples, max_samples_per_class))
            self.data = new_data

        # store max_len for encoding later
        self._max_len = max_len if max_len is not None else MAX_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = torch.tensor(encode_text(text, max_len=self._max_len), dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return encoded, label

class CipherClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=4,
                 num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers>1 else 0)
        factor = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * factor, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        # take last layer's hidden state from both directions if bidirectional
        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]
        out = self.fc(h)
        return out

def train_model(augment_caesar=0, embed_dim=64, hidden_dim=128, num_layers=2,
                bidirectional=True, dropout=0.3, batch_size=64, epochs=10, lr=1e-3,
                use_class_weights=True, use_sampler=True, max_samples_per_class=0,
                max_len=None, num_workers=0, fp16=False):
    files = ["caesar.csv", "vigenere.csv", "skip.csv", "columnar.csv"]
    full = CipherDataset(files, augment_caesar=augment_caesar,
                         max_samples_per_class=max_samples_per_class,
                         max_len=max_len)
    # split train/val
    val_size = int(0.1 * len(full))
    train_size = len(full) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full, [train_size, val_size])

    # collect labels from train subset for weighting/sampling
    train_labels = []
    for _, lab in train_ds:
        if isinstance(lab, torch.Tensor):
            train_labels.append(int(lab.item()))
        else:
            train_labels.append(int(lab))

    num_classes = len(CIPHERS)
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes)
    # avoid zeros
    class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
    class_weights = 1.0 / class_counts.float()

    # per-sample weights for sampler
    sample_weights = [float(class_weights[l]) for l in train_labels]

    pin_memory = torch.cuda.is_available()
    if use_sampler:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    model = CipherClassifier(VOCAB_SIZE, embed_dim=embed_dim, hidden_dim=hidden_dim,
                             output_dim=num_classes, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if use_class_weights:
        weight_tensor = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler() if (fp16 and torch.cuda.is_available()) else None

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(X)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / total if total > 0 else 0.0

        # validation with per-class metrics and confusion matrix
        model.eval()
        val_correct = 0
        val_total = 0
        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                out = model(Xv)
                preds = out.argmax(dim=1)
                val_correct += (preds == yv).sum().item()
                val_total += yv.size(0)
                for t, p in zip(yv.view(-1), preds.view(-1)):
                    conf_mat[int(t), int(p)] += 1
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

        # per-class accuracy
        per_class_acc = []
        for i in range(num_classes):
            true_pos = conf_mat[i, i].item()
            total_i = conf_mat[i].sum().item()
            acc_i = (true_pos / total_i) if total_i > 0 else 0.0
            per_class_acc.append(acc_i)

        avg_loss = total_loss / (len(train_loader) if len(train_loader)>0 else 1)
        print(f"Epoch {epoch}/{epochs}, loss={avg_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        for i, acc_i in enumerate(per_class_acc):
            print(f" - {CIPHERS[i]} acc: {acc_i:.4f}")

    # save confusion matrix CSV for inspection
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    cm_path = os.path.join(os.path.dirname(MODEL_PATH), "cipher_classifier_confusion.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CIPHERS)
        for i, row in enumerate(conf_mat.tolist()):
            writer.writerow([CIPHERS[i]] + row)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}, confusion matrix saved to {cm_path}")

def predict_cipher(model, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        encoded = torch.tensor([encode_text(text)], dtype=torch.long).to(device)
        logits = model(encoded)
        pred = torch.argmax(logits, dim=1).item()
        return CIPHERS[pred]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train cipher classifier")
    parser.add_argument("--augment-caesar", type=int, default=0,
                        help="add random shifts to Caesar examples")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--no-bidirectional", action="store_true",
                        help="disable bidirectional LSTM")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="disable class-weighted loss")
    parser.add_argument("--no-sampler", action="store_true",
                        help="disable weighted random sampler for training")
    parser.add_argument("--max-samples-per-class", type=int, default=0,
                        help="limit number of samples per class for quick runs (0=no limit)")
    parser.add_argument("--max-len", type=int, default=0,
                        help="truncate/pad length for inputs (0=use default MAX_LEN)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader num_workers")
    parser.add_argument("--fp16", action="store_true", help="use mixed precision (FP16) if CUDA available")
    args = parser.parse_args()
    train_model(augment_caesar=args.augment_caesar,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                bidirectional=not args.no_bidirectional,
                dropout=args.dropout,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                use_class_weights=not args.no_class_weights,
                use_sampler=not args.no_sampler,
                max_samples_per_class=args.max_samples_per_class,
                max_len=(args.max_len if args.max_len>0 else None),
                num_workers=args.num_workers,
                fp16=args.fp16)