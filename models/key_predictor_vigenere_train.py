import os
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import csv
import string

# allow importing from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# defaults
DEFAULT_DATA_PATH = r"C:\Users\nikhi\Downloads\Kryptos\data\output\vigenere.csv"
DEFAULT_MODEL_PATH = r"C:\Users\nikhi\Downloads\Kryptos\models\key_predictor_vigenere.pt"

ALL_CHARS = string.ascii_lowercase + " .,!?'"
# key_char mapping: letters start at 1, padding 0
CHAR2IDX = {c: i+1 for i, c in enumerate(ALL_CHARS)}
PAD_IDX = 0
VOCAB_SIZE = len(ALL_CHARS) + 1  # include padding token

# encryption helper
from encode.vigenere import vigenere_cipher, generate_random_key


def encode_text(text, max_len):
    text = text.lower()[:max_len]
    idxs = [CHAR2IDX.get(c, 0) for c in text]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    return idxs


def encode_key(key, max_key_len):
    key = key.lower()
    idxs = [CHAR2IDX.get(c, PAD_IDX) for c in key[:max_key_len]]
    if len(idxs) < max_key_len:
        idxs += [PAD_IDX] * (max_key_len - len(idxs))
    return idxs

class VigenereDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_len, max_key_len, augment=0):
        self.samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                plaintext = row["original_sentence"]
                ciphertext = row["encrypted_sentence"]
                key = row["key"]
                self.samples.append((encode_text(ciphertext, max_len), encode_key(key, max_key_len)))
                for _ in range(augment):
                    new_key = generate_random_key(min_len=1, max_len=max_key_len)
                    new_cipher = vigenere_cipher(plaintext, new_key)
                    self.samples.append((encode_text(new_cipher, max_len), encode_key(new_key, max_key_len)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class VigenereKeyPredictor(nn.Module):
    """Simple encoder-decoder for predicting the Vigenere key.

    Encoder reads the ciphertext; decoder generates the key sequence with optional
    teacher forcing during training.
    """
    def __init__(self, vocab_size, embed_size=32, hidden_size=128,
                 num_layers=1, max_len=10, bidirectional=True):
        super().__init__()
        self.max_len = max_len
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        # encoder LSTM (optionally bidirectional)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=bidirectional)
        enc_out_dim = hidden_size * (2 if bidirectional else 1)
        # decoder LSTM
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # project encoder outputs to decoder hidden size for attention
        self.enc_proj = nn.Linear(enc_out_dim, hidden_size)
        # attention + output layers (dec_vec + context both have hidden_size)
        self.attn = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, target=None, teacher_forcing_ratio=1.0):
        # src: (batch, seq_len)
        # target: (batch, key_len) or None
        batch = src.size(0)
        device = src.device
        src_emb = self.embed(src)
        enc_out, (h, c) = self.encoder(src_emb)  # enc_out: (batch, seq, enc_out_dim)

        # if encoder is bidirectional, combine directions for decoder init
        if self.bidirectional:
            # h,c: (num_layers * num_directions, batch, hidden)
            num_dirs = 2
            h = h.view(self.num_layers, num_dirs, batch, self.hidden_size).sum(dim=1)
            c = c.view(self.num_layers, num_dirs, batch, self.hidden_size).sum(dim=1)

        # prepare initial decoder input (use PAD_IDX as start token)
        input_tok = torch.full((batch, 1), PAD_IDX, dtype=torch.long, device=device)
        input_emb = self.embed(input_tok)  # (batch,1,embed)

        outputs = []
        hidden = (h, c)
        # enc_out for attention: (batch, seq_len, enc_out_dim)
        # project encoder outputs to decoder hidden size
        enc_proj = self.enc_proj(enc_out)  # (batch, seq_len, hidden_size)
        for t in range(self.max_len):
            out_dec, hidden = self.decoder(input_emb, hidden)  # out_dec (batch,1,hidden)
            dec_vec = out_dec.squeeze(1)  # (batch, hidden)

            # dot-product style attention scores
            # compute similarity between projected encoder outputs and decoder vector
            # scores: (batch, seq_len)
            scores = torch.sum(enc_proj * dec_vec.unsqueeze(1), dim=2)
            attn_weights = torch.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq_len,1)
            context = torch.sum(attn_weights * enc_proj, dim=1)  # (batch, hidden_size)

            # combine decoder vector and context
            combined = torch.tanh(self.attn(torch.cat([dec_vec, context], dim=1)))
            logits = self.out(combined)
            outputs.append(logits.unsqueeze(1))

            # decide next input: teacher forcing or previous prediction
            if target is not None and random.random() < teacher_forcing_ratio:
                next_input = target[:, t].unsqueeze(1)
            else:
                next_input = logits.argmax(dim=1).unsqueeze(1)
            input_emb = self.embed(next_input)

        return torch.cat(outputs, dim=1)  # (batch, max_len, vocab)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vigenere key predictor")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--max-key-len", type=int, default=10)
    parser.add_argument("--augment", type=int, default=0,
                        help="extra re-encryptions per sample")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tf-initial", type=float, default=0.9,
                        help="initial teacher forcing ratio")
    parser.add_argument("--tf-min", type=float, default=0.3,
                        help="minimum teacher forcing ratio")
    parser.add_argument("--tf-decay", type=float, default=0.05,
                        help="amount to decay teacher forcing each epoch")
    args = parser.parse_args()

    full_dataset = VigenereDataset(args.data_path, args.max_len, args.max_key_len, augment=args.augment)
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = VigenereKeyPredictor(VOCAB_SIZE,
                                 embed_size=args.embed_size,
                                 hidden_size=args.hidden_size,
                                 num_layers=args.num_layers,
                                 max_len=args.max_key_len)
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
        # compute teacher forcing ratio schedule (decays each epoch)
        tf_ratio = max(args.tf_min, args.tf_initial - (epoch-1) * args.tf_decay)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            # teacher forcing during training
            logits = model(X, target=y, teacher_forcing_ratio=tf_ratio)
            loss = 0.0
            for pos in range(args.max_key_len):
                loss += criterion(logits[:, pos, :], y[:, pos])
            loss = loss / float(args.max_key_len)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=2)
            correct += (preds == y).sum().item()
            total += y.numel()
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                logits = model(Xv, target=None, teacher_forcing_ratio=0.0)
                preds = logits.argmax(dim=2)
                val_correct += (preds == yv).sum().item()
                val_total += yv.numel()
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
