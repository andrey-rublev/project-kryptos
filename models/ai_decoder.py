# ai_decoder_safe.py
import os
import csv
import importlib
import torch
from cipher_classifier import CipherClassifier, encode_text, VOCAB_SIZE, CIPHERS
from models.key_predictor_caesar_train import CaesarKeyPredictor, encode_text as encode_text_key

DATA_DIR = r"C:\Users\nikhi\Downloads\Kryptos\data\output"
DECODE_FOLDER = r"C:\Users\nikhi\Downloads\Kryptos\decode"
CLASSIFIER_MODEL_PATH = r"C:\Users\nikhi\Downloads\Kryptos\models\cipher_classifier.pt"
CAESAR_KEY_MODEL_PATH = r"C:\Users\nikhi\Downloads\Kryptos\models\key_predictor_caesar.pt"
PROCESSED_LOG = os.path.join(DATA_DIR, "processed_files.txt")  # tracks processed CSVs
OUTPUT_FILE = os.path.join(DATA_DIR, "decoded_results.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cipher_classifier = CipherClassifier(VOCAB_SIZE)
cipher_classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
cipher_classifier.to(device)
cipher_classifier.eval()

key_models = {}
key_models["caesar"] = CaesarKeyPredictor(VOCAB_SIZE)
key_models["caesar"].load_state_dict(torch.load(CAESAR_KEY_MODEL_PATH, map_location=device))
key_models["caesar"].to(device)
key_models["caesar"].eval()

def predict_cipher(text):
    with torch.no_grad():
        encoded = torch.tensor([encode_text(text)], dtype=torch.long).to(device)
        logits = cipher_classifier(encoded)
        pred = torch.argmax(logits, dim=1).item()
        return CIPHERS[pred]

def predict_key(cipher_name, text):
    model = key_models[cipher_name]
    with torch.no_grad():
        encoded = torch.tensor([encode_text_key(text)], dtype=torch.long).to(device)
        logits = model(encoded)
        pred = torch.argmax(logits, dim=1).item()
        return pred

def decode_text(cipher_name, key, ciphertext):
    decoder = importlib.import_module(f'decode.{cipher_name}')
    return decoder.decrypt(ciphertext, key)

if os.path.exists(PROCESSED_LOG):
    with open(PROCESSED_LOG, "r") as f:
        processed_files = set(line.strip() for line in f.readlines())
else:
    processed_files = set()

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f not in processed_files]
print(f"Files to process: {csv_files}")

decoded_results = []

for fname in csv_files:
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ciphertext = row["encrypted_sentence"]

            predicted_cipher = predict_cipher(ciphertext)

            # Only Caesar implemented here; extend for others
            if predicted_cipher == "caesar":
                predicted_key = predict_key(predicted_cipher, ciphertext)
                plaintext = decode_text(predicted_cipher, predicted_key, ciphertext)
            else:
                predicted_key = "N/A"
                plaintext = "N/A"

            decoded_results.append({
                "ciphertext": ciphertext,
                "cipher": predicted_cipher,
                "key": predicted_key,
                "plaintext": plaintext
            })

    # Update processed log
    processed_files.add(fname)
    with open(PROCESSED_LOG, "a") as f_log:
        f_log.write(fname + "\n")

fieldnames = ["ciphertext", "cipher", "key", "plaintext"]

if os.path.exists(OUTPUT_FILE):
    # Append new results
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in decoded_results:
            writer.writerow(row)
else:
    # Create new CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in decoded_results:
            writer.writerow(row)

print(f"Decoded {len(decoded_results)} entries. Results saved to {OUTPUT_FILE}")