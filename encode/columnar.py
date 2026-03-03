# encode/columnar.py
import csv
import random
import math
import os

INPUT_PATH = r"C:\Users\nikhi\Downloads\Kryptos\data\input\sentences.tsv"
OUTPUT_DIR = r"C:\Users\nikhi\Downloads\Kryptos\data\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "columnar.csv")

import argparse

def generate_numeric_key():
    length = random.randint(3, 9)
    digits = list(range(1, length + 1))
    random.shuffle(digits)
    return digits  # list of ints

def columnar_transposition_encrypt(text, key_digits):
    text = text.lower()
    n_cols = len(key_digits)
    n_rows = math.ceil(len(text) / n_cols)

    # Pad with 'x'
    padded_length = n_rows * n_cols
    text += 'x' * (padded_length - len(text))

    # Build grid row-wise
    grid = [text[i:i+n_cols] for i in range(0, len(text), n_cols)]

    ciphertext = ""
    for digit in sorted(key_digits):
        col_index = key_digits.index(digit)
        for row in grid:
            ciphertext += row[col_index]

    return ciphertext

def main():
    parser = argparse.ArgumentParser(description="Encode sentences with a Columnar cipher")
    parser.add_argument("--count", type=int, default=50000,
                        help="number of samples to generate")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many lines from the input")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as tsvfile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:

        reader = csv.reader(tsvfile, delimiter="\t")
        writer = csv.writer(csvfile)
        writer.writerow(["original_sentence", "encrypted_sentence", "key", "cipher"])

        for _ in range(args.start):
            next(reader, None)

        count = 0
        for row in reader:
            if count >= args.count:
                break
            if not row:
                continue

            original = row[0].strip()
            if not original:
                continue

            key_digits = generate_numeric_key()
            encrypted = columnar_transposition_encrypt(original, key_digits)
            key_string = ''.join(str(d) for d in key_digits)

            writer.writerow([original, encrypted, key_string, "columnar"])
            count += 1

    print(f"Finished encoding {count} lines. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()