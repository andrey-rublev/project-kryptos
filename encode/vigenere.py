# encode/vigenere.py
import csv
import random
import string
import os

INPUT_PATH = r"C:\Users\nikhi\Downloads\Kryptos\data\input\sentences.tsv"
OUTPUT_DIR = r"C:\Users\nikhi\Downloads\Kryptos\data\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vigenere.csv")

import argparse

def generate_random_key(min_len=3, max_len=10):
    key_length = random.randint(min_len, max_len)
    key = ""
    for _ in range(key_length):
        key += chr(random.randint(0, 25) + ord('a'))
    return key

def vigenere_cipher(text, key):
    result = ""
    key_index = 0
    key_length = len(key)
    for char in text:
        if char in string.ascii_lowercase:
            shift = ord(key[key_index % key_length]) - ord('a')
            new_index = (ord(char) - ord('a') + shift) % 26
            result += chr(new_index + ord('a'))
            key_index += 1
        else:
            result += char
    return result

def main():
    parser = argparse.ArgumentParser(description="Encode sentences with a Vigenere cipher")
    parser.add_argument("--count", type=int, default=50000,
                        help="number of samples to generate")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many input lines before encoding")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as tsvfile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:

        reader = csv.reader(tsvfile, delimiter="\t")
        writer = csv.writer(csvfile)

        writer.writerow([
            "original_sentence",
            "encrypted_sentence",
            "key",
            "cipher"
        ])

        for _ in range(args.start):
            next(reader, None)

        count = 0
        for row in reader:
            if count >= args.count:
                break

            if not row:
                continue

            original = row[0].strip().lower()
            if not original:
                continue

            key = generate_random_key()
            encrypted = vigenere_cipher(original, key)

            writer.writerow([
                original,
                encrypted,
                key,
                "vigenere"
            ])
            count += 1

    print(f"Finished encoding {count} lines. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()