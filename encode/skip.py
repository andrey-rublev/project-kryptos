# encode/skip.py
import csv
import random
import os
from math import gcd

INPUT_PATH = r"C:\Users\nikhi\Downloads\Kryptos\data\input\sentences.tsv"
OUTPUT_DIR = r"C:\Users\nikhi\Downloads\Kryptos\data\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "skip.csv")

import argparse

def generate_coprime_key(length, min_skip=2, max_skip=10):
    possible_keys = [k for k in range(min_skip, max_skip + 1) if gcd(k, length) == 1]
    if not possible_keys:
        return 1
    return random.choice(possible_keys)

def skip_cipher_coprime(text, key):
    n = len(text)
    if n == 0:
        return ""
    result = []
    index = 0
    for _ in range(n):
        result.append(text[index])
        index = (index + key) % n
    return "".join(result)

def main():
    parser = argparse.ArgumentParser(description="Encode sentences with a Skip cipher")
    parser.add_argument("--count", type=int, default=50000,
                        help="number of samples to generate")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many lines from input before enciphering")
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

            original = row[0].strip().lower()
            if not original:
                continue

            key = generate_coprime_key(len(original), 2, 10)
            encrypted = skip_cipher_coprime(original, key)

            writer.writerow([original, encrypted, key, "skip"])
            count += 1

    print(f"Finished encoding {count} lines. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()