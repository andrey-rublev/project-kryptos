# encode/caesar.py
import csv
import random
import string
import os

INPUT_PATH = r"C:\Users\nikhi\Downloads\Kryptos\data\input\sentences.tsv"
OUTPUT_DIR = r"C:\Users\nikhi\Downloads\Kryptos\data\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "caesar.csv")

# allow parameters from command line
import argparse

def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char in string.ascii_lowercase:  # only shift a-z
            new_index = (ord(char) - ord('a') + shift) % 26
            result += chr(new_index + ord('a'))
        else:
            result += char
    return result

def main():
    parser = argparse.ArgumentParser(description="Encode sentences with a Caesar cipher")
    parser.add_argument("--count", type=int, default=50000,
                        help="number of samples to generate")
    parser.add_argument("--start", type=int, default=0,
                        help="skip this many lines from the input before encoding")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as tsvfile, \
         open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:

        reader = csv.reader(tsvfile, delimiter="\t")
        writer = csv.writer(csvfile)

        writer.writerow([
            "original_sentence",
            "encrypted_sentence",
            "shift",
            "cipher"
        ])

        # skip lines if requested
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

            shift = random.randint(1, 25)
            encrypted = caesar_cipher(original, shift)

            writer.writerow([
                original,
                encrypted,
                shift,
                "caesar"
            ])

            count += 1

    print(f"Finished encoding {count} lines. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()