"""Batch-decode every ciphertext in one or more CSV files.

Reads CSVs that contain an ``encrypted_sentence`` column (the format produced by
the ``encode/`` generators) and writes ``cipher``, ``key`` and ``plaintext`` for
each row using the ``kryptos`` cryptanalysis package.

    python models/ai_decoder.py data/output/caesar.csv --limit 100
    python models/ai_decoder.py data/output/*.csv -o decoded.csv
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kryptos.fitness import get_fitness  # noqa: E402
from kryptos.pipeline import decode      # noqa: E402

COLUMN = "encrypted_sentence"


def iter_rows(paths, limit):
    seen = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if COLUMN not in (reader.fieldnames or []):
                print(f"  skip {path}: no '{COLUMN}' column", file=sys.stderr)
                continue
            for row in reader:
                yield os.path.basename(path), row
                seen += 1
                if limit and seen >= limit:
                    return


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="CSV file(s) or globs to decode")
    p.add_argument("-o", "--output", default="data/output/decoded_results.csv")
    p.add_argument("--limit", type=int, default=0, help="max rows total (0 = all)")
    args = p.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    fit = get_fitness()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_ok = 0
    with open(args.output, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "source", "ciphertext", "true_cipher", "predicted_cipher",
            "key", "confidence", "plaintext"])
        writer.writeheader()
        for source, row in iter_rows(paths, args.limit):
            ct = row[COLUMN]
            result = decode(ct, fit)
            writer.writerow({
                "source": source,
                "ciphertext": ct,
                "true_cipher": row.get("cipher", ""),
                "predicted_cipher": result.cipher,
                "key": result.key,
                "confidence": f"{result.confidence:.3f}",
                "plaintext": result.plaintext,
            })
            n_ok += 1
            if row.get("cipher") == result.cipher:
                pass
    print(f"Decoded {n_ok} rows -> {args.output}")


if __name__ == "__main__":
    main()
