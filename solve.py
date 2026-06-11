#!/usr/bin/env python
"""Command-line entry point for identifying and decoding classical ciphers.

Examples
--------
    python solve.py "wkh txlfn eurzq ira"
    python solve.py --cipher vigenere --key lemon "rijvs uyvjn"   # force decode
    echo "ciphertext" | python solve.py -
    python solve.py --all "wkh txlfn eurzq ira"                    # show every candidate
"""

from __future__ import annotations

import argparse
import sys

from kryptos import ciphers
from kryptos.fitness import get_fitness
from kryptos.pipeline import decode
from kryptos.solvers import SOLVERS


def _read_text(value: str) -> str:
    if value == "-":
        return sys.stdin.read().strip()
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ciphertext", help="ciphertext to decode, or '-' to read stdin")
    p.add_argument("--cipher", choices=ciphers.CIPHERS,
                   help="force a specific cipher instead of auto-detecting")
    p.add_argument("--key", help="decrypt with a known key (requires --cipher)")
    p.add_argument("--all", action="store_true",
                   help="show every cipher's best candidate, ranked")
    args = p.parse_args(argv)

    ciphertext = _read_text(args.ciphertext)
    if not ciphertext:
        p.error("empty ciphertext")

    # Known key -> just decrypt.
    if args.key is not None:
        if not args.cipher:
            p.error("--key requires --cipher")
        plain = ciphers.get(args.cipher).decrypt(ciphertext, args.key)
        print(plain)
        return 0

    fit = get_fitness()

    # Forced cipher -> run only that solver.
    if args.cipher:
        res = SOLVERS[args.cipher](ciphertext, fit)
        print(f"cipher: {res.cipher}")
        print(f"key:    {res.key}")
        print(f"plaintext:\n{res.plaintext}")
        return 0

    result = decode(ciphertext, fit)
    if args.all:
        print(f"{'cipher':10} {'score':>8}  key")
        for r in result.candidates:
            mark = "*" if r.cipher == result.cipher else " "
            print(f"{mark}{r.cipher:9} {r.score:8.3f}  {r.key}")
        print()
    print(f"Predicted cipher: {result.cipher} (confidence {result.confidence:.0%})")
    print(f"Key:              {result.key}")
    print(f"Plaintext:\n{result.plaintext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
