"""Canonical encrypt/decrypt implementations for the four supported ciphers.

These are the single source of truth used by the dataset generators, the
solvers and the tests, so encoding and decoding can never drift apart.

Conventions shared by every cipher:

* Only ASCII letters ``a-z`` are transformed; spaces, punctuation and digits
  pass through untouched (except Columnar, which pads with ``'x'``).
* Text is lower-cased before enciphering, matching the generated datasets.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from math import gcd
from typing import List, Sequence, Union

_LOWER = string.ascii_lowercase
_IS_LOWER = set(_LOWER)

CIPHERS = ("caesar", "vigenere", "skip", "columnar")


# --------------------------------------------------------------------------- #
# Caesar
# --------------------------------------------------------------------------- #
class caesar:
    """Shift every letter by a fixed amount (the key is an int 0-25)."""

    name = "caesar"

    @staticmethod
    def encrypt(text: str, shift: int) -> str:
        shift %= 26
        out = []
        for ch in text.lower():
            if ch in _IS_LOWER:
                out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def decrypt(ciphertext: str, shift: int) -> str:
        return caesar.encrypt(ciphertext, -int(shift))


# --------------------------------------------------------------------------- #
# Vigenere
# --------------------------------------------------------------------------- #
class vigenere:
    """Polyalphabetic shift; key is a lowercase word, advanced per letter."""

    name = "vigenere"

    @staticmethod
    def _shifts(key: str) -> List[int]:
        shifts = [ord(k) - 97 for k in key.lower() if k in _IS_LOWER]
        if not shifts:
            raise ValueError("Vigenere key must contain at least one letter")
        return shifts

    @staticmethod
    def encrypt(text: str, key: str, _sign: int = 1) -> str:
        shifts = vigenere._shifts(key)
        klen = len(shifts)
        out = []
        ki = 0
        for ch in text.lower():
            if ch in _IS_LOWER:
                s = _sign * shifts[ki % klen]
                out.append(chr((ord(ch) - 97 + s) % 26 + 97))
                ki += 1
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def decrypt(ciphertext: str, key: str) -> str:
        return vigenere.encrypt(ciphertext, key, _sign=-1)


# --------------------------------------------------------------------------- #
# Skip (modular-walk transposition)
# --------------------------------------------------------------------------- #
class skip:
    """Read characters by repeatedly stepping ``key`` positions (mod n).

    The transformation is a permutation of the characters, so it is only
    invertible when ``gcd(key, len(text)) == 1``.  The dataset generator
    guarantees this; :func:`is_invertible` lets callers check.
    """

    name = "skip"

    @staticmethod
    def is_invertible(length: int, key: int) -> bool:
        return length > 0 and key > 0 and gcd(key, length) == 1

    @staticmethod
    def encrypt(text: str, key: int) -> str:
        n = len(text)
        if n == 0:
            return ""
        out = []
        index = 0
        for _ in range(n):
            out.append(text[index])
            index = (index + key) % n
        return "".join(out)

    @staticmethod
    def decrypt(ciphertext: str, key: int) -> str:
        n = len(ciphertext)
        if n == 0:
            return ""
        result = [""] * n
        index = 0
        for ch in ciphertext:
            result[index] = ch
            index = (index + key) % n
        return "".join(result)


# --------------------------------------------------------------------------- #
# Columnar transposition
# --------------------------------------------------------------------------- #
def _key_to_digits(key: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(key, str):
        return [int(c) for c in key]
    return [int(d) for d in key]


class columnar:
    """Classic columnar transposition with a numeric permutation key.

    ``key`` is a permutation of ``1..k`` (as a string like ``"3142"`` or a list
    of ints).  Plaintext is written row-wise into ``k`` columns, padded with
    ``'x'``, then columns are read out in ascending key order.
    """

    name = "columnar"

    @staticmethod
    def encrypt(text: str, key: Union[str, Sequence[int]], pad: str = "x") -> str:
        digits = _key_to_digits(key)
        n_cols = len(digits)
        text = text.lower()
        n_rows = math.ceil(len(text) / n_cols) if text else 0
        text = text.ljust(n_rows * n_cols, pad)
        grid = [text[i:i + n_cols] for i in range(0, len(text), n_cols)]

        # Columns are emitted in the order given by the sorted key digits.
        order = sorted(range(n_cols), key=lambda c: digits[c])
        out = []
        for col in order:
            for row in grid:
                out.append(row[col])
        return "".join(out)

    @staticmethod
    def decrypt(ciphertext: str, key: Union[str, Sequence[int]], pad: str = "x") -> str:
        digits = _key_to_digits(key)
        n_cols = len(digits)
        if n_cols == 0:
            return ciphertext
        n_rows = math.ceil(len(ciphertext) / n_cols)
        text = ciphertext.ljust(n_rows * n_cols, pad)

        grid = [[""] * n_cols for _ in range(n_rows)]
        order = sorted(range(n_cols), key=lambda c: digits[c])
        pos = 0
        for col in order:
            for row in range(n_rows):
                grid[row][col] = text[pos]
                pos += 1
        plaintext = "".join("".join(row) for row in grid)
        return plaintext.rstrip(pad)


_BY_NAME = {c.name: c for c in (caesar, vigenere, skip, columnar)}


def get(name: str):
    """Return the cipher class for ``name`` (``caesar``/``vigenere``/...)."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown cipher {name!r}; expected one of {CIPHERS}")


@dataclass(frozen=True)
class Sample:
    """A single (plaintext, ciphertext, key) training/eval example."""

    cipher: str
    plaintext: str
    ciphertext: str
    key: object
