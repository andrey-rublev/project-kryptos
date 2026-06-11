"""Kryptos: identify and break four classical ciphers.

Caesar, Vigenere, Skip (modular-walk transposition) and Columnar transposition.

The public surface is intentionally small::

    from kryptos import decode, identify_cipher

    result = decode(ciphertext)
    print(result.cipher, result.key, result.plaintext)

Decoding is done with classical cryptanalysis (frequency analysis, Index of
Coincidence, transposition column matching) scored by an English-language
fitness model built from the project's own corpus.  This is dramatically more
reliable than guessing keys with a neural network, and it needs no training.
"""

from .ciphers import caesar, vigenere, skip, columnar, CIPHERS
from .pipeline import decode, identify_cipher, DecodeResult

__all__ = [
    "caesar",
    "vigenere",
    "skip",
    "columnar",
    "CIPHERS",
    "decode",
    "identify_cipher",
    "DecodeResult",
]
