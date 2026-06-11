"""Correctness tests for the kryptos cipher and solver package.

Run with:  python -m pytest tests/ -v
"""

import os
import random
import sys
from math import gcd

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kryptos import ciphers
from kryptos.fitness import get_fitness
from kryptos.pipeline import decode, identify_cipher

# A few real English sentences with enough letters for cryptanalysis.
SENTENCES = [
    "the history of the ancient city was written by scholars who studied its many ruins",
    "she carefully placed the old photographs into the wooden box before closing the lid",
    "modern science depends on careful observation and the patient testing of every theory",
    "after the long winter the farmers prepared their fields for the coming planting season",
    "the orchestra rehearsed the symphony for several weeks before the opening night concert",
]


def _skip_key(n):
    keys = [k for k in range(2, 11) if gcd(k, n) == 1]
    return keys[0] if keys else 1


def _columnar_key(seed):
    rng = random.Random(seed)
    k = rng.randint(3, 9)
    digits = list(range(1, k + 1))
    rng.shuffle(digits)
    return digits


def _vigenere_key(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 8)
    return "".join(chr(rng.randint(0, 25) + 97) for _ in range(n))


# --------------------------------------------------------------------------- #
# Round-trip: decrypt(encrypt(x)) == x
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pt", SENTENCES)
def test_caesar_roundtrip(pt):
    for shift in (1, 7, 13, 25):
        assert ciphers.caesar.decrypt(ciphers.caesar.encrypt(pt, shift), shift) == pt


@pytest.mark.parametrize("pt", SENTENCES)
def test_vigenere_roundtrip(pt):
    for key in ("lemon", "abc", "secretkey"):
        assert ciphers.vigenere.decrypt(ciphers.vigenere.encrypt(pt, key), key) == pt


@pytest.mark.parametrize("pt", SENTENCES)
def test_skip_roundtrip(pt):
    key = _skip_key(len(pt))
    assert ciphers.skip.decrypt(ciphers.skip.encrypt(pt, key), key) == pt


@pytest.mark.parametrize("pt", SENTENCES)
def test_columnar_roundtrip(pt):
    key = _columnar_key(seed=len(pt))
    # Columnar pads with 'x'; compare ignoring trailing padding.
    out = ciphers.columnar.decrypt(ciphers.columnar.encrypt(pt, key), key)
    assert out.rstrip("x") == pt.rstrip("x")


def test_columnar_key_recovery_reencrypts():
    """A solved columnar key must reproduce the ciphertext when re-applied."""
    fit = get_fitness()
    pt = SENTENCES[0]
    ct = ciphers.columnar.encrypt(pt, _columnar_key(seed=3))
    res = decode(ct, fit)
    assert res.cipher == "columnar"
    assert ciphers.columnar.encrypt(res.plaintext, res.key) == ct


# --------------------------------------------------------------------------- #
# End-to-end solve: decode(encrypt(x)) recovers x and the cipher
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pt", SENTENCES)
def test_decode_caesar(pt):
    res = decode(ciphers.caesar.encrypt(pt, 7))
    assert res.cipher == "caesar"
    assert res.plaintext == pt


@pytest.mark.parametrize("pt", SENTENCES)
def test_decode_vigenere(pt):
    res = decode(ciphers.vigenere.encrypt(pt, _vigenere_key(seed=len(pt))))
    assert res.cipher == "vigenere"
    assert res.plaintext == pt


@pytest.mark.parametrize("pt", SENTENCES)
def test_decode_skip(pt):
    res = decode(ciphers.skip.encrypt(pt, _skip_key(len(pt))))
    assert res.cipher == "skip"
    assert res.plaintext == pt


@pytest.mark.parametrize("pt", SENTENCES)
def test_decode_columnar(pt):
    res = decode(ciphers.columnar.encrypt(pt, _columnar_key(seed=len(pt))))
    assert res.cipher == "columnar"
    assert res.plaintext.rstrip("x") == pt.rstrip("x")


def test_identify_returns_scores_for_all_ciphers():
    info = identify_cipher(ciphers.caesar.encrypt(SENTENCES[0], 5))
    assert info["cipher"] == "caesar"
    assert set(info["scores"]) == set(ciphers.CIPHERS)
    assert 0.0 <= info["confidence"] <= 1.0


def test_unknown_cipher_lookup_raises():
    with pytest.raises(KeyError):
        ciphers.get("playfair")
