"""Cryptanalytic solvers that recover the key and plaintext for each cipher.

Every solver returns a :class:`SolveResult` and is scored with the shared
English-fitness model, so results are directly comparable across ciphers.
None of them require any trained model or knowledge of the key.

Techniques
----------
* **Caesar**   exhaustive search over 26 shifts.
* **Vigenere** key length from candidate search; each column solved by
  chi-squared frequency analysis against English.
* **Skip**     exhaustive search over coprime step sizes.
* **Columnar** for each column count, the optimal column ordering under a
  bigram-adjacency metric is found with a Held-Karp dynamic program, then
  rescored with quadgrams.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from math import gcd
from typing import List, Optional

import numpy as np

from . import ciphers
from .fitness import EnglishFitness, get_fitness, _letters_codes, _CODE

_LOWER = string.ascii_lowercase


@dataclass
class SolveResult:
    cipher: str
    key: object
    plaintext: str
    score: float          # normalized (per-quadgram) English fitness


# --------------------------------------------------------------------------- #
# Caesar
# --------------------------------------------------------------------------- #
def solve_caesar(ciphertext: str, fit: Optional[EnglishFitness] = None) -> SolveResult:
    fit = fit or get_fitness()
    best = None
    for shift in range(26):
        cand = ciphers.caesar.decrypt(ciphertext, shift)
        sc = fit.normalized(cand)
        if best is None or sc > best.score:
            best = SolveResult("caesar", shift, cand, sc)
    return best


# --------------------------------------------------------------------------- #
# Vigenere
# --------------------------------------------------------------------------- #
def _best_shift_for_column(col_codes: np.ndarray, eng_freq: np.ndarray) -> int:
    """Chi-squared: find the shift making the column match English freqs."""
    n = col_codes.size
    if n == 0:
        return 0
    counts = np.bincount(col_codes, minlength=26).astype(np.float64)
    expected = eng_freq * n
    expected = np.where(expected <= 0, 1e-9, expected)
    best_shift, best_chi = 0, None
    for shift in range(26):
        rolled = np.roll(counts, -shift)          # try decrypting by `shift`
        chi = ((rolled - expected) ** 2 / expected).sum()
        if best_chi is None or chi < best_chi:
            best_chi, best_shift = chi, shift
    return best_shift


def _reduce_key(key: str) -> str:
    """Collapse a key that is a repetition of a shorter unit (e.g. lemonlemon)."""
    n = len(key)
    for L in range(1, n):
        if n % L == 0 and key[:L] * (n // L) == key:
            return key[:L]
    return key


def _polish_vigenere_key(ciphertext: str, key: str, fit: EnglishFitness) -> str:
    """Hill-climb each key letter to maximize full quadgram fitness.

    Chi-squared per column is occasionally off by a letter on short columns;
    this cleans up those residual errors against the real language model.
    """
    key = list(key)
    best_score = fit.score(ciphers.vigenere.decrypt(ciphertext, "".join(key)))
    improved = True
    while improved:
        improved = False
        for i in range(len(key)):
            original = key[i]
            best_ch = original
            for c in range(26):
                key[i] = chr(c + 97)
                sc = fit.score(ciphers.vigenere.decrypt(ciphertext, "".join(key)))
                if sc > best_score:
                    best_score, best_ch, improved = sc, chr(c + 97), True
            key[i] = best_ch
    return "".join(key)


def solve_vigenere(ciphertext: str, fit: Optional[EnglishFitness] = None,
                   max_key_len: int = 16) -> SolveResult:
    fit = fit or get_fitness()
    codes = _letters_codes(ciphertext)
    eng_freq = fit.uni
    n = codes.size
    best = None
    upper = min(max_key_len, max(1, n // 2))
    for klen in range(1, upper + 1):
        shifts = []
        for offset in range(klen):
            col = codes[offset::klen]
            shifts.append(_best_shift_for_column(col, eng_freq))
        key = "".join(chr(s + 97) for s in shifts)
        key = _polish_vigenere_key(ciphertext, key, fit)
        cand = ciphers.vigenere.decrypt(ciphertext, key)
        sc = fit.normalized(cand)
        if best is None or sc > best.score:
            best = SolveResult("vigenere", _reduce_key(key), cand, sc)
    if best is None:                       # empty ciphertext edge case
        return SolveResult("vigenere", "a", ciphertext, fit.quad_floor)
    return best


# --------------------------------------------------------------------------- #
# Skip
# --------------------------------------------------------------------------- #
def solve_skip(ciphertext: str, fit: Optional[EnglishFitness] = None,
               max_key: int = 200) -> SolveResult:
    fit = fit or get_fitness()
    n = len(ciphertext)
    best = None
    upper = min(max_key, n - 1)
    for key in range(1, upper + 1):
        if gcd(key, n) != 1:               # only coprime steps are invertible
            continue
        cand = ciphers.skip.decrypt(ciphertext, key)
        sc = fit.normalized(cand)
        if best is None or sc > best.score:
            best = SolveResult("skip", key, cand, sc)
    if best is None:
        return SolveResult("skip", 1, ciphertext, fit.normalized(ciphertext))
    return best


# --------------------------------------------------------------------------- #
# Columnar
# --------------------------------------------------------------------------- #
def _held_karp_order(adj: np.ndarray) -> List[int]:
    """Return the column order maximizing total adjacency score (a TSP path)."""
    k = adj.shape[0]
    if k <= 1:
        return list(range(k))
    NEG = -1e18
    size = 1 << k
    dp = np.full((size, k), NEG, dtype=np.float64)
    par = np.full((size, k), -1, dtype=np.int64)
    for j in range(k):
        dp[1 << j, j] = 0.0
    for mask in range(size):
        row = dp[mask]
        for j in range(k):
            cur = row[j]
            if cur <= NEG / 2 or not (mask >> j) & 1:
                continue
            for nxt in range(k):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                val = cur + adj[j, nxt]
                if val > dp[nm, nxt]:
                    dp[nm, nxt] = val
                    par[nm, nxt] = j
    full = size - 1
    j = int(np.argmax(dp[full]))
    order = []
    mask = full
    while j != -1:
        order.append(j)
        pj = int(par[mask, j])
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return order


def _read_columns(blocks: List[str], order: List[int], n_rows: int) -> str:
    return "".join("".join(blocks[col][r] for col in order) for r in range(n_rows))


def _column_codes(blocks: List[str]) -> np.ndarray:
    """Matrix (k, n_rows) of letter codes per column; -1 for non-letters."""
    k = len(blocks)
    n_rows = len(blocks[0]) if blocks else 0
    mat = np.full((k, n_rows), -1, dtype=np.int64)
    for i, b in enumerate(blocks):
        arr = np.frombuffer(b.encode("latin-1", "ignore"), dtype=np.uint8)
        mat[i, :arr.size] = _CODE[arr]
    return mat


def _score_order(mat: np.ndarray, order: List[int], fit: EnglishFitness) -> float:
    # mat[order].T reads row-by-row across the reordered columns.
    return fit.score_codes(mat[order].T.reshape(-1))


def _hill_climb(order: List[int], mat: np.ndarray, fit: EnglishFitness):
    """Local search over swaps + insertions, scored by full quadgram fitness."""
    best = list(order)
    best_score = _score_order(mat, best, fit)
    k = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(k):                          # pairwise swaps
            for j in range(i + 1, k):
                cand = list(best)
                cand[i], cand[j] = cand[j], cand[i]
                sc = _score_order(mat, cand, fit)
                if sc > best_score:
                    best, best_score, improved = cand, sc, True
        for i in range(k):                          # relocate one column
            for j in range(k):
                if i == j:
                    continue
                cand = list(best)
                cand.insert(j, cand.pop(i))
                sc = _score_order(mat, cand, fit)
                if sc > best_score:
                    best, best_score, improved = cand, sc, True
    return best, best_score


def _search_order(blocks: List[str], fit: EnglishFitness) -> List[int]:
    """Find the column ordering maximizing English fitness.

    Seeds the search with the optimal bigram-adjacency ordering (Held-Karp) plus
    several random restarts, then hill-climbs each.  This reliably reaches the
    global optimum for the 2-12 column range without a factorial brute force.
    """
    k = len(blocks)
    mat = _column_codes(blocks)
    if k <= 2:
        cands = [list(range(k)), list(range(k - 1, -1, -1))]
        return max(cands, key=lambda o: _score_order(mat, o, fit))

    adj = np.empty((k, k), dtype=np.float64)
    for a in range(k):
        for b in range(k):
            adj[a, b] = -1e18 if a == b else fit.column_adjacency(blocks[a], blocks[b])

    rng = random.Random(0xC0FFEE)
    seeds = [_held_karp_order(adj)]
    for _ in range(max(6, k * 5)):
        perm = list(range(k))
        rng.shuffle(perm)
        seeds.append(perm)

    best, best_score = None, None
    for seed in seeds:
        order, score = _hill_climb(seed, mat, fit)
        if best_score is None or score > best_score:
            best, best_score = order, score
    return best


def _columnar_for_ncols(ciphertext: str, n_cols: int, fit: EnglishFitness) -> Optional[SolveResult]:
    n = len(ciphertext)
    if n_cols < 2 or n % n_cols != 0:
        return None
    n_rows = n // n_cols
    blocks = [ciphertext[i * n_rows:(i + 1) * n_rows] for i in range(n_cols)]

    order = _search_order(blocks, fit)
    plaintext = _read_columns(blocks, order, n_rows).rstrip("x")
    # Recover a valid key: output column i came from ciphertext block order[i],
    # and blocks are emitted in ascending key order, so digit = rank + 1.
    key_digits = [0] * n_cols
    for out_col, block_idx in enumerate(order):
        key_digits[out_col] = block_idx + 1
    key = "".join(str(d) for d in key_digits)
    return SolveResult("columnar", key, plaintext, fit.normalized(plaintext))


def solve_columnar(ciphertext: str, fit: Optional[EnglishFitness] = None,
                   min_cols: int = 2, max_cols: int = 12) -> SolveResult:
    fit = fit or get_fitness()
    best = None
    for n_cols in range(min_cols, max_cols + 1):
        res = _columnar_for_ncols(ciphertext, n_cols, fit)
        if res is None:
            continue
        if best is None or res.score > best.score:
            best = res
    if best is None:
        return SolveResult("columnar", "1", ciphertext, fit.normalized(ciphertext))
    return best


SOLVERS = {
    "caesar": solve_caesar,
    "vigenere": solve_vigenere,
    "skip": solve_skip,
    "columnar": solve_columnar,
}


def solve_all(ciphertext: str, fit: Optional[EnglishFitness] = None) -> List[SolveResult]:
    """Run every solver and return results sorted best-first."""
    fit = fit or get_fitness()
    results = [SOLVERS[name](ciphertext, fit) for name in ciphers.CIPHERS]
    results.sort(key=lambda r: r.score, reverse=True)
    return results
