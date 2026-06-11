"""English-language fitness scoring used to rank candidate decryptions.

The core signal is the **quadgram log-probability** of a piece of text: the
sum over all overlapping 4-letter sequences of ``log10 P(quadgram)``.  Real
English scores far higher than transposed/substituted gibberish, which is what
lets the solvers pick the correct key out of thousands of candidates.

Statistics are estimated once from the project corpus
(``data/input/sentences.tsv``) and cached to ``kryptos/data/english_stats.npz``.
If the cache is missing it is built on first use; if the corpus is also missing
we fall back to a small built-in letter/bigram model so the package still runs.
"""

from __future__ import annotations

import csv
import os
import string
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np

_LOWER = string.ascii_lowercase
_A = ord("a")
# Map a-z -> 0..25, everything else -> -1 (filtered out).
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _c in enumerate(_LOWER):
    _CODE[ord(_c)] = _i

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_PKG_DIR)
DATA_DIR = os.path.join(_PKG_DIR, "data")
CACHE_PATH = os.path.join(DATA_DIR, "english_stats.npz")
DEFAULT_CORPUS = os.path.join(_PROJECT_DIR, "data", "input", "sentences.tsv")

# Standard English letter frequencies (%), used as a fallback / sanity anchor.
_FALLBACK_FREQ = np.array([
    8.17, 1.49, 2.78, 4.25, 12.70, 2.23, 2.02, 6.09, 6.97, 0.15, 0.77, 4.03,
    2.41, 6.75, 7.51, 1.93, 0.10, 5.99, 6.33, 9.06, 2.76, 0.98, 2.36, 0.15,
    1.97, 0.07,
], dtype=np.float64)
_FALLBACK_FREQ /= _FALLBACK_FREQ.sum()


def _letters_codes(text: str) -> np.ndarray:
    """Return the a-z letters of ``text`` as int codes 0..25 (others dropped)."""
    if not text:
        return np.empty(0, dtype=np.int64)
    arr = np.frombuffer(text.lower().encode("latin-1", "ignore"), dtype=np.uint8)
    codes = _CODE[arr]
    return codes[codes >= 0]


class EnglishFitness:
    """Score text by how English-like it looks.

    Parameters
    ----------
    quad, bi, uni:
        log10-probability tables for quadgrams (26**4,), bigrams (26, 26) and
        unigram frequencies (26,).
    quad_floor, bi_floor:
        log10-probability assigned to never-seen quadgrams/bigrams.
    """

    def __init__(self, quad, bi, uni, quad_floor, bi_floor):
        self.quad = np.asarray(quad, dtype=np.float32)
        self.bi = np.asarray(bi, dtype=np.float32)
        self.uni = np.asarray(uni, dtype=np.float64)
        self.quad_floor = float(quad_floor)
        self.bi_floor = float(bi_floor)

    # -- scoring ---------------------------------------------------------- #
    def score(self, text: str) -> float:
        """Total quadgram log-probability (higher = more English)."""
        codes = _letters_codes(text)
        if codes.size < 4:
            # Too short for quadgrams: fall back to bigram evidence.
            return self._bigram_score(codes)
        idx = (codes[:-3] * 17576 + codes[1:-2] * 676
               + codes[2:-1] * 26 + codes[3:])
        return float(self.quad[idx].sum())

    def score_codes(self, codes: np.ndarray) -> float:
        """Quadgram score for a pre-computed code array (-1 marks non-letters).

        Equivalent to :meth:`score` but skips string decoding, so the columnar
        solver can rescore thousands of column orderings cheaply.
        """
        c = codes[codes >= 0]
        if c.size < 4:
            return self.bi_floor * max(1, int(c.size))
        idx = c[:-3] * 17576 + c[1:-2] * 676 + c[2:-1] * 26 + c[3:]
        return float(self.quad[idx].sum())

    def normalized(self, text: str) -> float:
        """Average quadgram log-probability per quadgram (length-independent)."""
        codes = _letters_codes(text)
        n = codes.size - 3
        if n <= 0:
            return self.quad_floor
        idx = (codes[:-3] * 17576 + codes[1:-2] * 676
               + codes[2:-1] * 26 + codes[3:])
        return float(self.quad[idx].mean())

    def _bigram_score(self, codes: np.ndarray) -> float:
        if codes.size < 2:
            return self.bi_floor * max(1, codes.size)
        return float(self.bi[codes[:-1], codes[1:]].sum())

    def column_adjacency(self, left: str, right: str) -> float:
        """Bigram score of placing column ``left`` immediately before ``right``.

        Used by the columnar solver to order columns: for each row, the last
        char of ``left`` is followed by the first char of ``right``.
        """
        lc = _letters_codes_keep_len(left)
        rc = _letters_codes_keep_len(right)
        total = 0.0
        for a, b in zip(lc, rc):
            if a >= 0 and b >= 0:
                total += float(self.bi[a, b])
            else:
                total += self.bi_floor
        return total

    # -- index of coincidence (a cipher-type feature, lives here for reuse) #
    def index_of_coincidence(self, text: str) -> float:
        codes = _letters_codes(text)
        n = codes.size
        if n < 2:
            return 0.0
        counts = np.bincount(codes, minlength=26)
        return float((counts * (counts - 1)).sum() / (n * (n - 1)))

    # -- construction ----------------------------------------------------- #
    @classmethod
    def load(cls, cache_path: str = CACHE_PATH,
             corpus_path: str = DEFAULT_CORPUS,
             build_if_missing: bool = True) -> "EnglishFitness":
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return cls(data["quad"], data["bi"], data["uni"],
                       float(data["quad_floor"]), float(data["bi_floor"]))
        if build_if_missing and os.path.exists(corpus_path):
            stats = build_stats(corpus_path)
            save_stats(stats, cache_path)
            return cls(**stats)
        return cls._fallback()

    @classmethod
    def _fallback(cls) -> "EnglishFitness":
        """A crude model from letter frequencies when no corpus is available."""
        uni = _FALLBACK_FREQ.copy()
        loguni = np.log10(uni)
        # Independence assumption: P(quadgram) = prod of letter freqs.
        bi = loguni[:, None] + loguni[None, :]
        quad = np.zeros(26 ** 4, dtype=np.float32)
        codes = np.arange(26)
        # Build via broadcasting in a memory-friendly way.
        base = loguni
        for a in range(26):
            block = base[a] + base[:, None, None] + base[None, :, None] + base[None, None, :]
            quad[a * 17576:(a + 1) * 17576] = block.reshape(-1)
        floor = float(loguni.min() * 4)
        return cls(quad, bi.astype(np.float32), uni, floor, float(loguni.min() * 2))


def _letters_codes_keep_len(text: str) -> np.ndarray:
    """Codes for each char (non-letters -> -1), preserving length/alignment."""
    if not text:
        return np.empty(0, dtype=np.int64)
    arr = np.frombuffer(text.lower().encode("latin-1", "ignore"), dtype=np.uint8)
    return _CODE[arr]


def _iter_corpus_lines(corpus_path: str, max_lines: Optional[int]) -> Iterable[str]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_lines is not None and i >= max_lines:
                break
            if row and row[0]:
                yield row[0]


def build_stats(corpus_path: str = DEFAULT_CORPUS, max_lines: Optional[int] = 200_000) -> dict:
    """Estimate quad/bi/uni statistics from the corpus. Returns a dict of arrays."""
    quad_counts = np.zeros(26 ** 4, dtype=np.float64)
    bi_counts = np.zeros((26, 26), dtype=np.float64)
    uni_counts = np.zeros(26, dtype=np.float64)

    for line in _iter_corpus_lines(corpus_path, max_lines):
        codes = _letters_codes(line)
        if codes.size == 0:
            continue
        uni_counts += np.bincount(codes, minlength=26)
        if codes.size >= 2:
            np.add.at(bi_counts, (codes[:-1], codes[1:]), 1)
        if codes.size >= 4:
            idx = (codes[:-3] * 17576 + codes[1:-2] * 676
                   + codes[2:-1] * 26 + codes[3:])
            np.add.at(quad_counts, idx, 1)

    quad_total = quad_counts.sum()
    bi_total = bi_counts.sum()
    uni_total = uni_counts.sum()
    if quad_total == 0:
        raise RuntimeError(f"No letters found in corpus {corpus_path!r}")

    # Laplace-style floor for unseen n-grams.
    quad_floor = np.log10(0.01 / quad_total)
    bi_floor = np.log10(0.01 / bi_total)

    quad = np.full(26 ** 4, quad_floor, dtype=np.float32)
    seen = quad_counts > 0
    quad[seen] = np.log10(quad_counts[seen] / quad_total).astype(np.float32)

    bi = np.full((26, 26), bi_floor, dtype=np.float32)
    seen_bi = bi_counts > 0
    bi[seen_bi] = np.log10(bi_counts[seen_bi] / bi_total).astype(np.float32)

    uni = (uni_counts / uni_total).astype(np.float64)

    return {
        "quad": quad,
        "bi": bi,
        "uni": uni,
        "quad_floor": float(quad_floor),
        "bi_floor": float(bi_floor),
    }


def save_stats(stats: dict, cache_path: str = CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **stats)


@lru_cache(maxsize=1)
def get_fitness() -> EnglishFitness:
    """Process-wide cached fitness model."""
    return EnglishFitness.load()
