"""High-level identify-and-decode pipeline.

``decode`` runs every solver, scores each recovered plaintext with the shared
English-fitness model and returns the best one.  Because identification is a
by-product of actually decrypting, the predicted cipher is verified rather than
guessed -- this is what makes the system reliable where the original
neural-only approach was not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .fitness import EnglishFitness, get_fitness
from .solvers import SolveResult, solve_all


@dataclass
class DecodeResult:
    cipher: str
    key: object
    plaintext: str
    confidence: float                       # 0..1, separation from runner-up
    score: float                            # winning fitness (per-quadgram)
    candidates: List[SolveResult] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"cipher={self.cipher} key={self.key} "
                f"confidence={self.confidence:.2f}\n{self.plaintext}")


def _confidence(results: List[SolveResult]) -> float:
    """Map the gap between the best and best *distinct* decryption to 0..1.

    Quadgram scores are log10 per quadgram; a gap of ~0.5 (an order of
    magnitude per quadgram) is a decisive win, so we squash with that scale.
    Candidates that reproduce the winning plaintext (e.g. a Caesar shift is also
    a length-1 Vigenere key) are not real competitors and are skipped.
    """
    if not results:
        return 0.0
    top = results[0]
    runner_up = next((r.score for r in results[1:]
                      if r.plaintext != top.plaintext), None)
    if runner_up is None:
        return 1.0
    gap = top.score - runner_up
    return 1.0 / (1.0 + math.exp(-gap / 0.15))


def decode(ciphertext: str, fit: Optional[EnglishFitness] = None) -> DecodeResult:
    """Identify the cipher, recover the key and return the plaintext."""
    fit = fit or get_fitness()
    results = solve_all(ciphertext, fit)
    best = results[0]
    return DecodeResult(
        cipher=best.cipher,
        key=best.key,
        plaintext=best.plaintext,
        confidence=_confidence(results),
        score=best.score,
        candidates=results,
    )


def identify_cipher(ciphertext: str, fit: Optional[EnglishFitness] = None) -> Dict[str, object]:
    """Predict which cipher produced ``ciphertext`` (verified by decoding).

    Returns ``{"cipher", "confidence", "scores"}`` where ``scores`` maps every
    cipher name to its best achievable fitness.
    """
    fit = fit or get_fitness()
    results = solve_all(ciphertext, fit)
    return {
        "cipher": results[0].cipher,
        "confidence": _confidence(results),
        "scores": {r.cipher: r.score for r in results},
    }
