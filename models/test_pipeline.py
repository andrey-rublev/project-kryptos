"""Backwards-compatible decode entry point.

The original implementation relied on neural key-predictors that could not
actually recover Vigenere/Skip/Columnar keys.  It now delegates to the
``kryptos`` cryptanalysis package, which reliably identifies and decrypts all
four ciphers.  The ``decode_with_models`` function and CLI are preserved so
existing scripts keep working.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kryptos.pipeline import decode  # noqa: E402


def decode_with_models(ciphertext, conf_threshold=0.6, brute_force=False):
    """Identify the cipher and decrypt ``ciphertext``.

    Returns a dict with ``cipher``, ``key``, ``plaintext`` and
    ``classifier_confidence`` (kept for compatibility with the old API).
    ``brute_force`` is accepted but ignored -- the solvers are already
    exhaustive/optimal where it matters.
    """
    result = decode(ciphertext)
    return {
        "cipher": result.cipher,
        "key": result.key,
        "plaintext": result.plaintext,
        "classifier_confidence": float(result.confidence),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decode a ciphertext")
    parser.add_argument("ciphertext", nargs="?",
                        default="wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj",
                        help="ciphertext string to decode")
    parser.add_argument("--bruteforce", action="store_true",
                        help="accepted for compatibility; solvers are already exhaustive")
    args = parser.parse_args()

    result = decode_with_models(args.ciphertext, brute_force=args.bruteforce)
    print("Ciphertext:", args.ciphertext)
    print("Predicted cipher:", result["cipher"],
          f"(confidence={result['classifier_confidence']:.3f})")
    print("Predicted key:", result["key"])
    print("Decoded plaintext:", result["plaintext"])
