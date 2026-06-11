# Kryptos

Identify and **decode** four classical ciphers — **Caesar**, **Vigenère**, **Skip**
(modular-walk transposition) and **Columnar** transposition — from ciphertext
alone, with no key supplied.

```powershell
python solve.py "wkh klvwrub ri wkh dqflhqw flwb zdv zulwwhq eb vfkroduv"
```

```
Predicted cipher: caesar (confidence 100%)
Key:              3
Plaintext:
the history of the ancient city was written by scholars
```

## How it works

The original version tried to *guess keys with a neural network* and could only
ever decode Caesar. This version uses **classical cryptanalysis** scored by an
**English-language fitness model**, which reliably recovers the key and
plaintext for all four ciphers:

| Cipher    | Attack |
|-----------|--------|
| Caesar    | exhaustive search over 26 shifts |
| Vigenère  | key length search → per-column chi-squared frequency analysis → fitness polish |
| Skip      | exhaustive search over invertible (coprime) step sizes |
| Columnar  | column ordering via Held-Karp bigram adjacency + hill-climb on quadgram fitness |

Identification is a *by-product of decoding*: every solver runs, each recovered
plaintext is scored, and the most English-like result wins. The cipher type is
therefore **verified**, not guessed.

The fitness model is built from the project's own corpus
(`data/input/sentences.tsv`) as quadgram/bigram/unigram log-probabilities,
cached to `kryptos/data/english_stats.npz`. It builds automatically on first
use; to build it explicitly:

```powershell
python build_fitness.py --max-lines 200000
```

## Usage

```powershell
# Auto-detect and decode
python solve.py "<ciphertext>"

# Show every cipher's best candidate, ranked by fitness
python solve.py --all "<ciphertext>"

# Force a specific solver
python solve.py --cipher vigenere "<ciphertext>"

# Decrypt with a known key (no analysis)
python solve.py --cipher caesar --key 3 "wkh txlfn eurzq ira"

# Read from stdin
"<ciphertext>" | python solve.py -
```

From Python:

```python
from kryptos import decode, identify_cipher

result = decode(ciphertext)
print(result.cipher, result.key, result.confidence)
print(result.plaintext)

identify_cipher(ciphertext)   # {"cipher", "confidence", "scores": {...}}
```

Batch-decode generated CSVs:

```powershell
python models/ai_decoder.py data/output/caesar.csv --limit 200
```

## Accuracy

Measured on 100 real corpus sentences (≈120 letters each), one fresh random key
per cipher per sentence (400 ciphertexts, no keys supplied):

| Cipher    | Identify | Decode  |
|-----------|----------|---------|
| Caesar    | 100.0 %  | 100.0 % |
| Vigenère  | 100.0 %  |  99.0 % |
| Skip      |  99.0 %  | 100.0 % |
| Columnar  | 100.0 %  |  99.0 % |
| **Overall** | **99.8 %** | **99.5 %** |

(~0.6 s per ciphertext on CPU, running all four solvers.)

Accuracy degrades on very short ciphertext (cryptanalysis needs enough letters
for the statistics to separate); Vigenère in particular wants at least a few
multiples of the key length.

## Generating datasets (optional, for ML experiments)

The `encode/` scripts slice `sentences.tsv` into labelled CSVs of
`(plaintext, ciphertext, key)`:

```powershell
python encode\caesar.py --count 50000          # 50k Caesar samples
python encode\vigenere.py --start 50000        # skip the first 50k lines
```

`--count` sets how many examples to write, `--start` how many source lines to
skip, so you can build arbitrarily large non-overlapping splits.

The `models/` directory still contains the original LSTM cipher-type classifier
and key-predictor training scripts. They are kept for reference, but decoding no
longer depends on them — the cryptanalytic solvers are both more accurate and
require no training.

## Project layout

```
kryptos/
  ciphers.py     canonical encrypt/decrypt for all four ciphers
  fitness.py     English quadgram/bigram fitness model (corpus-derived, cached)
  solvers.py     per-cipher cryptanalysis
  pipeline.py    identify + decode
solve.py         command-line interface
build_fitness.py build/refresh the fitness cache
tests/           round-trip and end-to-end solve tests
encode/          dataset generators
decode/          low-level decrypt helpers (decrypt(ciphertext, key))
models/          legacy ML scripts + compatibility shims
```

## Tests

```powershell
python -m pytest tests/ -v
```
