# project-krptos

This repository implements encoding, decoding and neural key‑prediction for four classical ciphers: Caesar, Vigenère, Skip and Columnar.

## Generating datasets

Run one of the encoder scripts in `encode/` to produce a CSV of (plaintext, ciphertext, key):

```powershell
python encode\caesar.py --count 50000           # produce 50 k Caesar samples
python encode\vigenere.py --start 50000         # skip first 50 k input lines
```

Each encoder supports `--count` (how many examples to write) and `--start` (how many source lines to skip), so you can slice the `sentences.tsv` file into arbitrarily large datasets by invoking them multiple times with different offsets.

## Training key predictors

Training scripts live under `models/` and are now fully configurable via command‑line flags. Example:

```powershell
python models\key_predictor_caesar_train.py \
    --data-path data\output\caesar.csv \
    --model-path models\key_predictor_caesar.pt \
    --epochs 20 --batch-size 128 --lr 0.0005 \
    --hidden-size 256 --num-layers 2 \
    --augment 1            # add one extra random re‑encryption per plaintext
```

Common options available for all four trainers:

- `--augment N`: generate `N` additional cipher texts per input using random keys.
- `--val-split`: fraction of data held out for validation (default 0.1).
- `--patience`: early‑stopping patience on validation accuracy (default 3).
- `--embed-size`, `--hidden-size`, `--num-layers`: network capacity.
- `--max-len` (and `--max-key-len` in the Vigenère script): maximum input/key length.

The trainer prints epoch loss, training and validation accuracy, and saves the best model based on validation accuracy.

## Decoding/inference

`models/test_pipeline.py` loads the classifier and any available key predictors, then attempts to decrypt a given ciphertext. Trained models should be placed in `models/` with names like `key_predictor_caesar.pt`.

---

