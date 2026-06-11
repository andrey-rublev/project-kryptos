#!/usr/bin/env python
"""Build the English-fitness statistics used by the solvers.

Estimates quadgram/bigram/unigram probabilities from the corpus and caches them
to ``kryptos/data/english_stats.npz``.  Run once after cloning (the solvers also
build this automatically on first use):

    python build_fitness.py --max-lines 200000
"""

import argparse
import time

from kryptos.fitness import CACHE_PATH, DEFAULT_CORPUS, build_stats, save_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="path to sentences.tsv")
    p.add_argument("--out", default=CACHE_PATH, help="output .npz cache path")
    p.add_argument("--max-lines", type=int, default=200_000,
                   help="corpus lines to sample (0 = all)")
    args = p.parse_args()

    t = time.time()
    print(f"Reading corpus: {args.corpus}")
    stats = build_stats(args.corpus, max_lines=(args.max_lines or None))
    save_stats(stats, args.out)
    seen = (stats["quad"] > stats["quad_floor"]).sum()
    print(f"Saved {args.out} in {time.time() - t:.1f}s "
          f"({seen:,} distinct quadgrams observed)")


if __name__ == "__main__":
    main()
