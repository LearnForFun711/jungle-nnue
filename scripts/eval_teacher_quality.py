#!/usr/bin/env python3
"""Measure how well the MCTS teacher's values predict the game outcome.

    python3 scripts/eval_teacher_quality.py --data "data/round1/data.*.bin"

Every position in a shard carries the value the teacher gave it at the time,
and the result of the game the position came from.  The correlation between
the two is the quality of the teacher, and it is the number that decides
whether a round of expert iteration produced better data than the round
before it.

The comparison is only fair if the two teachers are measured on data with the
same shape, so the script also reports the value distribution: a teacher that
has collapsed to "everything is a draw" has a correlation of zero, and one
that has collapsed to "everything is +-1" has a correlation that is high but
useless.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from dataset import load_shards  # noqa: E402


def report(label, mcts_value, result, ply=None):
    v = mcts_value.astype(float)
    r = result.astype(float)
    c_outcome = float(np.corrcoef(v, r)[0, 1])
    decisive = np.abs(r) > 0.5
    c_decisive = (float(np.corrcoef(v[decisive], r[decisive])[0, 1])
                  if decisive.sum() > 10 else float("nan"))
    sharp = float((np.abs(v) > 0.5).mean())
    print(f"{label}")
    print(f"    positions                {len(v):,}")
    print(f"    corr(teacher, outcome)   {c_outcome:+.4f}")
    print(f"    corr on decisive games   {c_decisive:+.4f}")
    print(f"    teacher value  mean {v.mean():+.4f}  std {v.std():.4f}"
          f"  |v|>0.5 in {sharp*100:.1f}%")
    print(f"    outcome       mean {r.mean():+.4f}  std {r.std():.4f}"
          f"  draws in {(np.abs(r) < 0.5).mean()*100:.1f}%")
    return c_outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="a shard file or a glob")
    ap.add_argument("--positions", type=int, default=0,
                    help="limit the number of positions (0 = all)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    pattern = args.data
    if not any(ch in pattern for ch in "*?["):
        pattern = pattern  # a plain file name is fine
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no shards matched {args.data!r}", file=sys.stderr)
        return 2

    board, stm, result, mcts_value, ply, cause = load_shards(pattern)
    n = len(board)
    if args.positions:
        n = min(n, args.positions)
    result, mcts_value = result[:n], mcts_value[:n]

    label = args.label or os.path.basename(os.path.dirname(files[0]) or "data")
    print()
    report(f"teacher: {label}", mcts_value, result)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
