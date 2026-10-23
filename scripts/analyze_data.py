#!/usr/bin/env python3
"""Measure how well the teacher's value predicts the actual game result.

This is the direct quality metric for the MCTS teacher: a good teacher assigns
values that correlate with how the game actually ends.  We report

  * Pearson correlation between the MCTS value and the game outcome
    (both from the side-to-move's point of view), and
  * the calibration curve: for each value bucket, the observed win rate.

A teacher with a higher correlation produces better training targets.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

from dataset import load_shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.t*.bin")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    board, stm, result, value, ply, cause = load_shards(args.data)
    n = board.shape[0]
    if args.max:
        n = min(n, args.max)
        board, stm, result, value, ply = board[:n], stm[:n], result[:n], value[:n], ply[:n]
        cause = cause[:n]

    decisive = np.abs(result) > 0.5
    print(f"positions: {n}   decisive: {decisive.sum()} ({100*decisive.mean():.1f}%)")

    v = value[decisive]
    r = result[decisive]
    if v.size > 2:
        corr = float(np.corrcoef(v, r)[0, 1])
        print(f"Pearson corr(MCTS value, game outcome) = {corr:.4f}")

    # Also correlate with the *sign* only (wins vs losses).
    wins = v[r > 0.5]
    losses = v[r < -0.5]
    if wins.size and losses.size:
        print(f"  mean value in won positions : {wins.mean():+.4f} (n={wins.size})")
        print(f"  mean value in lost positions : {losses.mean():+.4f} (n={losses.size})")
        print(f"  separation                   : {wins.mean()-losses.mean():+.4f}")

    # Calibration curve.
    print("\ncalibration (value bucket -> observed win rate):")
    edges = [-1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (value >= lo) & (value < hi) & decisive
        if m.sum() < 20:
            continue
        obs = float((result[m] > 0.5).mean())
        print(f"  [{lo:+.1f},{hi:+.1f})  n={int(m.sum()):8d}  observed win rate {obs:+.3f}"
              f"  (ideal {0.5*(lo+hi):+.3f})")

    # Value distribution -- a teacher whose values are all saturated near +-1
    # would give the network almost no signal.
    # Per ending cause.
    names = {0: "den entry", 1: "no legal moves", 2: "repetition", 3: "length limit"}
    print("\nby ending cause:")
    for c, name in names.items():
        m = cause == c
        if m.sum() == 0:
            continue
        dec = m & decisive
        if dec.sum() > 2:
            cc = float(np.corrcoef(value[dec], result[dec])[0, 1])
        else:
            cc = float("nan")
        print(f"  {name:<16} positions {int(m.sum()):7d} ({100*m.mean():4.1f}%)  "
              f"corr(value,result) = {cc:+.4f}")

    # Consistency check: in the final position of a game the side to move has
    # usually just won, so its value and the result must agree in sign.
    starts = np.flatnonzero(ply == 0)
    ends = np.append(starts[1:] - 1, len(ply) - 1)
    agree = 0
    total = 0
    for s0, e0 in zip(starts, ends):
        if e0 < s0:
            continue
        if abs(result[e0]) < 0.5:
            continue
        total += 1
        if np.sign(value[e0]) == np.sign(result[e0]):
            agree += 1
    if total:
        print(f"\nfinal position: value sign agrees with the result in "
              f"{agree}/{total} decisive games ({100*agree/total:.1f}%)")

    print(f"\nvalue stats: mean {value.mean():+.4f} std {value.std():.4f} "
          f"min {value.min():+.3f} max {value.max():+.3f}")
    print(f"  fraction |v| > 0.8 : {float((np.abs(value) > 0.8).mean()):.3f}")
    print(f"  fraction |v| < 0.2 : {float((np.abs(value) < 0.2).mean()):.3f}")
    print(f"  mean ply: {ply.mean():.1f}   max ply: {ply.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
