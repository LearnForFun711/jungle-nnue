#!/usr/bin/env python3
"""Diagnose *why* a network plays the way it does.

    python3 scripts/diagnose_net.py --net net/net-new.nnue --compare net/net.nnue

The question this answers is not "is the net correlated with the outcome"
(eval_net_quality.py does that) but "what does the net actually feed to the
search".  Two things matter for the engine:

  1. The *magnitude* of the net's output.  The engine multiplies the network
     value by ``value_scale`` (10.0) and treats it as a centipawn-like
     evaluation.  A net that outputs +-0.02 everywhere looks like a flat
     evaluation to the search: the engine can still see mate, but it cannot
     tell a good position from a slightly less good one, so it plays
     shallow, tactically blind lines.  A net that is *saturated* (everything
     at +-1.0) has the opposite problem -- it sees nothing but wins and
     losses.

  2. The *calibration*: is the value that the net reports the win probability
     it should be?

Both nets are compared on exactly the same positions, and the handcrafted
evaluation is included as the reference scale.

Run it on the data the network was actually trained on.  The "did the network
learn anything its teacher did not know?" section compares the network with
the teacher value stored in the shard, so it only means something when the
shard was produced by that network's own teacher.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from model import JungleNNUE, make_features  # noqa: E402
from dataset import load_shards  # noqa: E402


def fen_of(board, stm):
    letters = {1: "R", 2: "C", 3: "D", 4: "W", 5: "P", 6: "T", 7: "L", 8: "E"}
    rows = []
    for r in range(9):
        row, empty = [], 0
        for c in range(7):
            p = board[r * 7 + c]
            if p == 0:
                empty += 1
                continue
            if empty:
                row.append(str(empty))
                empty = 0
            ch = letters[(p - 1) % 8 + 1]
            if p > 8:
                ch = ch.lower()
            row.append(ch)
        if empty:
            row.append(str(empty))
        rows.append("".join(row))
    return "/".join(rows) + (" r" if stm == 0 else " b")


def net_values(net, board, stm):
    model = JungleNNUE(256, 128)
    from export_nnue import load_nnue
    model = load_nnue(net)
    model.eval()
    b = torch.from_numpy(board).long()
    s = torch.from_numpy(stm)
    with torch.no_grad():
        fr, fb = make_features(b, s)
        return model(fr, fb, s).numpy()


def describe(name, v, result):
    sat = float(np.mean(np.abs(v) > 0.95))
    dead = float(np.mean(np.abs(v) < 0.05))
    c_out = float(np.corrcoef(v, result)[0, 1])
    print(f"  {name}")
    print(f"    value      mean {v.mean():+.4f}  std {v.std():.4f}  "
          f"min {v.min():+.3f}  max {v.max():+.3f}")
    print(f"    saturation |v|>0.95  {sat*100:5.1f}%      "
          f"flat |v|<0.05  {dead*100:5.1f}%")
    print(f"    corr with outcome     {c_out:+.4f}")
    # engine scale: lround(v * 1000 * value_scale), value_scale = 10
    print(f"    engine eval units     mean {v.mean()*1000*10:+8.1f}  "
          f"std {v.std()*1000*10:8.1f}")
    return c_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--data", default="data/data.t6.bin")
    ap.add_argument("--positions", type=int, default=8000)
    args = ap.parse_args()

    board, stm, result, mcts_value, ply, cause = load_shards(args.data)
    n = min(args.positions, len(board))
    board, stm = board[:n], stm[:n]
    result = result[:n].astype(float)
    mcts_value = mcts_value[:n].astype(float)
    ply = ply[:n]

    print(f"data: {args.data}  ({n} positions)")
    print()

    nets = [("NEW  " + args.net, args.net)]
    if args.compare:
        nets.append(("OLD  " + args.compare, args.compare))

    vals = {}
    for label, path in nets:
        v = net_values(path, board, stm)
        vals[label] = v
        describe(label, v, result)
        print()

    # handcrafted reference, on the same positions
    binary = os.path.join(ROOT, "engine", "build", "eval_nnue")
    hc = None
    if os.path.exists(binary):
        fens = "\n".join(fen_of(board[i], stm[i]) for i in range(n)) + "\n"
        proc = subprocess.run([binary, "--handcrafted"], input=fens,
                              capture_output=True, text=True, check=True)
        hc = np.array([float(x) for x in proc.stdout.split()])
        print("  HANDCRAFTED (reference scale)")
        print(f"    value      mean {hc.mean():+.1f}  std {hc.std():.1f}")
        print(f"    corr with outcome     {np.corrcoef(hc, result)[0,1]:+.4f}")
        print()

    # calibration of each net: predicted value vs actual outcome
    edges = [-1.01, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.01]
    for label, values in vals.items():
        print(f"  calibration of {label.strip()}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            if m.sum() == 0:
                continue
            err = result[m].mean() - values[m].mean()
            print(f"    v in [{lo:+.2f},{hi:+.2f})  n={int(m.sum()):6d}  "
                  f"actual outcome {result[m].mean():+.3f}  "
                  f"expected {values[m].mean():+.3f}  "
                  f"error {err:+.3f}")
        print()

    # Accuracy of the sign of the value.  `result` is -1 / 0 / +1 from the
    # side-to-move's point of view, so a decisive game has result != 0 and a
    # draw has result == 0.
    def sign_accuracy(values, mask):
        dec = mask & (result != 0)
        if dec.sum() < 20:
            return None, 0
        pred = values[dec] > 0
        actual = result[dec] > 0
        return float((pred == actual).mean()), int(dec.sum())

    print("  sign accuracy on decisive games (predicted win vs actual win)")
    phases = [("opening (ply<20)", (ply < 20)),
              ("middle (20-60)", (ply >= 20) & (ply < 60)),
              ("end (ply>=60)", (ply >= 60))]
    for name, pmask in phases:
        row = [f"    {name:18s}"]
        for label, values in vals.items():
            acc, cnt = sign_accuracy(values, pmask)
            row.append(f"{label.split()[0]} {acc*100:5.1f}% (n={cnt})")
        if hc is not None:
            acc, cnt = sign_accuracy(hc, pmask)
            row.append(f"HC {acc*100:5.1f}% (n={cnt})")
        print("  ".join(row))
    print()

    # The question that predicts playing strength: did the network learn
    # anything the teacher did not know?
    #
    # A network distilled from the teacher is a copy of it, and a copy is
    # never better than the original.  So compare the network's correlation
    # with the game outcome against the *stored teacher value on the very
    # same positions*: if the two numbers are the same, the network is a
    # mimic and it will not be stronger than its teacher.  If the network is
    # well above the teacher, it has learned the game's result on its own and
    # that is where its strength comes from.
    print("  did the network learn anything its teacher did not know?")
    c_teacher = float(np.corrcoef(mcts_value, result)[0, 1])
    print(f"    stored teacher value    corr(outcome) {c_teacher:+.4f}")
    for label, values in vals.items():
        c_net = float(np.corrcoef(values, result)[0, 1])
        delta = c_net - c_teacher
        verdict = ("a MIMIC -- it will not be stronger than its teacher"
                   if delta < 0.01 else
                   "it learned beyond its teacher -- this is where the strength "
                   "comes from")
        print(f"    {label.strip():<28} corr(outcome) {c_net:+.4f}   "
              f"({delta:+.4f} vs the teacher)  {verdict}")
    print()

    # How much of the decision-relevant band is noise?  Count positions where
    # the net and the handcrafted evaluation disagree on the sign, in the
    # band where the net is not already confident.
    if hc is not None:
        band = np.abs(v) < 0.3
        disagree = (np.sign(v[band]) != np.sign(hc[band]))
        print(f"  net vs handcrafted, in the band |v|<0.3 (n={int(band.sum())}):")
        print(f"    they disagree on the sign in {disagree.mean()*100:.1f}% of them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
