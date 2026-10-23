#!/usr/bin/env python3
"""Measure how well a network predicts the outcome of the game.

    python3 scripts/eval_net_quality.py --net net.nnue --data "data/data.t6.bin"

The number that matters is the correlation between the network's value and
the *result of the game* the position came from, from the side-to-move's
point of view.  That is what a value function is for, and it is the number
the handcrafted evaluation has to be compared against.

The same script reports the handcrafted evaluation's correlation on the same
positions, so the two are always measured on the same data.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--checkpoint", default="",
                    help="a checkpoint to load instead of the binary "
                         "(the binary is the thing that matters, so this is "
                         "only for cross-checking)")
    ap.add_argument("--data", default="data/data.t6.bin")
    ap.add_argument("--positions", type=int, default=30000)
    ap.add_argument("--skip-handcrafted", action="store_true")
    args = ap.parse_args()

    board, stm, result, mcts_value, ply, cause = load_shards(args.data)
    n = min(args.positions, len(board))
    board, stm = board[:n], stm[:n]
    result, mcts_value = result[:n].astype(float), mcts_value[:n].astype(float)

    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu")
        model = JungleNNUE(ck.get("ft_hidden", 256), ck.get("hidden", 128))
        model.load_state_dict(ck["model"])
        model.eval()
    else:
        # Read the binary the engine actually plays.
        model = JungleNNUE(256, 128)
        from export_nnue import load_nnue
        model = load_nnue(args.net)

    b = torch.from_numpy(board).long()
    s = torch.from_numpy(stm)
    with torch.no_grad():
        fr, fb = make_features(b, s)
        net_values = model(fr, fb, s).numpy()

    print(f"network: {args.net}")
    print(f"data:    {args.data}  ({n} positions)")
    print()
    print(f"  correlation with the game outcome      {np.corrcoef(net_values, result)[0,1]:+.4f}")
    print(f"  correlation with the MCTS teacher      {np.corrcoef(net_values, mcts_value)[0,1]:+.4f}")
    print(f"  network value: mean {net_values.mean():+.4f}  std {net_values.std():.4f}")

    if not args.skip_handcrafted:
        # The handcrafted evaluation on exactly the same positions.
        binary = os.path.join(ROOT, "engine", "build", "eval_nnue")
        fens = "\n".join(fen_of(board[i], stm[i]) for i in range(n)) + "\n"
        proc = subprocess.run([binary, "--handcrafted"], input=fens,
                              capture_output=True, text=True, check=True)
        hc = np.array([float(x) for x in proc.stdout.split()])
        print()
        print(f"  handcrafted evaluation, same positions:")
        print(f"    correlation with the game outcome    {np.corrcoef(hc, result)[0,1]:+.4f}")
        print(f"    correlation with the MCTS teacher    {np.corrcoef(hc, mcts_value)[0,1]:+.4f}")
        print(f"    value: mean {hc.mean():+.1f}  std {hc.std():.1f}")
        print()
        print(f"  ==> the network is {'BETTER' if np.corrcoef(net_values, result)[0,1] > np.corrcoef(hc, result)[0,1] else 'WORSE'}"
              f" than the handcrafted evaluation at predicting the outcome")
    return 0


if __name__ == "__main__":
    sys.exit(main())
