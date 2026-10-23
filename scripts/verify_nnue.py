#!/usr/bin/env python3
"""Prove that the PyTorch network and the C++ inference agree.

The training code quantises weights the same way the engine does and applies
the same rounding/clipping after every layer, so the two forward passes must
produce identical values.  This script exports a checkpoint, evaluates a set
of random positions with both implementations and reports the largest
discrepancy.

    python3 scripts/verify_nnue.py --checkpoint runs/exp/best.pt
"""
import argparse
import os
import random
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from model import JungleNNUE, make_features  # noqa: E402
from export_nnue import export  # noqa: E402

ENGINE_BUILD = os.path.join(ROOT, "engine", "build")

# The starting position and a handful of reachable ones.
START_FEN = "l5t/1c3d1/r1p1w1e/7/7/6R/E1W1P2/1D3C1/T5L r"


def random_positions(n, seed=3):
    """Random legal-ish positions built by shuffling the starting army."""
    rng = random.Random(seed)
    out = [START_FEN]
    letters = "RCDWPTLE"
    while len(out) < n:
        board = ["."] * 63
        squares = list(range(63))
        rng.shuffle(squares)
        it = iter(squares)
        for _ in range(8):                       # Red army
            board[next(it)] = rng.choice(letters)
        for _ in range(8):                       # Black army
            board[next(it)] = rng.choice(letters).lower()
        rows = []
        for r in range(9):
            row, empty = "", 0
            for c in range(7):
                p = board[r * 7 + c]
                if p == ".":
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                row += p
            if empty:
                row += str(empty)
            rows.append(row)
        out.append("/".join(rows) + " " + rng.choice("rb"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--net", default="")
    ap.add_argument("--binary", default=os.path.join(ENGINE_BUILD, "eval_nnue"))
    args = ap.parse_args()

    net_path = args.net or os.path.join(ROOT, "build", "verify.nnue")
    os.makedirs(os.path.dirname(net_path), exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = JungleNNUE(ckpt.get("ft_hidden", 512), ckpt.get("hidden", 256))
    model.load_state_dict(ckpt["model"])
    model.eval()
    info = export(model, net_path)
    print(f"exported {net_path} ({info['bytes']/1024:.1f} KiB)")

    fens = random_positions(args.positions)
    board = np.zeros((len(fens), 63), dtype=np.int64)
    stm = np.zeros(len(fens), dtype=np.int64)
    for i, fen in enumerate(fens):
        parts = fen.split()
        rows, side = parts[0], parts[1]
        r = c = 0
        for ch in rows:
            if ch == "/":
                r += 1
                c = 0
                continue
            if ch.isdigit():
                c += int(ch)
                continue
            t = "RCDWPTLE".index(ch.upper()) + 1
            col = 0 if ch.isupper() else 1
            board[i][r * 7 + c] = t if col == 0 else t + 8
            c += 1
        stm[i] = 0 if side == "r" else 1

    b = torch.from_numpy(board)
    s = torch.from_numpy(stm)
    with torch.no_grad():
        fr, fb = make_features(b, s)
        v = model(fr, fb, s)
    torch_vals = v.numpy()

    proc = subprocess.run([args.binary, net_path], input="\n".join(fens) + "\n",
                          capture_output=True, text=True, check=True)
    cpp_vals = np.array([float(x) for x in proc.stdout.split()])

    # Both sides report the value from the point of view of the side to move.
    torch_stm = torch_vals
    cpp_stm = cpp_vals

    diff = np.abs(torch_stm - cpp_stm)
    print(f"positions: {len(fens)}")
    print(f"max |torch - cpp| = {diff.max():.6f}")
    print(f"mean |torch - cpp| = {diff.mean():.6f}")
    if diff.max() > 1e-3:
        worst = int(np.argmax(diff))
        print(f"WORST position {worst}: fen={fens[worst]}")
        print(f"  torch {torch_stm[worst]:.6f}  cpp {cpp_stm[worst]:.6f}")
        print("FAIL: the two implementations disagree")
        return 1
    print("PASS: PyTorch and C++ inference agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
