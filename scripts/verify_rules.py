#!/usr/bin/env python3
"""Cross-check the C++ engine's move generator against the Python reference.

The reference implementation lives in the original project at
``backend/engine.py``.  We play random games with it, snapshot positions along
the way, and compare the legal move sets produced by both implementations.

Usage:
    python3 scripts/verify_rules.py [--games 40] [--samples 400]

Exit code 0 means the two implementations agree on every sampled position.
"""
import argparse
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(HERE, "..", "engine")
REFERENCE_DIR = os.path.join(HERE, "..", "..", "backend")

sys.path.insert(0, os.path.abspath(REFERENCE_DIR))


def reference_moves(engine):
    """Legal moves of the reference engine, as a set of ((fr,fc),(tr,tc))."""
    out = set()
    for m in engine.get_valid_moves():
        out.add((tuple(m["from"]), tuple(m["to"])))
    return out


def cpp_moves(binary, fens):
    """Legal moves of the C++ engine for a list of FENs."""
    proc = subprocess.run(
        [binary],
        input="\n".join(fens) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    results = []
    for line in proc.stdout.splitlines():
        if "|" not in line:
            results.append(set())
            continue
        _, _, moves = line.partition("|")
        s = set()
        for mv in moves.split():
            if len(mv) != 4:
                continue
            f = (9 - int(mv[1]), ord(mv[0]) - ord("a"))
            t = (9 - int(mv[3]), ord(mv[2]) - ord("a"))
            s.add((f, t))
        results.append(s)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--binary", default=os.path.join(ENGINE_DIR, "build", "dump_moves"))
    ap.add_argument("--write-golden", default="",
                    help="write the sampled positions and their reference legal "
                         "moves to this file instead of comparing")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        print(f"missing {args.binary} -- run `make` in engine/ first", file=sys.stderr)
        return 2

    from engine import JungleEngine  # the original project's reference engine

    random.seed(args.seed)
    fens, py_sets = [], []

    for _ in range(args.games):
        eng = JungleEngine()
        for _ply in range(120):
            if eng.winner:
                break
            moves = eng.get_valid_moves()
            if not moves:
                break
            # Collect the position *and* its legal moves for comparison.
            fen = eng.to_fen().split("|")[0].strip()
            fens.append(fen)
            py_sets.append(reference_moves(eng))
            mv = random.choice(moves)
            eng.apply_move(mv)
            if len(fens) >= args.samples:
                break
        if len(fens) >= args.samples:
            break

    print(f"sampled {len(fens)} positions from {args.games} random games")

    if args.write_golden:
        # The golden file is produced from the *reference* implementation, so it
        # encodes the reference behaviour, not the C++ behaviour.
        with open(args.write_golden, "w") as f:
            f.write("# Jungle Chess rule golden file.\n")
            f.write("# Each line: <FEN> | <move> <move> ...\n")
            f.write("# Generated from the reference Python implementation by\n")
            f.write("# scripts/verify_rules.py --write-golden\n")
            for fen, moves in zip(fens, py_sets):
                names = sorted(f"{chr(ord('a') + fc)}{9 - fr}{chr(ord('a') + tc)}{9 - tr}"
                               for (fr, fc), (tr, tc) in moves)
                f.write(fen + " | " + " ".join(names) + "\n")
        print(f"wrote {args.write_golden} with {len(fens)} positions")
        return 0

    cpp_sets = cpp_moves(args.binary, fens)

    bad = 0
    for i, (fen, pys, cps) in enumerate(zip(fens, py_sets, cpp_sets)):
        if pys != cps:
            bad += 1
            if bad <= 10:
                print(f"\nMISMATCH #{i}  fen={fen}")
                print(f"  only in python: {sorted(pys - cps)}")
                print(f"  only in c++   : {sorted(cps - pys)}")
    if bad:
        print(f"\nFAIL: {bad}/{len(fens)} positions disagree")
        return 1
    print("PASS: C++ and Python rule implementations agree on every sampled position")
    return 0


if __name__ == "__main__":
    sys.exit(main())
