#!/usr/bin/env python3
"""Check the C++ engine against the recorded golden rule data.

The golden file was generated from an independent reference implementation of
the rules (see scripts/verify_rules.py --write-golden), so this test catches
rule regressions without needing that project present.

    python3 scripts/verify_golden.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(ROOT, "engine", "tests", "golden_rules.txt")
BINARY = os.path.join(ROOT, "engine", "build", "dump_moves")


def main():
    if not os.path.exists(BINARY):
        print(f"missing {BINARY} -- run `make -C engine` first", file=sys.stderr)
        return 2
    if not os.path.exists(GOLDEN):
        print(f"missing {GOLDEN}", file=sys.stderr)
        return 2

    fens, expected = [], []
    with open(GOLDEN) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fen, _, moves = line.partition("|")
            fens.append(fen.strip())
            expected.append(set(moves.split()))

    proc = subprocess.run([BINARY], input="\n".join(fens) + "\n",
                          capture_output=True, text=True, check=True)

    got = []
    for line in proc.stdout.splitlines():
        if "|" not in line:
            got.append(set())
            continue
        _, _, moves = line.partition("|")
        got.append(set(moves.split()))

    bad = 0
    for i, (fen, exp) in enumerate(zip(fens, expected)):
        if got[i] != exp:
            bad += 1
            if bad <= 5:
                print(f"MISMATCH #{i}  fen={fen}")
                print(f"  missing: {sorted(exp - got[i])}")
                print(f"  extra  : {sorted(got[i] - exp)}")
    print(f"checked {len(fens)} positions")
    if bad:
        print(f"FAIL: {bad} positions disagree")
        return 1
    print("PASS: C++ engine matches the golden rule data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
