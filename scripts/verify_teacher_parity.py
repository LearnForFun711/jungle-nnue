#!/usr/bin/env python3
"""Check that the GPU teacher and the engine's teacher give the same values.

The GPU teacher is a port of the engine's teacher.  A port is only worth
anything if it produces the same numbers, and the way to find out is to run
both on the same positions and compare.

With the root's noise turned off (--dirichlet-eps 0) both teachers are
deterministic, so the values should agree exactly -- up to the difference
between the engine's double arithmetic and the GPU's single.

    python3 scripts/verify_teacher_parity.py --shard data/data.t0.bin \\
                                              --net net/net.nnue --positions 300

Exit code 0 means they agree.
"""
import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE = os.path.join(ROOT, "engine")


def load_csv(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def run(cmd, what):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAIL: {what} did not run")
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--leaf-scale", type=float, default=0.125)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "teacher", "parity"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    engine_csv = os.path.join(args.out, "engine.csv")
    gpu_csv = os.path.join(args.out, "gpu.csv")

    engine_probe = os.path.join(ENGINE, "build", "teacher_probe")
    gpu_probe = os.path.join(ENGINE, "build", "gpu_teacher_probe")
    for path in (engine_probe, gpu_probe):
        if not os.path.exists(path):
            print(f"FAIL: {path} is missing -- run 'make -C engine gpu' first")
            return 1

    common = ["--shard", args.shard, "--positions", str(args.positions),
              "--sims", str(args.sims), "--seed", str(args.seed),
              "--dirichlet-eps", "0", "--out"]
    if not run([engine_probe] + common + [engine_csv] + ["--nnue", args.net,
                                                         "--threads", "2"],
               "the engine's teacher"):
        return 1
    if not run([gpu_probe] + common + [gpu_csv] + ["--net", args.net],
               "the GPU teacher"):
        return 1

    a = load_csv(engine_csv)
    b = load_csv(gpu_csv)
    if len(a) != len(b) or not a:
        print(f"FAIL: the two probes returned {len(a)} and {len(b)} rows")
        return 1

    root_diffs = [abs(x["value"] - y["value"]) for x, y in zip(a, b)]
    leaf_diffs = [abs(x["leaf_value"] - y["leaf_value"]) for x, y in zip(a, b)]
    worst_root = max(root_diffs)
    worst_leaf = max(leaf_diffs)
    exact_leaf = sum(1 for d in leaf_diffs if d < 1e-6)
    close_root = sum(1 for d in root_diffs if d < 1e-5)

    print(f"positions            : {len(a)}")
    print(f"leaf value, worst    : {worst_leaf:.3g}  (identical on {exact_leaf}/{len(a)})")
    print(f"root value, worst    : {worst_root:.3g}  (within 1e-5 on {close_root}/{len(a)})")

    # The leaf values are the same arithmetic, so they have to be identical.
    # The root values are a sum of many leaf values, in single precision on
    # the GPU and double on the engine, so they are allowed to drift a
    # little.
    if worst_leaf > 1e-6:
        print("FAIL: the leaf evaluations disagree")
        return 1
    if worst_root > 1e-3:
        print("FAIL: the teacher's root values disagree")
        return 1

    print("PASS: the GPU teacher and the engine's teacher agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
