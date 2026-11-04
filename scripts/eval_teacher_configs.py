#!/usr/bin/env python3
"""Compare MCTS teacher configurations on the same positions.

    python3 scripts/eval_teacher_configs.py --shard "data/data.t0.bin" \
        --positions 3000 --net net/net.nnue

The teacher produces the values the network is trained on, so the teacher's
quality is the ceiling on the network's quality.  This script runs the
teacher on a fixed sample of positions that already came out of self-play
games -- positions whose final result is known -- and reports, for every
configuration:

  corr(value, result)   how well the teacher's value predicts who won
  corr(value, stored)   how well it reproduces the value the teacher gave
                         when the game was played (a sanity check: the
                         configuration the data was generated with should
                         score near 1.0 here)
  std / |v|>0.5         the shape of the value distribution, so that a
                         configuration which has collapsed to "everything is a
                         draw" cannot masquerade as a good one

The sample is the same for every configuration, so the comparison is fair.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBE = os.path.join(ROOT, "engine", "buildprobe", "teacher_probe")


def run_probe(shards, positions, sims, leaf_depth, leaf_agg, leaf_scale, net, threads, seed,
              hybrid=0.0):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out = tmp.name
    cmd = [PROBE]
    for s in shards:
        cmd += ["--shard", s]
    cmd += ["--positions", str(positions), "--sims", str(sims),
            "--leaf-depth", str(leaf_depth), "--leaf-agg", str(leaf_agg),
            "--leaf-scale", str(leaf_scale), "--threads", str(threads),
            "--seed", str(seed), "--hybrid", str(hybrid), "--out", out]
    if net:
        cmd += ["--nnue", net]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"probe failed:\n{res.stderr}")
    value, leaf, result, ply, stored = [], [], [], [], []
    with open(out, newline="") as f:
        for row in csv.DictReader(f):
            value.append(float(row["value"]))
            leaf.append(float(row["leaf_value"]))
            result.append(float(row["result"]))
            ply.append(int(row["ply"]))
            stored.append(float(row["stored"]))
    os.unlink(out)
    return (np.array(value), np.array(leaf), np.array(result), np.array(ply), np.array(stored))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", action="append", required=True,
                    help="a shard file; may be given several times")
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--net", default="", help="a network to use as the teacher's leaf")
    ap.add_argument("--out", default="", help="write a JSON summary here")
    ap.add_argument("--custom", action="append", default=[],
                    help="a configuration to measure instead of the built-in "
                         "list: label:leaf_depth:leaf_agg:leaf_scale:hybrid "
                         "The sixth field is 'net' or 'hc' and says which "
                         "leaf evaluation to use; without it the network is "
                         "used whenever --net was given.")
    args = ap.parse_args()

    if not os.path.exists(PROBE):
        raise SystemExit(f"missing {PROBE}; build it with: make -C engine BUILD=buildprobe teacher_probe")

    # (label, leaf_depth, leaf_agg, net, leaf_scale, hybrid)
    configs = [
        ("static eval (leaf depth 0)", 0, 1, "", 1.0, 0.0),
        ("handcrafted, mean over replies", 1, 1, "", 1.0, 0.0),
        ("handcrafted, best reply (minimax)", 1, 0, "", 1.0, 0.0),
        ("handcrafted, depth 2 alpha-beta", 2, 1, "", 1.0, 0.0),
    ]
    if args.net:
        # The neural leaf is clamped to [-1,1], and the network's own output
        # uses only a small part of that range, so the scale decides how much
        # of the distribution gets clipped -- and, for the "best reply"
        # aggregation, whether the maximum over the opponent's replies
        # saturates and stops carrying any information at all.
        for scale in (0.25, 0.5, 0.75, 1.0, 2.0):
            configs.append((f"network x{scale:g}, mean over replies", 1, 1, args.net, scale, 0.0))
        # The handcrafted evaluation knows the tactical rules of the game --
        # the traps, the pieces that are attacked and not defended -- and a
        # network trained on positions does not necessarily learn a rule
        # exactly.  Mixing it back in may keep both.
        for hybrid in (0.25, 0.5, 0.75):
            configs.append((f"network x1 + {hybrid:g} handcrafted, mean", 1, 1, args.net, 1.0, hybrid))

    if args.custom:
        configs = []
        for spec in args.custom:
            bits = spec.split(":")
            label = bits[0]
            depth = int(bits[1]) if len(bits) > 1 else 1
            agg = int(bits[2]) if len(bits) > 2 else 1
            scale = float(bits[3]) if len(bits) > 3 else 1.0
            hybrid = float(bits[4]) if len(bits) > 4 else 0.0
            which = bits[5] if len(bits) > 5 else "net"
            if which == "hc" or not args.net or depth <= 0:
                net = ""
            else:
                net = args.net
            configs.append((label, depth, agg, net, scale, hybrid))

    summary = []
    print(f"probing {args.positions} positions, {args.sims} sims, seed {args.seed}")
    print()
    header = (f"{'configuration':<34} {'corr(tree,result)':>17} {'corr(leaf,result)':>17} "
              f"{'corr(tree,stored)':>18} {'std':>7} {'|v|>0.5':>7} {'mean':>7}")
    print(header)
    print("-" * len(header))
    for label, depth, agg, net, scale, hybrid in configs:
        v, leaf, r, ply, stored = run_probe(args.shard, args.positions, args.sims,
                                            depth, agg, scale, net, args.threads,
                                            args.seed, hybrid)
        c_out = float(np.corrcoef(v, r)[0, 1])
        c_leaf = float(np.corrcoef(leaf, r)[0, 1])
        c_stored = float(np.corrcoef(v, stored)[0, 1])
        sharp = float((np.abs(v) > 0.5).mean())
        print(f"{label:<34} {c_out:>+19.4f} {c_leaf:>+19.4f} {c_stored:>+19.4f} "
              f"{v.std():>7.4f} {sharp*100:>6.1f}% {v.mean():>+7.4f}")
        summary.append({
            "config": label, "leaf_depth": depth, "leaf_agg": agg,
            "net": bool(net), "leaf_scale": scale, "hybrid": hybrid,
            "corr_result": c_out, "corr_leaf": c_leaf, "corr_stored": c_stored,
            "std": float(v.std()), "sharp": sharp,
        })

    print()
    best = max(summary, key=lambda s: s["corr_result"])
    print(f"best configuration: {best['config']}  (corr {best['corr_result']:+.4f})")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "best": best}, f, indent=2)
        print("written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
