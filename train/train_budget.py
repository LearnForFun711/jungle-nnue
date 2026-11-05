#!/usr/bin/env python3
"""Spend a wall-clock budget on a sequence of training runs.

    python3 train/train_budget.py --data "data/data.t*.bin" --out runs/budget \
        --minutes 480

Each configuration gets a slice of the budget and trains on whatever data is
on disk *at that moment*, so self-play data generated in parallel on the
machine's CPUs is picked up by the later runs.  Every candidate is exported,
because the model with the lowest validation loss is not necessarily the one
that plays best: a wider network evaluates more accurately but also more
slowly, and at a fixed clock the slower engine searches less deeply.  The
candidates are meant to be compared with engine/build/match.

The runs are independent (no warm start), so a crash or a timeout costs at
most one slice.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_configs(spec):
    """``"256:128:0.25,384:192:0.25"`` -> [(256,128,0.25), ...]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        ft = int(bits[0])
        hid = int(bits[1]) if len(bits) > 1 else ft // 2
        rw = float(bits[2]) if len(bits) > 2 else 0.25
        out.append((ft, hid, rw))
    return out


def count_positions(data_glob):
    """Count the positions in the shards the glob matches (68 bytes each)."""
    total = 0
    for path in sorted(glob.glob(data_glob, recursive=True)):
        if "policy" in os.path.basename(path):
            continue
        total += os.path.getsize(path) // 68
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.t*.bin")
    ap.add_argument("--out", default="runs/budget")
    ap.add_argument("--minutes", type=float, default=480.0,
                    help="total wall-clock budget")
    ap.add_argument("--configs", default="256:128:0.25,384:192:0.25",
                    help="comma separated ft_hidden:hidden:result_weight")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--slices", type=int, default=0,
                    help="how many slices to split the budget into "
                         "(0 = one per configuration)")
    ap.add_argument("--best-out", default="",
                    help="where to copy the best network")
    ap.add_argument("--select-by", default="corr", choices=("corr", "val_loss"),
                    help="how to pick the network for --best-out: 'corr' "
                         "(correlation with the game outcome, the default and "
                         "the only metric that is comparable between "
                         "configurations) or 'val_loss' (the old rule)")
    args = ap.parse_args()

    configs = parse_configs(args.configs)
    if not configs:
        print("no configurations given", file=sys.stderr)
        return 2
    if args.slices <= 0:
        args.slices = len(configs)
    slice_minutes = args.minutes / args.slices

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    summary = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (ValueError, OSError):
            summary = []

    t_start = time.time()
    for i, (ft, hid, rw) in enumerate(configs):
        name = f"h{ft}-{hid}-rw{str(rw).replace('.', '')}"
        outdir = os.path.join(args.out, name)
        net_path = os.path.join(outdir, "net.nnue")
        if os.path.exists(net_path):
            print(f"==> {name} already done, skipping", flush=True)
            continue

        positions = count_positions(args.data)
        print(f"\n==> [{i+1}/{len(configs)}] {name}  "
              f"budget {slice_minutes:.1f} min  data {positions:,} positions",
              flush=True)

        cmd = [sys.executable, os.path.join(HERE, "train.py"),
               "--data", args.data, "--out", outdir,
               "--budget-minutes", f"{slice_minutes}",
               "--batch", str(args.batch), "--lr", str(args.lr),
               "--ft-hidden", str(ft), "--hidden", str(hid),
               "--result-weight", str(rw)]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        elapsed = (time.time() - t0) / 60

        val_loss = None
        corr = None
        hist_path = os.path.join(outdir, "history.json")
        if os.path.exists(hist_path):
            try:
                with open(hist_path) as f:
                    hist = json.load(f)
                val_loss = hist.get("best")
                corr = hist.get("best_corr")
            except (ValueError, OSError):
                pass

        # Two checkpoints per run, because they answer different questions.
        #   best.pt      -- the best imitation of the training target.
        #   best_corr.pt -- the best predictor of who actually won.
        # For a *single* configuration the two usually agree.  Across
        # configurations they do not, and the second is the one that decides
        # how strong the engine is.
        if proc.returncode == 0 and os.path.exists(os.path.join(outdir, "best.pt")):
            subprocess.run([sys.executable, os.path.join(HERE, "export_nnue.py"),
                            "--checkpoint", os.path.join(outdir, "best.pt"),
                            "--out", net_path], cwd=ROOT, check=True)
            print(f"    exported {net_path}", flush=True)
            corr_ckpt = os.path.join(outdir, "best_corr.pt")
            if os.path.exists(corr_ckpt):
                subprocess.run([sys.executable, os.path.join(HERE, "export_nnue.py"),
                                "--checkpoint", corr_ckpt,
                                "--out", net_path.replace(".nnue", "-corr.nnue")],
                               cwd=ROOT, check=True)
                print(f"    exported {net_path.replace('.nnue', '-corr.nnue')}",
                      flush=True)
        else:
            print(f"    run failed (exit {proc.returncode})", flush=True)

        summary.append({"name": name, "ft_hidden": ft, "hidden": hid,
                        "result_weight": rw, "val_loss": val_loss, "corr": corr,
                        "minutes": round(elapsed, 2),
                        "positions": positions, "net": net_path})
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n==> budget spent after {(time.time()-t_start)/60:.1f} min")
    for s in summary:
        vl = f"{s['val_loss']:.6f}" if s["val_loss"] is not None else "n/a"
        cr = f"{s['corr']:+.4f}" if s.get("corr") is not None else "n/a"
        print(f"    {s['name']:<24} val_loss {vl}   corr {cr}   "
              f"{s['minutes']:.1f} min")

    if args.best_out:
        # The selection metric matters here.
        #
        # The validation loss is *not* comparable between configurations that
        # use different target blends.  The target is
        #
        #     (1 - w) * teacher_value + w * game_result
        #
        # and the game result is a step function at the end of the game: it
        # cannot be predicted from a single position, so every extra weight on
        # it raises the MSE floor of a *perfect* network.  Measured on the
        # training data, the target variance is 0.034 at w=0.00, 0.048 at
        # w=0.25, 0.102 at w=0.50 and 0.198 at w=0.75.  Choosing the smallest
        # validation loss therefore always picks the smallest w, i.e. the
        # candidate that imitates the teacher most closely -- and the teacher
        # is a ceiling, so that candidate is the *weakest* one.  That is not a
        # subtle effect: it is the selection rule shipping the wrong network.
        #
        # The correlation between the network's value and the game result is
        # measured on the same validation positions for every candidate and
        # does not depend on the target blend, so it can be compared across
        # configurations.  It is also what playing strength follows: the
        # handcrafted evaluation correlates +0.21 with the outcome on the
        # same data and the strong network +0.38, and the engine's Elo
        # follows the same order.
        if args.select_by == "corr":
            good = [s for s in summary if s.get("corr") is not None]
            if good:
                best = max(good, key=lambda s: s["corr"])
                import shutil
                src = best["net"].replace(".nnue", "-corr.nnue")
                if not os.path.exists(src):
                    src = best["net"]
                shutil.copyfile(src, args.best_out)
                print(f"\n==> best by correlation with the outcome: "
                      f"{best['name']} (corr {best['corr']:+.4f}, "
                      f"result_weight {best['result_weight']}) -> {args.best_out}")

                by_loss = [s for s in summary if s["val_loss"] is not None]
                if by_loss:
                    bl = min(by_loss, key=lambda s: s["val_loss"])
                    if bl["name"] != best["name"]:
                        print(f"    (note: the lowest validation loss was "
                              f"{bl['name']}, val_loss {bl['val_loss']:.6f}.  "
                              f"It is NOT the same candidate, because the "
                              f"validation loss is not comparable between "
                              f"different result weights -- see the comment "
                              f"above.)")
        else:
            good = [s for s in summary if s["val_loss"] is not None]
            if good:
                best = min(good, key=lambda s: s["val_loss"])
                import shutil
                shutil.copyfile(best["net"], args.best_out)
                print(f"\n==> best by validation loss: {best['name']} -> "
                      f"{args.best_out}")
                print(f"    (this is the old rule.  It tends to pick the "
                      f"candidate with the smallest result weight, which is "
                      f"the one that imitates the teacher most closely, and "
                      f"the teacher is a ceiling.  Prefer --select-by corr.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
