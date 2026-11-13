#!/usr/bin/env python3
"""Tune the NNUE evaluation scale against the handcrafted engine.

    python3 scripts/tune_scale.py --net net/net.nnue --scales 1,3,10,30 \
                                  --games 60 --time 0.3

The network's value is in [-1, 1] and the engine multiplies it by
``value_scale * 1000`` before handing it to the search, which thinks in
centipawn-like units.  The handcrafted evaluation, which the search also
sees, has a standard deviation of about 450 in the same units.  So the scale
decides how much the search listens to the network:

    scale 1    the network's evaluations are the same size as the
               handcrafted ones -- the search sees both, and a small deficit
               looks worse than a repetition (which is worth zero).
    scale 10   the network's evaluations dominate -- the search plays the
               network's game, and the handcrafted evaluation is a rounding
               error.

Which one is better is an empirical question, and the answer changed when
the match harness was fixed (see the note in engine/src/nnue.hpp), so it is
measured rather than assumed.

Each point is a real match against the handcrafted engine, so it takes as
long as the matches do.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def play(net, scale, games, seconds, threads):
    spec = f"ab:nnue={net}:scale={scale}"
    proc = subprocess.run(
        [os.path.join(ROOT, "engine", "build", "match"),
         "--games", str(games), "--threads", str(threads),
         "--time", str(seconds),
         "--engine-a", spec, "--engine-b", "ab"],
        capture_output=True, text=True, cwd=ROOT)
    try:
        return json.loads(proc.stdout)
    except Exception:
        sys.stderr.write(proc.stdout[-800:] + "\n" + proc.stderr[-800:] + "\n")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="net/net.nnue")
    ap.add_argument("--scales", default="1,3,10,30")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--time", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(ROOT, args.net)):
        print(f"no such network: {args.net}", file=sys.stderr)
        return 2

    results = []
    print(f"net {args.net}   {args.games} games at {args.time}s/move\n")
    print(f"  {'scale':>6}  {'W':>4} {'L':>4} {'D':>4}   {'score':>6}   "
          f"{'95% CI':>15}   {'Elo':>7}")
    for scale in [float(s) for s in args.scales.split(",") if s.strip()]:
        d = play(args.net, scale, args.games, args.time, args.threads)
        if d is None:
            print(f"  {scale:6g}  (the match failed)")
            continue
        ci = d.get("score_ci95", [float('nan'), float('nan')])
        print(f"  {scale:6g}  {d['a_wins']:4d} {d['b_wins']:4d} {d['draws']:4d}   "
              f"{d['score_a']:6.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]   "
              f"{d['elo_a']:+7.1f}")
        results.append({"scale": scale, "a_wins": d["a_wins"],
                        "b_wins": d["b_wins"], "draws": d["draws"],
                        "score_a": d["score_a"], "elo_a": d["elo_a"],
                        "score_ci95": ci})

    if results:
        best = max(results, key=lambda r: r["score_a"])
        print(f"\n==> best scale {best['scale']:g} "
              f"(score {best['score_a']:.3f}, Elo {best['elo_a']:+.1f})")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"net": args.net, "games": args.games,
                           "time": args.time, "results": results}, f, indent=2)
            print(f"    written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
