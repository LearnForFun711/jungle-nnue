#!/usr/bin/env python3
"""Turn an exported network back into a training checkpoint.

    python3 scripts/net_to_checkpoint.py --net net/net.nnue --out runs/champion.pt

The engine plays an exported binary network, but training needs a checkpoint
that can be loaded, mutated and trained further.  This is the way back: the
exported weights are read into the model, and the model is saved as a
checkpoint.

It is also how the reinforcement loop gets its starting point: the champion
of the population has to be a checkpoint, so that a mutant can be made by
adding a little noise to it.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

import torch  # noqa: E402

from export_nnue import load_nnue  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", action="store_true",
                    help="make a fresh, untrained network instead of reading one")
    ap.add_argument("--ft-hidden", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="perturb the weights by this much, relative to their "
                         "own size.  This is what a mutant is: the champion, "
                         "moved a little.")
    args = ap.parse_args()

    import torch
    from model import JungleNNUE

    if args.random:
        torch.manual_seed(args.seed)
        model = JungleNNUE(args.ft_hidden, args.hidden)
        model.eval()
    else:
        if not args.net:
            print("either --net or --random is needed", file=sys.stderr)
            return 2
        model = load_nnue(args.net)
        model.eval()

    if args.noise > 0.0:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            for name, p in model.state_dict().items():
                if not p.is_floating_point() or p.numel() <= 1:
                    continue
                scale = float(p.std())
                if not (scale > 0.0) or scale != scale:  # nan check
                    scale = 0.1
                p.add_(torch.randn_like(p) * (args.noise * scale))
        print(f"perturbed the weights by {args.noise}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "ft_hidden": model.ft_hidden,
                "hidden": model.hidden,
                "epoch": 0,
                "val_loss": float("nan"),
                "corr": float("nan")}, args.out)
    print(f"wrote {args.out}  (ft_hidden {model.ft_hidden}, hidden {model.hidden})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
