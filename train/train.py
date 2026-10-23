#!/usr/bin/env python3
"""Train the Jungle NNUE network on MCTS self-play data.

The target for every position is a blend of

  * the MCTS teacher's value estimate (dense, low variance), and
  * the final result of the self-play game (the ground truth),

both taken from the point of view of the side to move and then converted to
Red's point of view, which is what the network outputs.

The network is trained *quantisation aware*: the forward pass uses the same
int8 weights and the same rounding/clipping as the C++ inference code, so the
exported network behaves exactly like the trained one.

Example (on a Colab A100):

    python3 train.py --data "data/data.t*.bin" --epochs 12 --batch 8192 \
        --ft-hidden 512 --hidden 256 --out runs/exp1
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import load_shards, split_indices  # noqa: E402
from model import JungleNNUE, make_features  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(n, batch_size, shuffle=True, seed=0):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def to_gpu(arrays, device):
    out = []
    for a in arrays:
        t = torch.from_numpy(a)
        out.append(t.to(device, non_blocking=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.t*.bin")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--ft-hidden", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--result-weight", type=float, default=0.25,
                    help="weight of the game-result term in the target blend")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"device: {device}", flush=True)

    print("loading data...", flush=True)
    t0 = time.time()
    board, stm, result, mcts_value, ply, _cause = load_shards(args.data)
    n = board.shape[0]
    print(f"  {n} positions loaded in {time.time()-t0:.1f}s", flush=True)
    if n < 1000:
        print("ERROR: not enough data", file=sys.stderr)
        return 2

    tr_idx, va_idx = split_indices(n, args.val_fraction, args.seed)
    # Keep the board as uint8: 63 bytes per position instead of 504.  It is
    # widened to int64 per batch, which is where the feature indices need it.
    board_t = torch.from_numpy(board)
    stm_t = torch.from_numpy(stm)
    result_t = torch.from_numpy(result)
    value_t = torch.from_numpy(mcts_value)
    print(f"  train {len(tr_idx)}  val {len(va_idx)}", flush=True)

    # The dataset stores both the MCTS value and the game result from the
    # point of view of the side to move, which is exactly what the network
    # outputs, so no sign flipping is needed here.
    w = args.result_weight
    target_all = (1.0 - w) * value_t + w * result_t

    # Keep everything on the GPU when it fits -- the feature construction is
    # then essentially free.
    gpu_bytes = board.size
    if device.type == "cuda" and gpu_bytes < 8e9:
        board_all = board_t.to(device)
        stm_all = stm_t.to(device)
        target_all = target_all.to(device)
        resident = True
    else:
        board_all, stm_all = board_t, stm_t
        target_all = target_all.to(device)
        resident = False
    target_all = target_all.contiguous()
    print(f"  resident on GPU: {resident}", flush=True)

    model = JungleNNUE(args.ft_hidden, args.hidden).to(device)
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["model"])
        print(f"  resumed from {args.resume}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(tr_idx) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, args.warmup_steps)

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        p = min(1.0, p)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    step = 0
    start = time.time()
    stop = False

    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        perm = tr_idx.copy()
        np.random.default_rng(args.seed + epoch).shuffle(perm)
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            if resident:
                ti = torch.from_numpy(idx).to(device, non_blocking=True)
                b = board_all[ti].long()
                s = stm_all[ti]
                y = target_all[ti]
            else:
                ti = torch.from_numpy(idx)
                b = board_t[ti].to(device, non_blocking=True).long()
                s = stm_t[ti].to(device, non_blocking=True)
                y = target_all[ti.to(device)]

            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                fr, fb = make_features(b, s)
                v = model(fr, fb, s)
                loss = torch.nn.functional.mse_loss(v.float(), y.float())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item() * len(idx)
            seen += len(idx)
            step += 1

            if step % 200 == 0:
                el = time.time() - start
                print(f"  epoch {epoch} step {step}/{total_steps} "
                      f"loss {running/max(1,seen):.5f} lr {lr_at(step):.2e} "
                      f"{seen/max(el,1e-9):.0f} pos/s elapsed {el/60:.1f}m", flush=True)
                running, seen = 0.0, 0

            if args.max_minutes and (time.time() - start) / 60 > args.max_minutes:
                print("  time budget reached, stopping", flush=True)
                stop = True
                break

        # --- validation ---------------------------------------------------
        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_idx), 8192):
                idx = va_idx[i:i + 8192]
                ti = torch.from_numpy(idx)
                if resident:
                    ti = ti.to(device)
                    b, s, y = board_all[ti].long(), stm_all[ti], target_all[ti]
                else:
                    b = board_t[ti].to(device).long()
                    s = stm_t[ti].to(device)
                    y = target_all[ti.to(device)]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    fr, fb = make_features(b, s)
                    v = model(fr, fb, s)
                vs += torch.nn.functional.mse_loss(v.float(), y.float()).item() * len(idx)
                vn += len(idx)
        val = vs / max(1, vn)
        print(f"epoch {epoch}: val_loss {val:.6f}  ({(time.time()-start)/60:.1f} min)", flush=True)

        ckpt = os.path.join(args.out, "last.pt")
        torch.save({"model": model.state_dict(), "ft_hidden": args.ft_hidden,
                    "hidden": args.hidden, "epoch": epoch, "val_loss": val}, ckpt)
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(), "ft_hidden": args.ft_hidden,
                        "hidden": args.hidden, "epoch": epoch, "val_loss": val},
                       os.path.join(args.out, "best.pt"))
            print(f"  new best val_loss {val:.6f}", flush=True)

        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump({"epoch": epoch, "val_loss": val, "best": best_val,
                       "minutes": (time.time() - start) / 60}, f, indent=2)

        if stop:
            break

    print(f"training done. best val_loss {best_val:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
