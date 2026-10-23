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
    ap.add_argument("--budget-minutes", type=float, default=0.0,
                    help="train for roughly this many minutes; the learning "
                         "rate schedule is re-aimed at the budget so it decays "
                         "across the whole run instead of stopping part-way "
                         "through a shorter schedule")
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
    if n < 32:
        print(f"ERROR: only {n} positions -- not enough to train", file=sys.stderr)
        return 2
    if n < 1000:
        print(f"WARNING: only {n} positions; this will not produce a usable network")

    tr_idx, va_idx = split_indices(n, args.val_fraction, args.seed)
    # Keep the board as uint8: 63 bytes per position instead of 504.  It is
    # widened to int64 per batch, which is where the feature indices need it.
    board_t = torch.from_numpy(board)
    stm_t = torch.from_numpy(stm)
    result_t = torch.from_numpy(result)
    value_t = torch.from_numpy(mcts_value)
    print(f"  train {len(tr_idx)}  val {len(va_idx)}", flush=True)

    # --- gradient self-check ----------------------------------------------
    #
    # The straight-through estimator that carries the gradient through the
    # int8 rounding is easy to break, and a broken one is invisible in the
    # loss curve: the loss simply plateaus at the variance of the target and
    # looks like an ordinary "the model is not very good".  Every parameter
    # has to receive a gradient, so check it before spending an hour on a run
    # that cannot learn anything.
    probe = JungleNNUE(args.ft_hidden, args.hidden).to(device)
    probe.train()
    idx = tr_idx[: min(512, len(tr_idx))]
    pb = board_t[idx].to(device).long()
    ps = stm_t[idx].to(device)
    py = ((1.0 - args.result_weight) * value_t[idx].to(device)
          + args.result_weight * result_t[idx].to(device)).float()
    pfr, pfb = make_features(pb, ps)
    ploss = torch.nn.functional.mse_loss(probe(pfr, pfb, ps).float(), py)
    ploss.backward()
    dead = [name for name, p in probe.named_parameters()
            if p.grad is None or float(p.grad.abs().max()) == 0.0]
    if dead:
        print(f"ERROR: these parameters receive no gradient: {dead}", file=sys.stderr)
        print("       The straight-through estimator is broken; the network "
              "cannot learn anything this way.", file=sys.stderr)
        return 3
    print("  gradient self-check: all parameters receive a gradient", flush=True)
    del probe

    # The dataset stores both the MCTS value and the game result from the
    # point of view of the side to move, which is exactly what the network
    # outputs, so no sign flipping is needed here.
    w = args.result_weight
    target_all = (1.0 - w) * value_t + w * result_t

    # Keep everything on the GPU when it fits -- the feature construction is
    # then essentially free.  The board is kept as uint8 (63 bytes per
    # position), so the whole 3.7M-position dataset is under 300 MB.
    gpu_bytes = (board_t.numel() * board_t.element_size()
                 + stm_t.numel() * stm_t.element_size()
                 + target_all.numel() * target_all.element_size())
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

    # Per-layer learning rates.
    #
    # Every layer's weights are quantised to int8 per output row, so the
    # smallest change that affects the forward pass is one quantisation
    # step, max|W|/127.  Adam moves every parameter by roughly `lr` per
    # step no matter how big its gradient is, so a single global learning
    # rate trains a layer whose weights are large far too slowly and a layer
    # whose weights are small far too violently.  The weight magnitudes in
    # this network span five orders of magnitude (the feature transformer
    # needs large weights because its input is binary, the output layer
    # needs tiny ones because its output is in [-1,1]), and the quantisation
    # steps span the same factor -- so the mismatch between the learning
    # rate and the step spans 40,000x.
    #
    # Scaling the learning rate with the layer's weight magnitude makes
    # lr/step the same for every layer, which is what makes a single
    # `--lr` meaningful at all.
    layers = [("ft", model.ft), ("h1", model.h1), ("h2", model.h2), ("out", model.out)]

    # The layer's quantisation step is max|W|/127, and the learning rate is
    # scaled with the weight magnitude so that lr/step is the same for every
    # layer.  The scale is capped at a small multiple of the initial
    # magnitude: if the weights ever grew, the learning rate would grow
    # with them, and that is a feedback loop.
    lr_cap = {}

    def layer_lr(layer, name):
        mag = float(layer.weight.detach().abs().max())
        cap = lr_cap.get(name)
        if cap is None:
            cap = lr_cap[name] = 3.0 * mag
        return args.lr * min(mag, cap)

    param_groups = []
    layer_groups = []
    for name, layer in layers:
        w_group = {"params": [layer.weight], "lr": layer_lr(layer, name),
                   "base_lr": layer_lr(layer, name), "name": f"{name}.weight"}
        # The gain is not quantised, so it keeps the plain learning rate.
        g_group = {"params": [layer.log_gain], "lr": args.lr,
                   "base_lr": args.lr, "name": f"{name}.gain"}
        param_groups.append(w_group)
        param_groups.append(g_group)
        layer_groups.append((name, layer, w_group))

    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4,
                            betas=(0.9, 0.95))
    print("  per-layer learning rates: " + "  ".join(
        f"{g['name']}={g['lr']:.2e}" for g in param_groups), flush=True)

    steps_per_epoch = max(1, len(tr_idx) // args.batch)
    # When a wall-clock budget is given the number of epochs is not known
    # in advance -- it depends on how fast the machine is.  The cosine
    # schedule is therefore re-aimed at the budget after every epoch, so the
    # learning rate decays across the whole budget instead of stopping
    # part-way through a schedule that was sized for fewer epochs.
    budget = args.budget_minutes if args.budget_minutes > 0 else args.max_minutes
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

    epoch = 0
    max_epochs = args.epochs if budget <= 0 else 10 ** 9
    while epoch < max_epochs:
        epoch += 1
        model.train()
        running, seen = 0.0, 0
        # Refresh the per-layer scaling: the weight magnitudes drift during
        # training, and the learning rate has to follow them to keep the
        # lr/quantisation-step ratio constant.
        for name, layer, w_group in layer_groups:
            w_group["base_lr"] = layer_lr(layer, name)
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
                g["lr"] = g["base_lr"] * lr_at(step) / args.lr

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

            if budget and (time.time() - start) / 60 > budget:
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

        # Activation health.  A layer whose units are clipped away is wasting
        # its capacity, and a layer that saturates is on its way to the
        # runaway that the learning-rate scaling can otherwise cause.
        if epoch == 1 or epoch % 10 == 0:
            try:
                st = model.activation_stats(board_t[va_idx[:2048]].to(device).long(),
                                            stm_t[va_idx[:2048]].to(device))
                print("    activations: " + "  ".join(
                    f"{k}={v:.3f}" for k, v in st.items()), flush=True)
            except Exception as exc:      # diagnostics must never break a run
                print(f"    (activation stats unavailable: {exc})", flush=True)

        # Re-aim the cosine schedule at the wall-clock budget.  The number of
        # epochs that fit in the budget is only known once the machine's
        # speed is known, and it changes as the data set grows.
        if args.budget_minutes > 0:
            elapsed_min = max((time.time() - start) / 60, 1e-9)
            epochs_fitting = max(1, int(args.budget_minutes / elapsed_min))
            total_steps = max(warmup + 1, steps_per_epoch * epochs_fitting)
            if epoch == 1 or epoch % 10 == 0:
                print(f"    budget {args.budget_minutes:.0f} min -> "
                      f"{epochs_fitting} epochs scheduled", flush=True)

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
    rc = main()
    # PyTorch's MPS backend intermittently segfaults while the interpreter
    # tears down, *after* the training has finished and the checkpoint has
    # been written.  In a Colab cell (or under `set -e`) that turns a
    # successful run into a failed one.  Everything that matters is on disk
    # by now, so skip the teardown entirely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
