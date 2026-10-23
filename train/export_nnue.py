#!/usr/bin/env python3
"""Export a trained Jungle NNUE checkpoint to the binary format read by the
C++ engine (``engine/src/nnue.hpp``).

The exported file is self-describing:

    char[8]  "JNNUE\\0\\0\\0"
    uint32   version = 3
    uint32   ft_hidden
    uint32   hidden
    uint32   num_layers = 2
    uint32   num_buckets = 16
    uint32   num_squares = 63
    int8     ft_w     [1008][ft_hidden]   (transposed)
    float32  ft_scale [ft_hidden]
    float32  ft_bias  [ft_hidden]
    int8     h0_w     [hidden][ft_hidden]
    float32  h0_scale [hidden]
    float32  h0_bias  [hidden]
    int8     h1_w     [hidden][hidden]
    float32  h1_scale [hidden]
    float32  h1_bias  [hidden]
    int8     out_w    [2*hidden]
    float32  out_scale
    float32  out_bias

The feature transformer is written transposed with respect to the torch
layout ``[ft_hidden, 1008]``: the C++ engine updates the accumulators one
piece at a time, and the transposed layout lets a single piece update touch
one contiguous run of weights instead of striding through the whole matrix.
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import JungleNNUE  # noqa: E402

MAGIC = b"JNNUE\x00\x00\x00"
NUM_BUCKETS = 16
NUM_SQUARES = 63


def export(model, path):
    ft_w, ft_scale, ft_bias = model.ft.export()
    h0_w, h0_scale, h0_bias = model.h1.export()
    h1_w, h1_scale, h1_bias = model.h2.export()
    out_w, out_scale, out_bias = model.out.export()

    # [ft_hidden, FT_INPUT] -> [FT_INPUT, ft_hidden]
    ft_w_t = np.ascontiguousarray(ft_w.T)

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<6I", 3, model.ft_hidden, model.hidden, 2,
                            NUM_BUCKETS, NUM_SQUARES))
        f.write(ft_w_t.astype(np.int8).tobytes())
        f.write(ft_scale.astype(np.float32).tobytes())
        f.write(ft_bias.astype(np.float32).tobytes())
        f.write(h0_w.astype(np.int8).tobytes())
        f.write(h0_scale.astype(np.float32).tobytes())
        f.write(h0_bias.astype(np.float32).tobytes())
        f.write(h1_w.astype(np.int8).tobytes())
        f.write(h1_scale.astype(np.float32).tobytes())
        f.write(h1_bias.astype(np.float32).tobytes())
        f.write(out_w.astype(np.int8).tobytes())
        f.write(np.float32(out_scale[0]).tobytes())
        f.write(np.float32(out_bias[0]).tobytes())

    return {
        "ft_hidden": int(model.ft_hidden),
        "hidden": int(model.hidden),
        "bytes": os.path.getsize(path),
    }


def load_nnue(path):
    """Read a binary network back into a torch model.

    Useful for checking the file the engine actually plays, rather than the
    checkpoint it came from: if the two ever disagree, the export is wrong.
    """
    import torch
    from model import JungleNNUE

    with open(path, "rb") as f:
        blob = f.read()

    off = 0

    def take(n):
        nonlocal off
        chunk = blob[off:off + n]
        off += n
        return chunk

    magic = take(8)
    if magic != MAGIC:
        raise ValueError(f"{path}: not a JNNUE file")
    version, ft_hidden, hidden, num_layers, buckets, squares = struct.unpack(
        "<6I", take(24))
    if version != 3:
        raise ValueError(f"{path}: unsupported version {version}")
    if buckets != NUM_BUCKETS or squares != NUM_SQUARES or num_layers != 2:
        raise ValueError(f"{path}: unexpected architecture")

    ft_input = NUM_SQUARES * NUM_BUCKETS

    def f32(n):
        return np.frombuffer(take(4 * n), dtype="<f4")

    def i8(n):
        return np.frombuffer(take(n), dtype=np.int8)

    model = JungleNNUE(ft_hidden, hidden)

    ft_w = i8(ft_input * ft_hidden).reshape(ft_input, ft_hidden).T.copy()
    ft_scale = f32(ft_hidden).copy()
    ft_bias = f32(ft_hidden).copy()
    # The export folds the gain into the weights, so the file holds the
    # effective weights; putting them back with a gain of one reproduces the
    # forward pass exactly.
    with torch.no_grad():
        model.ft.weight.copy_(torch.from_numpy(
            (ft_w.astype(np.float32) * ft_scale[:, None])))
        model.ft.bias.copy_(torch.from_numpy(ft_bias))
        model.ft.log_gain.zero_()

        for idx, (n_out, n_in) in enumerate(
                [(hidden, ft_hidden), (hidden, hidden)]):
            w = i8(n_out * n_in).reshape(n_out, n_in).copy()
            scale = f32(n_out).copy()
            bias = f32(n_out).copy()
            layer = model.h1 if idx == 0 else model.h2
            layer.weight.copy_(torch.from_numpy(
                w.astype(np.float32) * scale[:, None]))
            layer.bias.copy_(torch.from_numpy(bias))
            layer.log_gain.zero_()

        ow = i8(2 * hidden).copy()
        oscale = float(f32(1)[0])
        obias = float(f32(1)[0])
        model.out.weight.copy_(torch.from_numpy(
            ow.astype(np.float32) * np.float32(oscale)))
        model.out.bias.copy_(torch.tensor(np.float32(obias)))
        model.out.log_gain.zero_()

    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="", help="also print a checksum of the weights")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = JungleNNUE(ckpt.get("ft_hidden", 512), ckpt.get("hidden", 256))
    model.load_state_dict(ckpt["model"])
    model.eval()

    info = export(model, args.out)
    print(f"exported {args.out}: {info['bytes']/1024:.1f} KiB "
          f"(ft_hidden={info['ft_hidden']}, hidden={info['hidden']})")
    if args.check:
        w = torch.cat([p.detach().flatten() for p in model.parameters()])
        print(f"  weight checksum: {float(w.sum()):.6f}  absmax {float(w.abs().max()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
