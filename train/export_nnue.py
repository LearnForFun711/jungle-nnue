#!/usr/bin/env python3
"""Export a trained Jungle NNUE checkpoint to the binary format read by the
C++ engine (``engine/src/nnue.hpp``).

The exported file is self-describing:

    char[8]  "JNNUE\\0\\0\\0"
    uint32   version = 2
    uint32   ft_hidden
    uint32   hidden
    uint32   num_layers = 2
    uint32   num_buckets = 16
    uint32   num_squares = 63
    int8     ft_w     [1008][ft_hidden]   (transposed)
    float32  ft_scale [ft_hidden]
    int8     h0_w     [hidden][ft_hidden]
    float32  h0_scale [hidden]
    int8     h1_w     [hidden][hidden]
    float32  h1_scale [hidden]
    int8     out_w    [2*hidden]
    float32  out_scale

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
    ft_w, ft_scale = model.ft.export()
    h0_w, h0_scale = model.h1.export()
    h1_w, h1_scale = model.h2.export()
    out_w, out_scale = model.out.export()

    # [ft_hidden, FT_INPUT] -> [FT_INPUT, ft_hidden]
    ft_w_t = np.ascontiguousarray(ft_w.T)

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<6I", 2, model.ft_hidden, model.hidden, 2,
                            NUM_BUCKETS, NUM_SQUARES))
        f.write(ft_w_t.astype(np.int8).tobytes())
        f.write(ft_scale.astype(np.float32).tobytes())
        f.write(h0_w.astype(np.int8).tobytes())
        f.write(h0_scale.astype(np.float32).tobytes())
        f.write(h1_w.astype(np.int8).tobytes())
        f.write(h1_scale.astype(np.float32).tobytes())
        f.write(out_w.astype(np.int8).tobytes())
        f.write(np.float32(out_scale[0]).tobytes())

    return {
        "ft_hidden": int(model.ft_hidden),
        "hidden": int(model.hidden),
        "bytes": os.path.getsize(path),
    }


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
