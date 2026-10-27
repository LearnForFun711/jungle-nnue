"""Dataset reader for the Jungle NNUE self-play shards.

Shard format (68 bytes per position, little endian) -- written by
``engine/tools/selfplay.cpp``:

    [0:63]   board: 0 empty, 1..8 Red piece (type), 9..16 Black piece (type)
    [63]     meta: bit0 = side to move (0 Red, 1 Black)
                     bits 2..3 = game result from the side-to-move's view
                                (0 loss, 1 draw, 2 win)
    [64]     MCTS value quantised from [-1,1] to [0,255]
    [65:67]  ply (uint16)
    [67]     ending cause: 0 den entry, 1 no legal moves, 2 repetition, 3 length
"""
import glob
import os

import numpy as np

RECORD_SIZE = 68
RESULT_CODES = {0: -1.0, 1: 0.0, 2: 1.0}


def load_shards(pattern):
    """Load every shard matching ``pattern`` into flat numpy arrays.

    The pattern is matched recursively, so ``data/**/*.bin`` picks up the
    shards of every round of data generation, not just the ones in the
    top-level directory.  Shards that hold a policy target rather than a
    value are skipped: they are not in this format.
    """
    files = sorted(glob.glob(pattern, recursive=True))
    files = [p for p in files if "policy" not in os.path.basename(p)]
    if not files:
        raise FileNotFoundError(f"no shards matched {pattern!r}")

    boards, stm, result, value, ply, cause = [], [], [], [], [], []
    for path in files:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size % RECORD_SIZE:
            raw = raw[: (raw.size // RECORD_SIZE) * RECORD_SIZE]
        n = raw.size // RECORD_SIZE
        if n == 0:
            continue
        rec = raw.reshape(n, RECORD_SIZE)
        boards.append(rec[:, 0:63])
        stm.append(rec[:, 63] & 1)
        code = (rec[:, 63] >> 2) & 3
        result.append(np.array([RESULT_CODES.get(int(c), 0.0) for c in code], dtype=np.float32))
        value.append(rec[:, 64].astype(np.float32) / 127.5 - 1.0)
        ply.append(rec[:, 65:67].copy().view(np.uint16).reshape(n).astype(np.int32))
        cause.append(rec[:, 67].copy())

    board = np.concatenate(boards, axis=0)
    stm = np.concatenate(stm).astype(np.int64)
    result = np.concatenate(result)
    value = np.concatenate(value)
    ply = np.concatenate(ply)
    cause = np.concatenate(cause)
    return board, stm, result, value, ply, cause


def split_indices(n, val_fraction=0.02, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    return perm[n_val:], perm[:n_val]
