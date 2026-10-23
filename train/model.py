"""Stockfish-style NNUE for Jungle Chess, with quantisation-aware training.

The architecture and the quantisation are matched *exactly* to the C++
inference code in ``engine/src/nnue.hpp``:

    features (2 perspectives) -> FT(1008 -> ft_hidden) -> H1(ft_hidden -> hidden)
                                                 -> H2(hidden -> hidden)
                                                 -> OUT(concat -> 1)

    per-layer quantisation (per output row):
        M[o]     = max_i |W[o][i]|
        Wq[o][i] = round(W[o][i] / M[o] * 127)          (int8)
        scale[o] = M[o] / 127
        forward: acc[o] = sum_i Wq[o][i] * x[i]          (int32)
                 y[o]   = clamp(round(acc[o] * scale[o]), 0, 127)

Two perspectives are used:
  * RED   -- Red is "own", board as-is.
  * BLACK -- Black is "own", board flipped vertically, so that each side's own
              den is always at (8,3).  The board is vertically symmetric, so
              this flip is a legal relabelling.

Because the C++ side rounds after every layer, the torch side rounds too
(straight-through estimator) -- the two implementations then agree to within
one unit in the last place.

Activation scaling
------------------
The activations are clipped to [0,127], so the *magnitude* of the weights in
each layer decides where the activations land.  The required magnitude differs
by five orders of magnitude between the feature transformer (which sees
binary inputs) and the output layer, so every layer carries a learnable
scalar gain.  The gain is folded into the weights at export time; because the
per-row quantisation normalises each row to its own maximum, folding the gain
in leaves the int8 weights untouched and the two implementations agree
exactly.
"""
import math

import numpy as np
import torch
import torch.nn as nn

WIDTH = 7
HEIGHT = 9
NUM_SQ = WIDTH * HEIGHT
NUM_BUCKETS = 16
FT_INPUT = NUM_SQ * NUM_BUCKETS


def feature_indices(board, stm):
    """Build the two perspectives' feature indices.

    board: (B, 63) int64 tensor, 0 empty, 1..8 Red, 9..16 Black
    stm:   (B,)    int64 tensor, 0 Red to move, 1 Black to move

    Returns (idx_red, idx_black, mask) where mask is 1.0 for occupied squares.
    """
    b = board.shape[0]
    dev = board.device
    p = board
    occupied = (p > 0).to(torch.float32)

    is_red = (p >= 1) & (p <= 8)
    ptype = torch.where(p <= 8, p - 1, p - 9).clamp(min=0)

    s = torch.arange(NUM_SQ, device=dev, dtype=torch.int64).unsqueeze(0).expand(b, -1)
    r = s // WIDTH
    c = s % WIDTH
    sq_red = s
    sq_black = (HEIGHT - 1 - r) * WIDTH + c

    # Red perspective: own = Red, their = Black.
    bucket_red = torch.where(is_red, ptype, ptype + 8)
    idx_red = sq_red * NUM_BUCKETS + bucket_red

    # Black perspective: own = Black, their = Red (board is flipped).
    bucket_black = torch.where(is_red, ptype + 8, ptype)
    idx_black = sq_black * NUM_BUCKETS + bucket_black

    return idx_red, idx_black, occupied


def make_features(board, stm):
    """Sparse binary feature vectors for both perspectives: (B, FT_INPUT)."""
    b = board.shape[0]
    idx_red, idx_black, mask = feature_indices(board, stm)
    dev = board.device
    feat_red = torch.zeros(b, FT_INPUT, device=dev, dtype=torch.float32)
    feat_black = torch.zeros(b, FT_INPUT, device=dev, dtype=torch.float32)
    feat_red.scatter_add_(1, idx_red, mask)
    feat_black.scatter_add_(1, idx_black, mask)
    return feat_red, feat_black


def _ste_round_clip(x, lo=0.0, hi=127.0):
    """round + clamp with a straight-through gradient."""
    r = torch.round(x)
    return r + (r.clamp(lo, hi) - r).detach()


class QuantLinear(nn.Module):
    """Linear layer with per-output-row int8 quantisation (straight-through).

    The layer output is ``gain * (x @ W^T)`` and is expected to be clipped to
    [0,127] by the caller.  ``gain`` is a learnable positive scalar; it is
    folded into the weights when the network is exported.
    """

    def __init__(self, n_out, n_in, init_std=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * init_std)
        self.log_gain = nn.Parameter(torch.zeros(()))

    @property
    def gain(self):
        return torch.exp(self.log_gain)

    def quantised(self):
        """Return (int8 weights, per-row scale) exactly as the C++ side does."""
        m = self.weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        wq = torch.round(self.weight / m * 127.0).clamp(-127, 127)
        return wq, (m / 127.0).squeeze(1)

    def forward(self, x):
        wq, scale = self.quantised()
        wdq = wq * scale.unsqueeze(1)
        wdq = wdq.detach() + (self.weight - self.weight.detach())   # STE
        return (x @ wdq.t()) * self.gain

    def export(self):
        """int8 weights and per-row scales with the gain folded in."""
        gain = float(self.gain.detach())
        w = self.weight.detach() * gain
        m = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        wq = torch.round(w / m * 127.0).clamp(-127, 127)
        return wq.cpu().numpy().astype(np.int8), (m / 127.0).squeeze(1).cpu().numpy().astype(np.float32)


class JungleNNUE(nn.Module):
    """Two-perspective NNUE for Jungle Chess."""

    # Initial weight magnitudes chosen so that each layer's activations land
    # inside the [0,127] clipping range from the start.
    FT_STD = 9.0
    HID_STD = 0.09
    OUT_STD = 4.0e-4

    def __init__(self, ft_hidden=256, hidden=128):
        super().__init__()
        self.ft_hidden = ft_hidden
        self.hidden = hidden
        self.ft = QuantLinear(ft_hidden, FT_INPUT, self.FT_STD)
        self.h1 = QuantLinear(hidden, ft_hidden, self.HID_STD)
        self.h2 = QuantLinear(hidden, hidden, self.HID_STD)
        self.out = QuantLinear(1, 2 * hidden, self.OUT_STD)

    def forward(self, feat_red, feat_black, stm=None):
        y = _ste_round_clip(self.ft(feat_red))
        z = _ste_round_clip(self.ft(feat_black))
        a = _ste_round_clip(self.h1(y))
        b = _ste_round_clip(self.h1(z))
        c = _ste_round_clip(self.h2(a))
        d = _ste_round_clip(self.h2(b))
        if stm is None:
            joint = torch.cat([c, d], dim=1)
        else:
            # Order the perspectives by the side to move, exactly as the C++
            # inference does, so the network can learn the tempo effect.
            stm_red = (stm == 0).unsqueeze(1)
            first = torch.where(stm_red, c, d)
            second = torch.where(stm_red, d, c)
            joint = torch.cat([first, second], dim=1)
        v = self.out(joint).squeeze(-1)   # from the side to move's point of view
        return v

    def forward_board(self, board, stm):
        fr, fb = make_features(board, stm)
        return self(fr, fb, stm)

    def activation_stats(self, board, stm):
        """Diagnostics: how much of the [0,127] range the layers use."""
        fr, fb = make_features(board, stm)
        with torch.no_grad():
            y = self.ft(fr)
            z = self.ft(fb)
            a = _ste_round_clip(self.h1(y))
            c = _ste_round_clip(self.h2(a))
        return {
            "ft_mean": float(y.mean()), "ft_max": float(y.max()),
            "ft_saturated": float((y >= 127).float().mean()),
            "h1_mean": float(a.mean()), "h1_saturated": float((a >= 127).float().mean()),
            "h2_mean": float(c.mean()), "h2_saturated": float((c >= 127).float().mean()),
        }
