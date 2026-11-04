// jungle_gpu.cuh -- The portable core of the GPU self-play generator.
//
// Everything here compiles both as CUDA and as plain C++.  The plain C++
// build is what the test suite runs on a machine without a GPU, and it
// exercises exactly the same rules, the same move generator and the same
// network arithmetic as the CUDA build -- so the parts that can be tested
// locally are tested locally.
//
// Three pieces live here:
//
//   * the move generator, written with bitboards so that the legal moves of
//     every piece of a type come out of one shift and a few masks;
//
//   * the network's forward pass, which is the same arithmetic as the
//     engine's, down to the rounding, and is therefore bit-identical to it;
//
//   * the MCTS teacher, written as a state machine so that a thread can stop
//     in the middle of a simulation, ask for the evaluation of a leaf, and
//     carry on once the answer arrives.  That is what lets a thousand games
//     share one batched network evaluation.
#pragma once

#include "jungle_gpu.h"

#include <cmath>

#if defined(__CUDACC__)
namespace jungle_gpu {
// The device's copy of the Zobrist table.  The host fills it in before the
// kernel starts, so a position hashes to the same value on both sides.
__device__ ZobristTable g_zobrist;
}  // namespace jungle_gpu
#endif

namespace jungle_gpu {

// The Zobrist table, wherever the code happens to be running.
inline JGPU_DEVICE const ZobristTable& zobrist() {
#if defined(__CUDACC__)
    return g_zobrist;
#else
    return host_zobrist();
#endif
}

JGPU_DEVICE inline uint64_t Board::compute_key() const {
    const ZobristTable& z = zobrist();
    uint64_t k = 0;
    for (int s = 0; s < NUM_SQ; ++s)
        if (sq[s]) k ^= z.piece[sq[s]][s];
    if (stm == BLACK) k ^= z.side;
    return k;
}

JGPU_DEVICE inline uint64_t Board::key_after_move(int from, int to) const {
    const ZobristTable& z = zobrist();
    const int moved = sq[from];
    const int captured = sq[to];
    uint64_t k = key;
    k ^= z.piece[moved][from];
    if (captured) k ^= z.piece[captured][to];
    k ^= z.piece[moved][to];
    k ^= z.side;              // the side to move always flips
    return k;
}

// ---------------------------------------------------------------------------
// Bitboard helpers
// ---------------------------------------------------------------------------

// The squares in column 0 (0, 7, 14, ... 56) and in column 6, and the last
// row (56..62).  The row mask matters because shifting a piece down off the
// bottom of the board would otherwise land it on a square that does not
// exist.
constexpr uint64_t make_col0() {
    uint64_t m = 0;
    for (int r = 0; r < HEIGHT; ++r) m |= 1ULL << (r * WIDTH);
    return m;
}
constexpr uint64_t make_row_last() {
    uint64_t m = 0;
    for (int c = 0; c < WIDTH; ++c) m |= 1ULL << ((HEIGHT - 1) * WIDTH + c);
    return m;
}
constexpr uint64_t COL0 = make_col0();
constexpr uint64_t COL_LAST = COL0 << (WIDTH - 1);
constexpr uint64_t ROW_LAST = make_row_last();

inline uint64_t bit(int s) { return 1ULL << s; }

// Move every piece of `m` one square in direction `d`.
inline JGPU_DEVICE uint64_t shift_dir(uint64_t m, int d) {
    switch (d) {
        case DIR_UP:    return m >> WIDTH;
        case DIR_DOWN:  return (m & ~ROW_LAST) << WIDTH;
        case DIR_LEFT:  return (m & ~COL0) >> 1;
        case DIR_RIGHT: return (m & ~COL_LAST) << 1;
    }
    return 0;
}

// Can a piece of type `rank` take a piece of type `target`?  This is the
// rank rule from engine/src/eval.hpp's can_capture_square(), without the
// trap and water rules.
inline JGPU_DEVICE bool rank_can_capture(int rank, int target) {
    if (rank == RAT && target == ELEPHANT) return true;
    if (rank == ELEPHANT && target == RAT) return false;
    return rank >= target;
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

// The engine allows 128 legal moves (engine/src/jungle.hpp uses the same
// bound); the buffer has to be at least that large, or a position with
// more legal moves than the buffer would lose some of them.
constexpr int MAX_MOVES = 128;

struct MoveBuf {
    uint8_t from[MAX_MOVES];
    uint8_t to[MAX_MOVES];
    int n = 0;
};

// Generate every legal move for the side to move.
//
// The legal destinations for a whole piece type and direction come out of a
// handful of masks:
//
//   dst = shift(src, d)                 the squares the pieces could reach
//   dst &= ~ours                        not onto one of our own pieces
//   dst &= ~our_den                     not into our own den
//   dst &= ~river_dst[d]                only rats swim
//   dst &= ~theirs                      the empty squares
//   dst &= capturable                   the enemy squares we may take
//   dst &= ~shift(river_cross[d], d)    and not across the water
//
// The capturable mask is the union of the enemy piece types our rank beats,
// plus the enemy pieces standing on our traps, which have lost their rank
// and can be taken by anything.
JGPU_DEVICE int generate_moves(const Board& b, const Geometry& g, MoveBuf& out) {
    out.n = 0;

    uint64_t bb[17];
    uint64_t occ[2] = {0, 0};
    for (int i = 0; i < 17; ++i) bb[i] = 0;
    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = b.sq[s];
        if (!p) continue;
        bb[p] |= bit(s);
        occ[piece_color(p)] |= bit(s);
    }

    const int color = b.stm;
    const uint64_t ours = occ[color];
    const uint64_t theirs = occ[color ^ 1];
    const uint64_t our_den = g.den_mask[color];
    const uint64_t our_trap = g.trap_mask[color];

    // The legal destination mask for every (piece type, direction).
    uint64_t legal[9][4];
    for (int t = 1; t <= 8; ++t) {
        const int piece = (color == RED) ? t : t + 8;
        const uint64_t src = bb[piece];
        uint64_t cap = 0;
        if (src) {
            for (int u = 1; u <= 8; ++u) {
                const int epiece = (color == RED) ? u + 8 : u;
                if (rank_can_capture(t, u)) cap |= bb[epiece];
            }
            // A piece on the opponent's trap has lost its rank.
            cap |= theirs & our_trap;
        }
        for (int d = 0; d < 4; ++d) {
            uint64_t s0 = src;
            if (t != RAT) s0 &= ~g.river_dst[d];      // only rats swim
            uint64_t dst = shift_dir(s0, d);
            dst &= ~ours;
            dst &= ~our_den;
            uint64_t empty = dst & ~theirs;
            uint64_t enemy = dst & theirs & cap;
            enemy &= ~shift_dir(g.river_cross[d], d);  // the water rule
            legal[t][d] = empty | enemy;
        }
    }

    // Emit the moves in the engine's order: by square, then by direction,
    // then the jumps.
    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = b.sq[s];
        if (!p || piece_color(p) != color) continue;
        const int t = piece_type(p);
        for (int d = 0; d < 4; ++d) {
            const int to = g.neighbor[s][d];
            if (to < 0) continue;
            if (!((legal[t][d] >> to) & 1ULL)) continue;
            if (out.n >= MAX_MOVES) return out.n;
            out.from[out.n] = uint8_t(s);
            out.to[out.n] = uint8_t(to);
            ++out.n;
        }
        if (t == TIGER || t == LION) {
            for (int k = 0; k < g.jump_count[s]; ++k) {
                const JumpGeom& j = g.jumps[s][k];
                bool blocked = false;
                for (int q = 0; q < j.count; ++q)
                    if (b.sq[j.path[q]] != 0) { blocked = true; break; }
                if (blocked) continue;
                const int target = b.sq[j.to];
                if (target != 0 && piece_color(target) == color) continue;
                if (g.den_owner[j.to] == color) continue;
                bool ok = (target == 0);
                if (!ok) {
                    int t_rank = piece_type(target);
                    if (g.trap_owner[j.to] == color) t_rank = 0;
                    ok = rank_can_capture(t, t_rank);
                }
                if (!ok) continue;
                if (out.n >= MAX_MOVES) return out.n;
                out.from[out.n] = uint8_t(s);
                out.to[out.n] = j.to;
                ++out.n;
            }
        }
    }
    return out.n;
}

// Apply a move to the board.  The caller keeps the history for repetition.
JGPU_DEVICE void apply_move(Board& b, int from, int to) {
    const int moved = b.sq[from];
    b.key = b.key_after_move(from, to);
    b.sq[to] = uint8_t(moved);
    b.sq[from] = 0;
    b.stm = uint8_t(b.stm ^ 1);
    b.ply = uint16_t(b.ply + 1);
}

// ---------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------

// The engine clips a value to [0,127] after rounding it to the nearest even
// number.  The rounding matters: it is the same one the AVX2 path uses
// (_mm_cvtps_si32), and it is why the GPU's forward pass comes out bit for
// bit the same as the engine's.
inline JGPU_DEVICE float clip_to_127(float v) {
    const float r = rintf(v);
    if (r < 0.f) return 0.f;
    if (r > 127.f) return 127.f;
    return r;
}

// The value of a position as the teacher sees it.
//
// The engine's evaluator returns an integer in thousandths:
//
//   eval(pos) = lround(value * 1000 * value_scale)
//
// and the teacher uses eval(pos) / 1000 * leaf_scale, clamped to [-1,1].
// The teacher depends on that scale: the priors are in the same units, and a
// leaf that is too small compared to them turns the search into a random
// walk.  So the GPU teacher does the same thing, with the same rounding, so
// that the two teachers produce the same numbers.
inline JGPU_DEVICE float teacher_leaf_value(float net_value, float leaf_scale,
                                             float value_scale) {
    const float thousandths = roundf(net_value * 1000.0f * value_scale);
    const float v = thousandths / 1000.0f * leaf_scale;
    if (v < -1.f) return -1.f;
    if (v > 1.f) return 1.f;
    return v;
}

// The forward pass, one position at a time.  This is the engine's
// arithmetic, in the same order, with the same rounding.
//
// The dot products are sums of products of small integers, so they are exact
// in fp32 and the result is identical to the engine's int8 path: the AVX2
// code the engine uses multiplies unsigned bytes by signed ones and sums
// pairs into int16, and with the activations clipped to [0,127] and the
// weights to [-127,127] no pair can reach the int16 limit, so nothing
// saturates and the sum is exact.
//
// `acc` is scratch space of 2 * ft_hidden int32s.
JGPU_DEVICE float nnue_eval(const Net& net, const uint8_t* board, int stm, int32_t* acc) {
    const int FH = net.ft_hidden;
    const int H = net.hidden;

    for (int h = 0; h < FH; ++h) { acc[h] = 0; acc[FH + h] = 0; }

    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = board[s];
        if (!p) continue;
        const int f_red = feature_index(s, p, RED);
        const int f_black = feature_index(s, p, BLACK);
        const int8_t* row_red = net.ft_w + size_t(f_red) * FH;
        const int8_t* row_black = net.ft_w + size_t(f_black) * FH;
        for (int h = 0; h < FH; ++h) {
            acc[h] += row_red[h];
            acc[FH + h] += row_black[h];
        }
    }

    // The feature transformer.
    int8_t ft_red[MAX_FT], ft_black[MAX_FT];
    for (int h = 0; h < FH; ++h) {
        ft_red[h] = int8_t(clip_to_127(float(acc[h]) * net.ft_scale[h] + net.ft_bias[h]));
        ft_black[h] = int8_t(clip_to_127(float(acc[FH + h]) * net.ft_scale[h] + net.ft_bias[h]));
    }

    // The hidden layers.
    int8_t h0_red[MAX_HIDDEN], h0_black[MAX_HIDDEN];
    for (int o = 0; o < H; ++o) {
        int32_t d_red = 0, d_black = 0;
        const int8_t* w = net.h_w[0] + size_t(o) * FH;
        for (int i = 0; i < FH; ++i) {
            d_red += int32_t(w[i]) * int32_t(ft_red[i]);
            d_black += int32_t(w[i]) * int32_t(ft_black[i]);
        }
        h0_red[o] = int8_t(clip_to_127(float(d_red) * net.h_scale[0][o] + net.h_bias[0][o]));
        h0_black[o] = int8_t(clip_to_127(float(d_black) * net.h_scale[0][o] + net.h_bias[0][o]));
    }
    int8_t h1_red[MAX_HIDDEN], h1_black[MAX_HIDDEN];
    for (int o = 0; o < H; ++o) {
        int32_t d_red = 0, d_black = 0;
        const int8_t* w = net.h_w[1] + size_t(o) * H;
        for (int i = 0; i < H; ++i) {
            d_red += int32_t(w[i]) * int32_t(h0_red[i]);
            d_black += int32_t(w[i]) * int32_t(h0_black[i]);
        }
        h1_red[o] = int8_t(clip_to_127(float(d_red) * net.h_scale[1][o] + net.h_bias[1][o]));
        h1_black[o] = int8_t(clip_to_127(float(d_black) * net.h_scale[1][o] + net.h_bias[1][o]));
    }

    // The two perspectives are ordered by the side to move, so the value
    // that comes out is already from the side to move's point of view.
    int32_t acc_out = 0;
    if (stm == RED) {
        for (int i = 0; i < H; ++i) {
            acc_out += int32_t(net.out_w[i]) * int32_t(h1_red[i]);
            acc_out += int32_t(net.out_w[H + i]) * int32_t(h1_black[i]);
        }
    } else {
        for (int i = 0; i < H; ++i) {
            acc_out += int32_t(net.out_w[i]) * int32_t(h1_black[i]);
            acc_out += int32_t(net.out_w[H + i]) * int32_t(h1_red[i]);
        }
    }
    return float(acc_out) * net.out_scale + net.out_bias;
}

}  // namespace jungle_gpu
