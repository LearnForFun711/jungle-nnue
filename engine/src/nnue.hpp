// nnue.hpp -- Int8 NNUE network: binary format, loading and inference.
//
// The architecture follows Stockfish's "perspective" network:
//
//     features (2 perspectives) -> FT(1008 -> 512) -> H1(512 -> 512)
//                                        -> H2(512 -> 512)
//                                        -> OUT(concat -> 1)
//
// Two perspectives are used:
//   * RED   -- Red is "own", board as-is.
//   * BLACK -- Black is "own", board flipped vertically so that each side's
//               own den always sits at (8,3).
//
// Because the board is vertically symmetric (dens mirror, traps mirror, the
// river mirrors) this flip is a legal relabelling of the position and makes
// the network's job much easier.
//
// Quantisation (per output row, exactly reproduced by train/model.py):
//   M[o]   = max_i |W[o][i]|
//   Wq[o]  = round(W[o] / M[o] * 127)                (int8)
//   scale[o] = M[o] / 127
//   forward: acc[o] = sum_i Wq[o][i] * x[i]           (int32)
//             y[o]   = clamp(round(acc[o] * scale[o]), 0, 127)
// The input of every layer after the FT is therefore an int8 in [0,127].
#pragma once

#include <cmath>
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "eval.hpp"
#include "jungle.hpp"

namespace jungle {

constexpr int NUM_BUCKETS = 16;  // 8 own piece types + 8 their piece types
constexpr int FT_INPUT    = NUM_SQ * NUM_BUCKETS;  // 1008

// square s (as seen from Black's perspective) -> flipped square
inline int flip_square(int s) {
    const int r = s / WIDTH, c = s % WIDTH;
    return (HEIGHT - 1 - r) * WIDTH + c;
}

// Feature index for a piece standing on `sq` (board coordinates) seen from
// `own_color`'s perspective.
inline int feature_index(int sq, int piece, int own_color) {
    const int t = piece_type(piece) - 1;  // 0..7
    const bool own = (piece_color(piece) == own_color);
    const int s = (own_color == RED) ? sq : flip_square(sq);
    const int bucket = own ? t : (NUM_BUCKETS / 2 + t);
    return s * NUM_BUCKETS + bucket;
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

struct NnueNetwork {
    uint32_t ft_hidden = 0;
    uint32_t hidden    = 0;
    uint32_t num_layers = 2;

    std::vector<int8_t>  ft_w;    // [FT_INPUT][ft_hidden]  (transposed)
    std::vector<float>   ft_scale;// [ft_hidden]
    std::vector<int8_t>  h_w[2];  // [hidden][in]
    std::vector<float>   h_scale[2];
    std::vector<int8_t>  out_w;   // [2*hidden]
    float                out_scale = 0.f;

    // Runtime state
    std::vector<int32_t> acc[2];  // FT accumulators for RED / BLACK perspective

    bool valid() const { return ft_hidden > 0 && hidden > 0 && !ft_w.empty(); }
};

// ---------------------------------------------------------------------------
// File format
// ---------------------------------------------------------------------------
//
//   char[8]  "JNNUE\0\0\0"
//   uint32   version = 2
//   uint32   ft_hidden
//   uint32   hidden
//   uint32   num_layers
//   uint32   num_buckets
//   uint32   num_squares
//   int8     ft_w    [FT_INPUT][ft_hidden]   (transposed, see below)
//   float32  ft_scale[ft_hidden]
//   int8     h0_w    [hidden][ft_hidden]
//   float32  h0_scale[hidden]
//   int8     h1_w    [hidden][hidden]
//   float32  h1_scale[hidden]
//   int8     out_w   [2*hidden]
//   float32  out_scale
//
// All scalars are little-endian.
//
// The feature transformer is stored *transposed* with respect to the natural
// [ft_hidden][FT_INPUT] layout: feature `f` owns a contiguous run of
// `ft_hidden` weights.  A search updates the accumulators one piece at a
// time, and in the natural layout a single piece update strides through the
// whole 250 KiB matrix (one cache line per hidden unit).  Transposed, one
// piece touches two contiguous 256-byte runs instead, which is what makes the
// incremental evaluation affordable.

inline bool nnue_load(const std::string& path, NnueNetwork& net) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    struct Closer {
        std::FILE* f;
        ~Closer() { std::fclose(f); }
    } closer{f};

    auto read_raw = [&](void* dst, size_t n) {
        return std::fread(dst, 1, n, f) == n;
    };
    auto read_u32 = [&](uint32_t& v) { return read_raw(&v, 4); };

    char magic[8];
    if (!read_raw(magic, 8)) return false;
    if (std::memcmp(magic, "JNNUE\0\0\0", 8) != 0) return false;

    uint32_t version = 0, buckets = 0, squares = 0;
    if (!read_u32(version) || !read_u32(net.ft_hidden) || !read_u32(net.hidden) ||
        !read_u32(net.num_layers) || !read_u32(buckets) || !read_u32(squares))
        return false;
    if (version != 2 || buckets != NUM_BUCKETS || squares != NUM_SQ) return false;
    if (net.num_layers != 2) return false;

    net.ft_w.resize(size_t(FT_INPUT) * net.ft_hidden);
    net.ft_scale.resize(net.ft_hidden);
    net.h_w[0].resize(size_t(net.hidden) * net.ft_hidden);
    net.h_scale[0].resize(net.hidden);
    net.h_w[1].resize(size_t(net.hidden) * net.hidden);
    net.h_scale[1].resize(net.hidden);
    net.out_w.resize(size_t(2) * net.hidden);
    net.out_scale = 0.f;

    if (!read_raw(net.ft_w.data(), net.ft_w.size())) return false;
    if (!read_raw(net.ft_scale.data(), net.ft_scale.size() * 4)) return false;
    if (!read_raw(net.h_w[0].data(), net.h_w[0].size())) return false;
    if (!read_raw(net.h_scale[0].data(), net.h_scale[0].size() * 4)) return false;
    if (!read_raw(net.h_w[1].data(), net.h_w[1].size())) return false;
    if (!read_raw(net.h_scale[1].data(), net.h_scale[1].size() * 4)) return false;
    if (!read_raw(net.out_w.data(), net.out_w.size())) return false;
    if (!read_raw(&net.out_scale, 4)) return false;

    net.acc[0].assign(net.ft_hidden, 0);
    net.acc[1].assign(net.ft_hidden, 0);
    return true;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

// Dot product of two int8 vectors, accumulated in int32.
//
// The inner loop is the single hottest piece of code in the engine: it runs
// ~400k times per evaluation, so the difference between a scalar loop and a
// vectorised one is the difference between the neural engine being able to
// search as deeply as the handcrafted one and not.
inline int32_t dot_i8(const int8_t* w, const int8_t* x, int n) {
    int32_t sum = 0;
    int i = 0;
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= n; i += 32) {
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i wlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv));
        const __m256i whi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv, 1));
        const __m256i xlo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv));
        const __m256i xhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wlo, xlo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(whi, xhi));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    sum = _mm_cvtsi128_si32(s);
#endif
    for (; i < n; ++i) sum += int32_t(w[i]) * int32_t(x[i]);
    return sum;
}

// round-to-nearest-even (matches torch.round) without a libm call.
inline int32_t fast_round(float v) {
#if defined(__SSE4_1__) || defined(__AVX__)
    return _mm_cvtss_si32(_mm_set_ss(v));
#else
    return int32_t(std::nearbyint(v));
#endif
}

// Add / remove the contribution of one piece to the two FT accumulators.
//
// The feature transformer is stored *transposed*: feature `f` owns a
// contiguous run of `ft_hidden` weights.  That matters a lot -- with the
// natural [ft_hidden][FT_INPUT] layout a single piece update strides through
// the whole matrix and thrashes the cache, which made the refresh the
// dominant cost of an evaluation.  Transposed, one piece touches two
// contiguous 256-byte runs.
inline void nnue_add_piece(NnueNetwork& net, int sq, int piece, int sign) {
    const int f_red   = feature_index(sq, piece, RED);
    const int f_black = feature_index(sq, piece, BLACK);
    const int8_t* row_red   = net.ft_w.data() + size_t(f_red) * net.ft_hidden;
    const int8_t* row_black = net.ft_w.data() + size_t(f_black) * net.ft_hidden;
    if (sign > 0) {
        for (uint32_t h = 0; h < net.ft_hidden; ++h) {
            net.acc[RED][h]   += row_red[h];
            net.acc[BLACK][h] += row_black[h];
        }
    } else {
        for (uint32_t h = 0; h < net.ft_hidden; ++h) {
            net.acc[RED][h]   -= row_red[h];
            net.acc[BLACK][h] -= row_black[h];
        }
    }
}

// Update the accumulators for a move that has just been made (or undone).
//
// `moved` is the piece that moved (always non-zero), `captured` the piece that
// stood on the destination (0 when the move was quiet).  `forward` selects
// make-move (true) or unmake-move (false).
inline void nnue_move_piece(NnueNetwork& net, int from, int to,
                             int moved, int captured, bool forward) {
    if (forward) {
        if (captured) nnue_add_piece(net, to, captured, -1);
        nnue_add_piece(net, from, moved, -1);
        nnue_add_piece(net, to, moved, +1);
    } else {
        nnue_add_piece(net, to, moved, -1);
        nnue_add_piece(net, from, moved, +1);
        if (captured) nnue_add_piece(net, to, captured, +1);
    }
}

// Recompute both accumulators from scratch.
inline void nnue_refresh(NnueNetwork& net, const Position& pos) {
    std::fill(net.acc[RED].begin(), net.acc[RED].end(), 0);
    std::fill(net.acc[BLACK].begin(), net.acc[BLACK].end(), 0);
    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = pos.board[s];
        if (p) nnue_add_piece(net, s, p, +1);
    }
}

struct NnueOutput {
    int8_t ft_red[512];
    int8_t ft_black[512];
    int8_t h0_red[512], h0_black[512];
    int8_t h1_red[512], h1_black[512];
};

inline int8_t clip127(int32_t v) {
    if (v < 0) return 0;
    if (v > 127) return 127;
    return int8_t(v);
}

// Run the network forward and return the raw value *from the side to move's
// point of view*.  `stm` is the side to move (RED or BLACK); it is the only
// part of the position the forward pass needs, because the feature
// accumulators already carry the rest.
inline float nnue_forward(const NnueNetwork& net, int stm) {
    const int FH = net.ft_hidden;
    const int H  = net.hidden;

    int8_t ft_red[1024], ft_black[1024];
    int8_t h0_red[1024], h0_black[1024];
    int8_t h1_red[1024], h1_black[1024];

    for (int h = 0; h < FH; ++h) {
        ft_red[h]   = clip127(fast_round(float(net.acc[RED][h])   * net.ft_scale[h]));
        ft_black[h] = clip127(fast_round(float(net.acc[BLACK][h]) * net.ft_scale[h]));
    }

    const int8_t* w0 = net.h_w[0].data();
    const int8_t* w1 = net.h_w[1].data();
    for (int o = 0; o < H; ++o) {
        h0_red[o]   = clip127(fast_round(float(dot_i8(w0 + size_t(o) * FH, ft_red, FH))   * net.h_scale[0][o]));
        h0_black[o] = clip127(fast_round(float(dot_i8(w0 + size_t(o) * FH, ft_black, FH)) * net.h_scale[0][o]));
    }
    for (int o = 0; o < H; ++o) {
        h1_red[o]   = clip127(fast_round(float(dot_i8(w1 + size_t(o) * H, h0_red, H))   * net.h_scale[1][o]));
        h1_black[o] = clip127(fast_round(float(dot_i8(w1 + size_t(o) * H, h0_black, H)) * net.h_scale[1][o]));
    }

    // The two perspectives are ordered by the side to move, exactly as
    // Stockfish does, so the network can learn the tempo effect.  The value
    // that comes out is therefore already from the side to move's point of
    // view and must not be sign-flipped anywhere else.
    int8_t joint[2 * 1024];
    if (stm == RED) {
        std::memcpy(joint, h1_red, H);
        std::memcpy(joint + H, h1_black, H);
    } else {
        std::memcpy(joint, h1_black, H);
        std::memcpy(joint + H, h1_red, H);
    }
    const int32_t acc = dot_i8(net.out_w.data(), joint, 2 * H);
    return float(acc) * net.out_scale;
}

// ---------------------------------------------------------------------------
// Evaluator abstraction: handcrafted or neural
//
// The network is held *by value* so that every thread can own its own copy
// (the feature accumulators are mutated during evaluation).
// ---------------------------------------------------------------------------

struct Evaluator {
    NnueNetwork net;  // invalid => use the handcrafted evaluation

    bool uses_nnue() const { return net.valid(); }

    // Side to move that the cached accumulators belong to.  The search keeps
    // it in step with the position; the forward pass needs it to order the
    // two perspectives.
    int stm_ = RED;

    // Full evaluation: refreshes the accumulators from the position first.
    int eval(const Position& pos) {
        if (uses_nnue()) {
            nnue_refresh(net, pos);
            stm_ = pos.stm;
            return forward_value();
        }
        return jungle::evaluate(pos);
    }

    // Evaluation of the position the accumulators currently describe.
    // Without a network this falls back to the handcrafted evaluation, which
    // does need the position -- hence the argument.
    int eval_cached(const Position& pos) {
        if (!uses_nnue()) return jungle::evaluate(pos);
        return forward_value();
    }

    // Bring the accumulators in line with `pos` from scratch.
    void refresh(const Position& pos) {
        if (!uses_nnue()) return;
        nnue_refresh(net, pos);
        stm_ = pos.stm;
    }

    // Account for a move that has just been made (forward) or unmade
    // (!forward).  The side to move flips in both cases.
    void on_move(int from, int to, int moved, int captured, bool forward) {
        if (!uses_nnue()) return;
        nnue_move_piece(net, from, to, moved, captured, forward);
        stm_ ^= 1;
    }

    int forward_value() const {
        const float v = nnue_forward(net, stm_);  // side to move's view
        return int(std::lround(v * 1000.0f));
    }
};

}  // namespace jungle
