// jungle_gpu_common.cpp -- The board geometry and the position key.
//
// The geometry is everything about the Jungle board that does not change
// from position to position: the river, the traps, the dens, the neighbours,
// the jumps, and the masks that make the bitboard shifts work.
//
// The masks come in two families.  `river_dst[d]` holds the squares whose
// *destination* in direction d is on the river -- a non-rat may not move
// there.  `river_cross[d]` holds the squares whose river-ness *changes* when
// they move in direction d -- a capture across the water is not allowed.
#include "jungle_gpu.h"

#include <cstdio>
#include <initializer_list>
#include <cstring>

namespace jungle_gpu {

namespace {

// The same Zobrist constants the engine uses, so that a position hashes to
// the same value in both programs.  The device gets a copy of this table,
// filled in from the host before the kernel starts.
ZobristTable make_zobrist() {
    ZobristTable z{};
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    auto next = [&]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    for (int p = 0; p < 17; ++p)
        for (int s = 0; s < NUM_SQ; ++s) z.piece[p][s] = next();
    z.side = next();
    return z;
}

}  // namespace

const ZobristTable& host_zobrist() {
    static const ZobristTable z = make_zobrist();
    return z;
}

Geometry build_geometry() {
    Geometry g;
    g.river_mask = 0;
    g.trap_mask[RED] = g.trap_mask[BLACK] = 0;
    g.den_mask[RED] = g.den_mask[BLACK] = 0;

    for (int s = 0; s < NUM_SQ; ++s) {
        const int r = s / WIDTH, c = s % WIDTH;
        g.river[s] = (r >= 3 && r <= 5) && (c == 1 || c == 2 || c == 4 || c == 5);
        if (g.river[s]) g.river_mask |= 1ULL << s;

        g.trap_owner[s] = -1;
        g.den_owner[s] = -1;
        if (r == 0 && (c == 2 || c == 4)) g.trap_owner[s] = BLACK;
        if (r == 1 && c == 3)             g.trap_owner[s] = BLACK;
        if (r == 8 && (c == 2 || c == 4)) g.trap_owner[s] = RED;
        if (r == 7 && c == 3)             g.trap_owner[s] = RED;
        if (r == 0 && c == 3)             g.den_owner[s] = BLACK;
        if (r == 8 && c == 3)             g.den_owner[s] = RED;

        if (g.trap_owner[s] >= 0) g.trap_mask[g.trap_owner[s]] |= 1ULL << s;
        if (g.den_owner[s] >= 0)  g.den_mask[g.den_owner[s]] |= 1ULL << s;

        const int dr[4] = {-1, 1, 0, 0};
        const int dc[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; ++d) {
            const int nr = r + dr[d], nc = c + dc[d];
            g.neighbor[s][d] = (nr < 0 || nr >= HEIGHT || nc < 0 || nc >= WIDTH)
                                   ? -1
                                   : sq_index(nr, nc);
        }
    }

    // The masks for the water rules.  For every direction: the sources whose
    // destination is on the river, and the sources whose river-ness changes.
    for (int s = 0; s < NUM_SQ; ++s) {
        for (int d = 0; d < 4; ++d) {
            const int to = g.neighbor[s][d];
            if (to < 0) continue;
            if (g.river[to]) g.river_dst[d] |= 1ULL << s;
            if (g.river[s] != g.river[to]) g.river_cross[d] |= 1ULL << s;
        }
    }

    // The jumps: horizontal along the river, vertical across it.
    for (int s = 0; s < NUM_SQ; ++s) g.jump_count[s] = 0;

    for (int r = 3; r <= 5; ++r) {
        for (int c : {0, 3, 6}) {
            const int from = sq_index(r, c);
            const int targets[2] = {c == 0 ? 3 : (c == 3 ? 6 : 3), c == 3 ? 0 : -1};
            for (int t = 0; t < 2; ++t) {
                const int to_c = targets[t];
                if (to_c < 0) continue;
                const int step = (to_c > c) ? 1 : -1;
                JumpGeom j;
                j.to = uint8_t(sq_index(r, to_c));
                j.count = 2;
                j.path[0] = uint8_t(sq_index(r, c + step));
                j.path[1] = uint8_t(sq_index(r, c + 2 * step));
                g.jumps[from][g.jump_count[from]++] = j;
            }
        }
    }
    for (int r = 2; r <= 8; r += 4) {
        for (int c : {1, 2, 4, 5}) {
            const int from = sq_index(r, c);
            const int to_r = 8 - r;
            const int dir = (to_r > r) ? 1 : -1;
            JumpGeom j;
            j.to = uint8_t(sq_index(to_r, c));
            j.count = 3;
            j.path[0] = uint8_t(sq_index(r + dir, c));
            j.path[1] = uint8_t(sq_index(r + 2 * dir, c));
            j.path[2] = uint8_t(sq_index(r + 3 * dir, c));
            g.jumps[from][g.jump_count[from]++] = j;
        }
    }

    return g;
}

// ---------------------------------------------------------------------------
// Loading the network
//
// The same file the engine reads, in the same format: the header, the
// feature transformer (transposed), the two hidden layers and the output
// layer.  The weights are copied out unchanged, so the GPU's forward pass
// works on exactly the numbers the engine works on.
// ---------------------------------------------------------------------------

bool load_net(const char* path, Net& net) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    auto read_raw = [&](void* dst, size_t n) {
        return std::fread(dst, 1, n, f) == n;
    };
    auto read_u32 = [&](uint32_t& v) { return read_raw(&v, 4); };

    char magic[8];
    if (!read_raw(magic, 8)) { std::fclose(f); return false; }
    if (std::memcmp(magic, "JNNUE\0\0\0", 8) != 0) { std::fclose(f); return false; }

    uint32_t version = 0, buckets = 0, squares = 0, layers = 0;
    if (!read_u32(version) || !read_u32(net.ft_hidden) || !read_u32(net.hidden) ||
        !read_u32(layers) || !read_u32(buckets) || !read_u32(squares)) {
        std::fclose(f);
        return false;
    }
    if (version != 3 || buckets != 16 || squares != NUM_SQ || layers != 2) {
        std::fclose(f);
        return false;
    }
    if (net.ft_hidden > MAX_FT || net.hidden > MAX_HIDDEN) {
        std::fclose(f);
        return false;
    }

    const int FH = int(net.ft_hidden);
    const int H = int(net.hidden);
    bool ok = true;
    ok = ok && read_raw(net.ft_w, size_t(FT_INPUT) * FH);
    ok = ok && read_raw(net.ft_scale, size_t(FH) * 4);
    ok = ok && read_raw(net.ft_bias, size_t(FH) * 4);
    ok = ok && read_raw(net.h_w[0], size_t(H) * FH);
    ok = ok && read_raw(net.h_scale[0], size_t(H) * 4);
    ok = ok && read_raw(net.h_bias[0], size_t(H) * 4);
    ok = ok && read_raw(net.h_w[1], size_t(H) * H);
    ok = ok && read_raw(net.h_scale[1], size_t(H) * 4);
    ok = ok && read_raw(net.h_bias[1], size_t(H) * 4);
    ok = ok && read_raw(net.out_w, size_t(2) * H);
    ok = ok && read_raw(&net.out_scale, 4);
    ok = ok && read_raw(&net.out_bias, 4);
    std::fclose(f);
    return ok;
}

}  // namespace jungle_gpu
