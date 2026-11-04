// jungle_gpu.h -- Shared types for the GPU self-play generator.
//
// The generator runs the MCTS teacher entirely on the GPU: one thread per
// game, the whole tree in device memory, and the network's leaf evaluations
// batched across every game that is in flight.  The code that decides what is
// legal and what the network says lives in jungle_gpu.cuh, which compiles
// both as CUDA and as plain C++ -- the plain C++ build is the one the test
// suite runs on a machine without a GPU, and it exercises exactly the same
// rules, the same move generator and the same network arithmetic.
//
// The board is 9 rows of 7 columns, 63 squares, numbered row * 7 + column.
// A piece code is 1..8 for Red and 9..16 for Black; 0 is an empty square.
// A square index fits in a 64-bit mask, so the move generator is written
// with bitboards: every legal move of every piece of a type is one shift and
// a few masks, and the same code runs on a thread for one game or on a
// thousand threads for a thousand games.
#pragma once

#include <cstdint>
#include <cstring>

// The device and host annotations.  Everything in the portable core is
// written so that it compiles unchanged for both.
#if defined(__CUDACC__)
#define JGPU_DEVICE __device__
#define JGPU_HOST __host__
#else
#define JGPU_DEVICE
#define JGPU_HOST
#endif

namespace jungle_gpu {

constexpr int WIDTH = 7;
constexpr int HEIGHT = 9;
constexpr int NUM_SQ = WIDTH * HEIGHT;

constexpr int RED = 0;
constexpr int BLACK = 1;

// Piece types, matching engine/src/jungle.hpp.
constexpr int RAT = 1, CAT = 2, DOG = 3, WOLF = 4;
constexpr int LEOPARD = 5, TIGER = 6, LION = 7, ELEPHANT = 8;

// The status of a tree node, matching engine/src/mcts.hpp.
constexpr int ST_ONGOING = 0, ST_WIN = 1, ST_LOSS = -1, ST_DRAW = 2;

// Why a game ended, matching engine/tools/selfplay.cpp.
constexpr int END_DEN = 0, END_NO_MOVES = 1, END_REPETITION = 2, END_LENGTH = 3;

inline int sq_index(int r, int c) { return r * WIDTH + c; }
inline int sq_row(int s) { return s / WIDTH; }
inline int sq_col(int s) { return s % WIDTH; }
inline int piece_type(int p) { return p <= 8 ? p : p - 8; }
inline int piece_color(int p) { return p == 0 ? -1 : (p <= 8 ? RED : BLACK); }
inline int make_piece(int color, int type) { return color == RED ? type : type + 8; }

// ---------------------------------------------------------------------------
// The position key
// ---------------------------------------------------------------------------

// The Zobrist constants.  The host keeps a copy and the device keeps a copy
// of the same table, filled in from the host before the kernel starts, so a
// position hashes to the same value in both.
struct ZobristTable {
    uint64_t piece[17][NUM_SQ];
    uint64_t side;
};

// The host's copy, built once.
const ZobristTable& host_zobrist();

// ---------------------------------------------------------------------------
// The board
// ---------------------------------------------------------------------------

struct Board {
    uint8_t  sq[NUM_SQ];
    uint8_t  stm = RED;
    uint16_t ply = 0;
    uint64_t key = 0;

    JGPU_DEVICE void clear() { memset(sq, 0, sizeof(sq)); }

    JGPU_DEVICE void setup_start() {
        clear();
        sq[sq_index(6, 0)] = make_piece(RED, ELEPHANT);
        sq[sq_index(8, 6)] = make_piece(RED, LION);
        sq[sq_index(8, 0)] = make_piece(RED, TIGER);
        sq[sq_index(6, 4)] = make_piece(RED, LEOPARD);
        sq[sq_index(6, 2)] = make_piece(RED, WOLF);
        sq[sq_index(7, 1)] = make_piece(RED, DOG);
        sq[sq_index(7, 5)] = make_piece(RED, CAT);
        sq[sq_index(6, 6)] = make_piece(RED, RAT);
        sq[sq_index(2, 6)] = make_piece(BLACK, ELEPHANT);
        sq[sq_index(0, 0)] = make_piece(BLACK, LION);
        sq[sq_index(0, 6)] = make_piece(BLACK, TIGER);
        sq[sq_index(2, 2)] = make_piece(BLACK, LEOPARD);
        sq[sq_index(2, 4)] = make_piece(BLACK, WOLF);
        sq[sq_index(1, 5)] = make_piece(BLACK, DOG);
        sq[sq_index(1, 1)] = make_piece(BLACK, CAT);
        sq[sq_index(2, 0)] = make_piece(BLACK, RAT);
        stm = RED;
        ply = 0;
        key = compute_key();
    }

    // The key of the position, and the key of the position after a move.
    // Both are written in jungle_gpu.cuh, where the Zobrist table is
    // reachable from the device as well as from the host.
    JGPU_DEVICE uint64_t compute_key() const;

    // The key of the position after the move from->to, worked out from the
    // current key instead of from the whole board: a move changes two
    // squares and the side to move, so three table lookups are enough.
    JGPU_DEVICE uint64_t key_after_move(int from, int to) const;
};

// ---------------------------------------------------------------------------
// The static geometry of the board
//
// Everything the move generator needs that does not change from position to
// position: which squares are river, which are traps and whose they are,
// which are dens, the four neighbours of every square, the jump geometry,
// and the masks that make the bitboard shifts work.
//
// The masks come in pairs.  `river_mask` has the river squares in it.
// `river_dst[d]` has the *sources* whose destination in direction d is on the
// river, and `river_cross[d]` has the sources whose river-ness changes when
// they move in direction d -- those are the captures the water rule forbids.
// ---------------------------------------------------------------------------

struct JumpGeom {
    uint8_t  to;         // the destination square
    uint8_t  count;      // how many squares the jump passes over
    uint8_t  path[3];    // those squares
};

struct Geometry {
    bool     river[NUM_SQ];
    int8_t   trap_owner[NUM_SQ];   // -1 none, else RED or BLACK
    int8_t   den_owner[NUM_SQ];    // -1 none, else RED or BLACK
    int8_t   neighbor[NUM_SQ][4];  // -1 when off-board

    uint64_t river_mask = 0;
    uint64_t occ_mask[2] = {0, 0};      // every square
    uint64_t trap_mask[2] = {0, 0};     // the traps owned by each side
    uint64_t den_mask[2] = {0, 0};      // the den owned by each side

    // Sources whose destination in direction d is on the river.
    uint64_t river_dst[4] = {0, 0, 0, 0};
    // Sources whose river-ness changes when they move in direction d.
    uint64_t river_cross[4] = {0, 0, 0, 0};

    // The river jump geometry: for every square, the jumps that start there.
    uint8_t  jump_count[NUM_SQ] = {0};
    JumpGeom jumps[NUM_SQ][3];
};

// Direction order, matching engine/src/jungle.hpp: up, down, left, right.
constexpr int DIR_UP = 0, DIR_DOWN = 1, DIR_LEFT = 2, DIR_RIGHT = 3;

// Build the geometry.  This is the host's job; the device gets a copy.
Geometry build_geometry();

// ---------------------------------------------------------------------------
// The network
//
// The weights are the same int8 values the engine uses, so the GPU's forward
// pass can be made bit-identical to the engine's: the dot products are sums
// of products of small integers, which are exact in fp32, and the clipping
// and rounding steps are the same ones.
// ---------------------------------------------------------------------------

constexpr int FT_INPUT = NUM_SQ * 16;   // 8 own types + 8 their types
constexpr int MAX_FT = 512;
constexpr int MAX_HIDDEN = 512;

struct Net {
    uint32_t ft_hidden = 0;
    uint32_t hidden = 0;
    int8_t  ft_w[FT_INPUT * MAX_FT];     // transposed: [feature][ft_hidden]
    float  ft_scale[MAX_FT];
    float  ft_bias[MAX_FT];
    int8_t  h_w[2][MAX_HIDDEN * MAX_FT];
    float  h_scale[2][MAX_HIDDEN];
    float  h_bias[2][MAX_HIDDEN];
    int8_t  out_w[2 * MAX_HIDDEN];
    float  out_scale = 0.f;
    float  out_bias = 0.f;

    // The feature transformer's bias has to be added after the accumulation,
    // so the accumulator is kept per hidden unit.
    int32_t acc[MAX_FT * 2];   // scratch, one per perspective
};

// Read a network from the engine's binary format.  The weights are copied
// out unchanged, so the GPU works on exactly the numbers the engine works
// on.
bool load_net(const char* path, Net& net);

// The feature index of a piece on a square, seen from `own`'s perspective.
// Matches feature_index() in engine/src/nnue.hpp.
inline int feature_index(int sq, int piece, int own) {
    const int t = piece_type(piece) - 1;
    const bool own_piece = (piece_color(piece) == own);
    const int r = sq_row(sq), c = sq_col(sq);
    const int s = (own == RED) ? sq : (HEIGHT - 1 - r) * WIDTH + c;
    const int bucket = own_piece ? t : (8 + t);
    return s * 16 + bucket;
}

}  // namespace jungle_gpu
