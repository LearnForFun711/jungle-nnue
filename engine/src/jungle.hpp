// jungle.hpp -- Core rules, board representation and move generation for
// Jungle Chess (斗兽棋 / 鬥獸棋).
//
// The rule set implemented here is the standard 7x9 variant and is byte-for-byte
// compatible with the reference implementation in `backend/engine.py` of the
// original project:
//
//   * Board is 9 rows x 7 columns. Row 0 is Black's home edge, row 8 is Red's.
//   * Red moves first.
//   * Dens: Red (8,3), Black (0,3). A piece entering the opponent's den wins
//     immediately. A piece may never enter its own den.
//   * Traps: Black (0,2) (0,4) (1,3), Red (8,2) (8,4) (7,3). A piece standing
//     on the *opponent's* trap loses its rank and may be captured by anything,
//     including a Rat.
//   * River: rows 3..5 and columns 1,2,4,5. Only the Rat may enter it.
//   * Rat in water cannot capture a piece on land; a piece on land cannot
//     capture a Rat in water. (Movement between water and land is legal --
//     only *captures* are restricted.)
//   * Tiger and Lion jump over the river (3 squares horizontally, 4
//     vertically) provided the intervening squares are empty.
//   * Rank order Rat < Cat < Dog < Wolf < Leopard < Tiger < Lion < Elephant,
//     except that a Rat captures an Elephant and an Elephant never captures a
//     Rat.
//   * A player with no legal move loses. Three-fold repetition is a draw.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace jungle {

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

constexpr int WIDTH  = 7;
constexpr int HEIGHT = 9;
constexpr int NUM_SQ = WIDTH * HEIGHT;  // 63

constexpr int MATE_SCORE = 30000;

constexpr int RED   = 0;
constexpr int BLACK = 1;

// Piece types. Type 1 = Rat ... type 8 = Elephant.
enum PieceType : int {
    RAT      = 1,
    CAT      = 2,
    DOG      = 3,
    WOLF     = 4,
    LEOPARD  = 5,
    TIGER    = 6,
    LION     = 7,
    ELEPHANT = 8
};

// Board encoding: 0 = empty, 1..8 = Red piece of that type, 9..16 = Black piece.
inline int sq_index(int r, int c) { return r * WIDTH + c; }
inline int sq_row(int s)          { return s / WIDTH; }
inline int sq_col(int s)          { return s % WIDTH; }
inline int piece_type(int p)      { return p <= 8 ? p : p - 8; }  // 0 when empty
inline int piece_color(int p)     { return p == 0 ? -1 : (p <= 8 ? RED : BLACK); }
inline int make_piece(int color, int type) { return color == RED ? type : type + 8; }
inline int opposite_color(int c)  { return c ^ 1; }

// ---------------------------------------------------------------------------
// Static terrain tables (built once)
// ---------------------------------------------------------------------------

struct Terrain {
    bool river[NUM_SQ];
    int  trap_owner[NUM_SQ];   // -1 none, RED or BLACK
    int  den_owner[NUM_SQ];    // -1 none, RED or BLACK
    int  neighbor[NUM_SQ][4];  // orthogonal neighbours, -1 when off-board
    // Legal river-jump geometry: for a square s, the list of jump targets.
    // Each target comes with the squares that must be empty to pass over.
    struct Jump {
        int  to;
        int  path[3];
        int  path_len;
    };
    std::vector<Jump> jumps[NUM_SQ];

    Terrain() {
        for (int s = 0; s < NUM_SQ; ++s) {
            const int r = s / WIDTH, c = s % WIDTH;
            river[s] = (r >= 3 && r <= 5) && (c == 1 || c == 2 || c == 4 || c == 5);

            trap_owner[s] = -1;
            den_owner[s]  = -1;
            if (r == 0 && (c == 2 || c == 4)) trap_owner[s] = BLACK;
            if (r == 1 && c == 3)             trap_owner[s] = BLACK;
            if (r == 8 && (c == 2 || c == 4)) trap_owner[s] = RED;
            if (r == 7 && c == 3)             trap_owner[s] = RED;
            if (r == 0 && c == 3)             den_owner[s]  = BLACK;
            if (r == 8 && c == 3)             den_owner[s]  = RED;

            const int dr[4] = {-1, 1, 0, 0};
            const int dc[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; ++d) {
                const int nr = r + dr[d], nc = c + dc[d];
                neighbor[s][d] = (nr < 0 || nr >= HEIGHT || nc < 0 || nc >= WIDTH)
                                     ? -1
                                     : sq_index(nr, nc);
            }
        }

        // Horizontal jumps: rows 3..5, columns {0,3,6}.  Column 0 jumps to 3,
        // column 3 jumps to 0 or 6, column 6 jumps to 3 -- always crossing the
        // two river squares in between.
        for (int r = 3; r <= 5; ++r) {
            for (int c : {0, 3, 6}) {
                const int from = sq_index(r, c);
                const int targets[2] = {c == 0 ? 3 : (c == 3 ? 6 : 3), c == 3 ? 0 : -1};
                for (int t = 0; t < 2; ++t) {
                    const int to_c = targets[t];
                    if (to_c < 0) continue;
                    const int step = (to_c > c) ? 1 : -1;
                    Jump j;
                    j.to       = sq_index(r, to_c);
                    j.path[0]  = sq_index(r, c + step);
                    j.path[1]  = sq_index(r, c + 2 * step);
                    j.path_len = 2;
                    jumps[from].push_back(j);
                }
            }
        }
        // Vertical jumps: rows {2,6}, columns 1,2,4,5.
        for (int r = 2; r <= 8; r += 4) {
            for (int c : {1, 2, 4, 5}) {
                const int from = sq_index(r, c);
                const int to_r = 8 - r;
                Jump j;
                j.to       = sq_index(to_r, c);
                j.path[0]  = sq_index(r + (to_r > r ? 1 : -1), c);
                j.path[1]  = sq_index(r + (to_r > r ? 2 : -2), c);
                j.path[2]  = sq_index(r + (to_r > r ? 3 : -3), c);
                j.path_len = 3;
                jumps[from].push_back(j);
            }
        }
    }
};

inline const Terrain& terrain() {
    static const Terrain t;
    return t;
}

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------

struct Zobrist {
    uint64_t piece[17][NUM_SQ];
    uint64_t side;

    Zobrist() {
        uint64_t seed = 0x9E3779B97F4A7C15ULL;
        auto next = [&]() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            return seed;
        };
        for (int p = 0; p < 17; ++p)
            for (int s = 0; s < NUM_SQ; ++s) piece[p][s] = next();
        side = next();
    }
};

inline const Zobrist& zobrist() {
    static const Zobrist z;
    return z;
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

struct Move {
    uint16_t from = 0;
    uint16_t to   = 0;

    bool operator==(const Move& o) const { return from == o.from && to == o.to; }
    bool operator!=(const Move& o) const { return !(*this == o); }
    bool operator<(const Move& o) const {
        return from != o.from ? from < o.from : to < o.to;
    }
    uint32_t encoded() const { return (uint32_t(from) << 6) | uint32_t(to); }
    static Move decode(uint32_t v) { return Move{uint16_t(v >> 6), uint16_t(v & 63)}; }
};

struct Position {
    uint8_t  board[NUM_SQ];
    uint8_t  stm = RED;
    uint16_t ply = 0;
    uint64_t key = 0;

    Position() { reset(); }

    void reset() {
        std::memset(board, 0, sizeof(board));
        // Red (bottom, row 8)
        board[sq_index(6, 0)] = make_piece(RED, ELEPHANT);  // r8
        board[sq_index(8, 6)] = make_piece(RED, LION);      // r7
        board[sq_index(8, 0)] = make_piece(RED, TIGER);     // r6
        board[sq_index(6, 4)] = make_piece(RED, LEOPARD);   // r5
        board[sq_index(6, 2)] = make_piece(RED, WOLF);      // r4
        board[sq_index(7, 1)] = make_piece(RED, DOG);       // r3
        board[sq_index(7, 5)] = make_piece(RED, CAT);       // r2
        board[sq_index(6, 6)] = make_piece(RED, RAT);       // r1
        // Black (top, row 0)
        board[sq_index(2, 6)] = make_piece(BLACK, ELEPHANT);  // b8
        board[sq_index(0, 0)] = make_piece(BLACK, LION);      // b7
        board[sq_index(0, 6)] = make_piece(BLACK, TIGER);     // b6
        board[sq_index(2, 2)] = make_piece(BLACK, LEOPARD);   // b5
        board[sq_index(2, 4)] = make_piece(BLACK, WOLF);      // b4
        board[sq_index(1, 5)] = make_piece(BLACK, DOG);       // b3
        board[sq_index(1, 1)] = make_piece(BLACK, CAT);       // b2
        board[sq_index(2, 0)] = make_piece(BLACK, RAT);       // b1
        stm = RED;
        ply = 0;
        key = compute_key();
    }

    uint64_t compute_key() const {
        const Zobrist& z = zobrist();
        uint64_t k = 0;
        for (int s = 0; s < NUM_SQ; ++s)
            if (board[s]) k ^= z.piece[board[s]][s];
        if (stm == BLACK) k ^= z.side;
        return k;
    }

    int piece_at(int r, int c) const { return board[sq_index(r, c)]; }

    // FEN-ish string used by the tooling (matches backend/engine.py layout).
    std::string to_fen() const {
        static const char map[9] = {'?', 'R', 'C', 'D', 'W', 'P', 'T', 'L', 'E'};
        std::string out;
        for (int r = 0; r < HEIGHT; ++r) {
            int empty = 0;
            for (int c = 0; c < WIDTH; ++c) {
                const int p = board[sq_index(r, c)];
                if (!p) {
                    ++empty;
                    continue;
                }
                if (empty) { out += char('0' + empty); empty = 0; }
                char ch = map[piece_type(p)];
                if (piece_color(p) == BLACK) ch = char(ch - 'A' + 'a');
                out += ch;
            }
            if (empty) out += char('0' + empty);
            if (r != HEIGHT - 1) out += '/';
        }
        out += ' ';
        out += (stm == RED ? 'r' : 'b');
        return out;
    }

    // Parse the FEN produced above (and the reference implementation's FEN).
    bool set_fen(const std::string& fen) {
        std::memset(board, 0, sizeof(board));
        std::string main_part = fen.substr(0, fen.find('|'));
        std::istringstream is(main_part);
        std::string rows, side;
        is >> rows >> side;
        if (rows.empty()) return false;
        int r = 0, c = 0;
        for (char ch : rows) {
            if (ch == '/') { ++r; c = 0; continue; }
            if (ch >= '0' && ch <= '9') { c += ch - '0'; continue; }
            int type = 0, color = RED;
            switch (ch) {
                case 'R': type = RAT;      break;  case 'r': type = RAT;      color = BLACK; break;
                case 'C': type = CAT;      break;  case 'c': type = CAT;      color = BLACK; break;
                case 'D': type = DOG;      break;  case 'd': type = DOG;      color = BLACK; break;
                case 'W': type = WOLF;     break;  case 'w': type = WOLF;     color = BLACK; break;
                case 'P': type = LEOPARD;  break;  case 'p': type = LEOPARD;  color = BLACK; break;
                case 'T': type = TIGER;    break;  case 't': type = TIGER;    color = BLACK; break;
                case 'L': type = LION;     break;  case 'l': type = LION;     color = BLACK; break;
                case 'E': type = ELEPHANT; break;  case 'e': type = ELEPHANT; color = BLACK; break;
                default: return false;
            }
            if (r >= HEIGHT || c >= WIDTH) return false;
            board[sq_index(r, c)] = make_piece(color, type);
            ++c;
        }
        stm = (side == "b" || side == "B") ? BLACK : RED;
        key = compute_key();
        return true;
    }
};

// ---------------------------------------------------------------------------
// Move legality
// ---------------------------------------------------------------------------

// Can `p` (owned by `color`) capture the piece standing on `to`?
inline bool capture_is_legal(const Position& pos, int from, int to, int p, int color) {
    const int target = pos.board[to];
    if (target == 0) return true;
    if (piece_color(target) == color) return false;

    int t_rank = piece_type(target);
    if (terrain().trap_owner[to] == color) t_rank = 0;  // trapped pieces lose their rank

    // Water restrictions apply to captures only.
    const bool from_river = terrain().river[from];
    const bool to_river   = terrain().river[to];
    if (from_river && !to_river) return false;   // rat in water cannot take land
    if (!from_river && to_river) return false;   // land cannot take a rat in water

    const int rank = piece_type(p);
    if (rank == RAT && t_rank == ELEPHANT) return true;
    if (rank == ELEPHANT && t_rank == RAT) return false;
    return rank >= t_rank;
}

// A one-square move.
inline bool step_is_legal(const Position& pos, int from, int to, int p, int color) {
    const int target = pos.board[to];
    if (target != 0 && piece_color(target) == color) return false;
    if (terrain().den_owner[to] == color) return false;        // cannot enter own den
    if (terrain().river[to] && piece_type(p) != RAT) return false;  // only rats swim
    return capture_is_legal(pos, from, to, p, color);
}

// A river jump (Tiger / Lion).
inline bool jump_is_legal(const Position& pos, const Terrain::Jump& j, int p, int color) {
    for (int i = 0; i < j.path_len; ++i)
        if (pos.board[j.path[i]] != 0) return false;   // blocked
    const int target = pos.board[j.to];
    if (target != 0 && piece_color(target) == color) return false;
    if (terrain().den_owner[j.to] == color) return false;
    // Jumps always start and end on land, so the water rules cannot trigger.
    int t_rank = piece_type(target);
    if (target != 0 && terrain().trap_owner[j.to] == color) t_rank = 0;
    const int rank = piece_type(p);
    if (target == 0) return true;
    if (rank == RAT && t_rank == ELEPHANT) return true;
    if (rank == ELEPHANT && t_rank == RAT) return false;
    return rank >= t_rank;
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------

inline bool move_is_legal(const Position& pos, Move m, int color) {
    const int from = m.from, to = m.to;
    if (from < 0 || from >= NUM_SQ || to < 0 || to >= NUM_SQ) return false;
    const int p = pos.board[from];
    if (!p || piece_color(p) != color) return false;
    const int type = piece_type(p);

    const int dr = sq_row(to) - sq_row(from);
    const int dc = sq_col(to) - sq_col(from);

    if (type == TIGER || type == LION) {
        for (const auto& j : terrain().jumps[from])
            if (j.to == to) return jump_is_legal(pos, j, p, color);
    }
    if (dr + dc < 0) return false;
    if (std::abs(dr) + std::abs(dc) != 1) return false;
    return step_is_legal(pos, from, to, p, color);
}

inline int generate_moves(const Position& pos, Move* out) {
    const Terrain& T = terrain();
    const int color = pos.stm;
    int n = 0;
    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = pos.board[s];
        if (!p || piece_color(p) != color) continue;
        const int type = piece_type(p);
        for (int d = 0; d < 4; ++d) {
            const int to = T.neighbor[s][d];
            if (to < 0) continue;
            if (!step_is_legal(pos, s, to, p, color)) continue;
            out[n].from = uint16_t(s);
            out[n].to   = uint16_t(to);
            ++n;
        }
        if (type == TIGER || type == LION) {
            for (const auto& j : T.jumps[s]) {
                if (!jump_is_legal(pos, j, p, color)) continue;
                out[n].from = uint16_t(s);
                out[n].to   = uint16_t(j.to);
                ++n;
            }
        }
    }
    return n;
}

inline bool has_legal_moves(const Position& pos) {
    Move buf[128];
    return generate_moves(pos, buf) > 0;
}

// ---------------------------------------------------------------------------
// Applying moves
// ---------------------------------------------------------------------------

inline void make_move(Position& pos, Move m) {
    const Zobrist& z = zobrist();
    const int from = m.from, to = m.to;
    const int p = pos.board[from];
    const int captured = pos.board[to];
    if (captured) pos.key ^= z.piece[captured][to];
    pos.key ^= z.piece[p][from];
    pos.board[to] = uint8_t(p);
    pos.board[from] = 0;
    pos.key ^= z.piece[p][to];
    pos.stm = uint8_t(opposite_color(pos.stm));
    pos.key ^= z.side;
    ++pos.ply;
}

inline void undo_move(Position& pos, Move m, int captured) {
    const Zobrist& z = zobrist();
    const int from = m.from, to = m.to;
    const int p = pos.board[to];
    pos.key ^= z.piece[p][to];
    pos.board[from] = uint8_t(p);
    pos.board[to] = uint8_t(captured);
    if (captured) pos.key ^= z.piece[captured][to];
    pos.key ^= z.piece[p][from];
    pos.stm = uint8_t(opposite_color(pos.stm));
    pos.key ^= z.side;
    --pos.ply;
}

// Did `m` capture a piece on `to`?  (helper for make/unmake pairs)
inline int captured_piece(const Position& pos, Move m) { return pos.board[m.to]; }

// Moving onto the opponent's den wins the game immediately.
inline bool is_den_entry(const Position& pos, Move m) {
    return terrain().den_owner[m.to] == opposite_color(pos.stm);
}

inline std::string move_to_uci(const Position& pos, Move m) {
    auto sq_name = [](int s) {
        const int r = sq_row(s), c = sq_col(s);
        std::string out;
        out += char('a' + c);
        out += char('1' + (HEIGHT - 1 - r));
        return out;
    };
    return sq_name(m.from) + sq_name(m.to);
}

inline Move uci_to_move(const Position& pos, const std::string& s) {
    auto parse = [](char f, char t, int& r, int& c) {
        c = f - 'a';
        r = HEIGHT - 1 - (t - '1');
    };
    int fr, fc, tr, tc;
    parse(s[0], s[1], fr, fc);
    parse(s[2], s[3], tr, tc);
    return Move{uint16_t(sq_index(fr, fc)), uint16_t(sq_index(tr, tc))};
}

}  // namespace jungle
