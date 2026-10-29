// jungle_api.cpp -- Flat C API for the engine.
//
// The interface mirrors the one used by the original project's
// `backend/model6.py` so the new engine can be dropped into the existing
// Flask application without changing the Python side.
//
// Board encoding (63 bytes, row-major, 9 rows x 7 columns):
//   0        empty
//   1..8     Red piece of type 1..8 (Rat .. Elephant)
//   9..16    Black piece of type 1..8
#include <cstring>
#include <string>

#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/nnue.hpp"
#include "../src/search.hpp"

using namespace jungle;

extern "C" {

// Forward declaration: the plain entry point is a wrapper around the
// version that can rule moves out.
int jungle_get_best_move_nnue_ex(const char* board_flat, int turn, int move_count, int time_ms,
                                 const char* nnue_path, const int* banned, int banned_cnt,
                                 int* out_move);

// Returns the number of legal moves, writing each as (from, to) pairs.
int jungle_get_valid_moves(const char* board_flat, int turn, int* out_moves) {
    Position pos;
    std::memcpy(pos.board, board_flat, NUM_SQ);
    pos.stm = uint8_t(turn);
    pos.key = pos.compute_key();
    Move moves[128];
    const int n = generate_moves(pos, moves);
    for (int i = 0; i < n; ++i) {
        out_moves[2 * i]     = moves[i].from;
        out_moves[2 * i + 1] = moves[i].to;
    }
    return n;
}

// out_move = {from, to, score, nodes}
int jungle_get_best_move(const char* board_flat, int turn, int move_count, int time_ms,
                          int* out_move) {
    Position pos;
    std::memcpy(pos.board, board_flat, NUM_SQ);
    pos.stm = uint8_t(turn);
    pos.ply = uint16_t(move_count);
    pos.key = pos.compute_key();

    Evaluator ev;                 // handcrafted evaluation
    Searcher searcher(ev, 32);
    std::vector<uint64_t> history;
    history.push_back(pos.key);

    SearchLimits limits;
    limits.time_limit = double(time_ms) / 1000.0;
    limits.max_depth = 64;
    SearchResult r = searcher.search(pos, limits, history);

    if (r.move.from == 0xFFFF) return 0;
    out_move[0] = r.move.from;
    out_move[1] = r.move.to;
    out_move[2] = r.score;
    out_move[3] = int(r.nodes);
    return 1;
}

// Same, but using an NNUE network loaded from `nnue_path`.
// If `nnue_path` is NULL the handcrafted evaluation is used instead.
int jungle_get_best_move_nnue(const char* board_flat, int turn, int move_count, int time_ms,
                              const char* nnue_path, int* out_move) {
    return jungle_get_best_move_nnue_ex(board_flat, turn, move_count, time_ms,
                                        nnue_path, nullptr, 0, out_move);
}

// The variant the web front end uses.
//
// `banned` is a flat list of (from, to) pairs the search must not play:
// `banned_cnt` is the number of *pairs*, not the number of ints.  The front
// end knows the history of the game and can rule out a move that would
// repeat a position; the engine cannot know that on its own.
//
// out_move = {from, to, score, nodes}
int jungle_get_best_move_nnue_ex(const char* board_flat, int turn, int move_count, int time_ms,
                                 const char* nnue_path, const int* banned, int banned_cnt,
                                 int* out_move) {
    Position pos;
    std::memcpy(pos.board, board_flat, NUM_SQ);
    pos.stm = uint8_t(turn);
    pos.ply = uint16_t(move_count);
    pos.key = pos.compute_key();

    Evaluator ev;
    ev.value_scale = 10.0f;
    if (nnue_path && *nnue_path) {
        if (!nnue_load(nnue_path, ev.net)) return 0;
    }
    Searcher searcher(ev, 32);
    std::vector<uint64_t> history;
    history.push_back(pos.key);

    std::vector<Move> banned_moves;
    for (int i = 0; i < banned_cnt; ++i) {
        banned_moves.push_back(Move{uint16_t(banned[2 * i]), uint16_t(banned[2 * i + 1])});
    }

    SearchLimits limits;
    limits.time_limit = double(time_ms) / 1000.0;
    limits.max_depth = 64;
    limits.banned = banned_moves.empty() ? nullptr : &banned_moves;
    SearchResult r = searcher.search(pos, limits, history);

    if (r.move.from == 0xFFFF) return 0;
    out_move[0] = r.move.from;
    out_move[1] = r.move.to;
    out_move[2] = r.score;
    out_move[3] = int(r.nodes);
    return 1;
}

// Evaluate a position with the handcrafted evaluation (positive = good for
// the side to move).  Used by the web UI's analysis bar.
int jungle_evaluate(const char* board_flat, int turn) {
    Position pos;
    std::memcpy(pos.board, board_flat, NUM_SQ);
    pos.stm = uint8_t(turn);
    return evaluate(pos);
}

const char* jungle_version() { return "jungle-nnue 0.1.0"; }

}  // extern "C"
