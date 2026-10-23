// search.hpp -- Iterative-deepening alpha-beta search with a transposition
// table, move ordering and repetition detection.
//
// The search is evaluation-agnostic: it can be driven either by the
// handcrafted evaluation or by the NNUE network, which lets us run a
// controlled "same search, different evaluation" ablation.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "eval.hpp"
#include "jungle.hpp"
#include "nnue.hpp"

namespace jungle {

constexpr int MAX_PLY = 96;

enum Bound : uint8_t { BOUND_EXACT = 0, BOUND_LOWER = 1, BOUND_UPPER = 2 };

struct TTEntry {
    uint64_t key    = 0;
    int32_t  score  = 0;
    int16_t  depth  = -1;
    uint8_t  bound  = BOUND_EXACT;
    uint16_t move   = 0xFFFF;
    uint8_t  valid  = 0;
};

struct SearchLimits {
    int      max_depth = 64;
    double   time_limit = 0.0;    // seconds; 0 => unlimited
    uint64_t node_limit = 0;      // 0 => unlimited
};

struct SearchResult {
    Move   move;
    int    score = 0;
    int    depth = 0;
    uint64_t nodes = 0;
    double time = 0.0;
};

class Searcher {
public:
    explicit Searcher(const Evaluator& ev, size_t tt_mb = 64) : ev_(ev) {
        const size_t entries = (tt_mb * 1024 * 1024) / sizeof(TTEntry);
        tt_.resize(entries);
        tt_mask_ = entries - 1;
    }

    void clear() { std::fill(tt_.begin(), tt_.end(), TTEntry{}); }

    // `history` holds the keys of all positions played so far in the game so
    // that repetition can be detected (three-fold repetition is a draw).
    SearchResult search(Position& pos, const SearchLimits& limits,
                         std::vector<uint64_t>& history) {
        start_time_ = std::chrono::steady_clock::now();
        node_limit_ = limits.node_limit;
        time_limit_ = limits.time_limit;
        nodes_ = 0;
        aborted_ = false;
        history_ = &history;

        SearchResult best;
        best.move = Move{0xFFFF, 0xFFFF};

        int alpha = -MATE_SCORE - 1, beta = MATE_SCORE + 1;
        Move last_best{0xFFFF, 0xFFFF};

        for (int depth = 1; depth <= limits.max_depth; ++depth) {
            const int prev_alpha = alpha;
            int score = alphabeta(pos, depth, alpha, beta, 0);
            if (aborted_) break;

            if (score > alpha) {
                alpha = score;
                last_best = root_best_;
            }
            best.score = score;
            best.depth = depth;
            if (last_best.from != 0xFFFF) best.move = last_best;

            if (std::abs(score) >= MATE_SCORE - 1) break;   // mate found
            if (limits.time_limit > 0.0 && out_of_time()) break;
            if (alpha >= beta) { beta = MATE_SCORE + 1; alpha = prev_alpha; }
        }

        if (best.move.from == 0xFFFF) {
            // Fall back to the first legal move.
            Move buf[128];
            const int n = generate_moves(pos, buf);
            if (n > 0) best.move = buf[0];
        }
        best.nodes = nodes_;
        best.time = elapsed();
        return best;
    }

private:
    // --- TT --------------------------------------------------------------
    TTEntry& tt_probe(uint64_t key) {
        TTEntry& e = tt_[key & tt_mask_];
        if (e.valid && e.key == key) return e;
        return e;
    }

    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_)
            .count();
    }
    bool out_of_time() const {
        if (time_limit_ <= 0.0) return false;
        return elapsed() >= time_limit_;
    }
    bool out_of_nodes() const {
        if (node_limit_ == 0) return false;
        return nodes_ >= node_limit_;
    }

    // --- Ordering --------------------------------------------------------
    struct ScoredMove {
        Move  move;
        int   score;
    };

    static int move_priority(const Position& pos, Move m, int tt_from, int tt_to) {
        if (m.from == tt_from && m.to == tt_to) return 1'000'000;
        const int captured = pos.board[m.to];
        const int mover    = pos.board[m.from];
        if (captured) {
            int victim = piece_type(captured);
            if (terrain().trap_owner[m.to] == piece_color(mover)) victim = 0;
            int attacker = piece_type(mover);
            // Capturing with a weak piece onto a trap is very good.
            return 500'000 + victim * 1000 - attacker * 10;
        }
        if (terrain().den_owner[m.to] >= 0) return 900'000;   // den entry
        return 1000 - sq_row(m.to) * 10;
    }

    void order_moves(const Position& pos, Move* moves, int n, int tt_from, int tt_to,
                     ScoredMove* out) {
        for (int i = 0; i < n; ++i) {
            out[i].move = moves[i];
            out[i].score = move_priority(pos, moves[i], tt_from, tt_to);
        }
        std::sort(out, out + n, [](const ScoredMove& a, const ScoredMove& b) {
            return a.score > b.score;
        });
    }

    // --- Core ------------------------------------------------------------
    int alphabeta(Position& pos, int depth, int alpha, int beta, int ply) {
        ++nodes_;
        if ((nodes_ & 1023) == 0) {
            if (out_of_time() || out_of_nodes()) {
                aborted_ = true;
                return 0;
            }
        }
        if (aborted_) return 0;

        const bool is_pv = (beta - alpha) > 1;
        const int alpha_orig = alpha;

        // Draw by repetition?
        if (ply > 0) {
            int repeats = 0;
            for (int i = int(history_->size()) - 1; i >= 0; --i) {
                if ((*history_)[i] == pos.key) {
                    if (++repeats >= 2) return 0;
                }
            }
        }

        TTEntry& tte = tt_probe(pos.key);
        int tt_from = 0xFFFF, tt_to = 0xFFFF;
        if (tte.valid && tte.key == pos.key) {
            tt_from = tte.move >> 6;
            tt_to = tte.move & 63;
            if (tte.depth >= depth && !is_pv) {
                if (tte.bound == BOUND_EXACT) return tte.score;
                if (tte.bound == BOUND_LOWER && tte.score >= beta) return tte.score;
                if (tte.bound == BOUND_UPPER && tte.score <= alpha) return tte.score;
            }
        }

        if (depth <= 0) {
            int v = quiesce(pos, alpha, beta, ply);
            return v;
        }

        Move moves[128];
        const int n = generate_moves(pos, moves);
        if (n == 0) return -MATE_SCORE + ply;   // no legal move: loss

        ScoredMove ordered[128];
        order_moves(pos, moves, n, tt_from, tt_to, ordered);

        int best = -MATE_SCORE - 1;
        Move best_move{0xFFFF, 0xFFFF};
        Move tried[128];
        int tried_n = 0;

        for (int i = 0; i < n; ++i) {
            const Move m = ordered[i].move;

            // Winning move straight into the opponent's den?
            if (terrain().den_owner[m.to] == opposite_color(pos.stm)) {
                best = MATE_SCORE - ply - 1;
                best_move = m;
                break;
            }

            const int captured = pos.board[m.to];
            make_move(pos, m);
            history_->push_back(pos.key);

            int score;
            if (i == 0) {
                score = -alphabeta(pos, depth - 1, -beta, -alpha, ply + 1);
            } else {
                // Late move reductions for quiet, late moves.
                int reduction = 0;
                if (depth >= 3 && !captured && ply > 0 && i >= 4) reduction = 1;
                score = -alphabeta(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
                if (!aborted_ && score > alpha && score < beta)
                    score = -alphabeta(pos, depth - 1, -beta, -alpha, ply + 1);
            }

            history_->pop_back();
            undo_move(pos, m, captured);

            if (aborted_) return 0;

            tried[tried_n++] = m;
            if (score > best) {
                best = score;
                best_move = m;
                if (ply == 0) root_best_ = m;
                if (score > alpha) alpha = score;
                if (alpha >= beta) break;
            }
        }

        tte.key   = pos.key;
        tte.valid = 1;
        tte.depth = int16_t(depth);
        tte.score = best;
        tte.move  = uint16_t((best_move.from << 6) | best_move.to);
        if (best <= alpha_orig) tte.bound = BOUND_UPPER;
        else if (best >= beta)   tte.bound = BOUND_LOWER;
        else                     tte.bound = BOUND_EXACT;
        return best;
    }

    // Simple capture-only extension: keeps searching while captures remain.
    int quiesce(Position& pos, int alpha, int beta, int ply) {
        ++nodes_;
        if ((nodes_ & 1023) == 0 && (out_of_time() || out_of_nodes())) {
            aborted_ = true;
            return 0;
        }
        int stand = ev_.eval(pos);
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;

        Move moves[128];
        const int n = generate_moves(pos, moves);
        ScoredMove ordered[128];
        order_moves(pos, moves, n, 0xFFFF, 0xFFFF, ordered);

        for (int i = 0; i < n; ++i) {
            const Move m = ordered[i].move;
            if (!pos.board[m.to]) continue;                 // captures only
            if (terrain().den_owner[m.to] == opposite_color(pos.stm))
                return MATE_SCORE - ply - 1;
            const int captured = pos.board[m.to];
            make_move(pos, m);
            history_->push_back(pos.key);
            const int score = -quiesce(pos, -beta, -alpha, ply + 1);
            history_->pop_back();
            undo_move(pos, m, captured);
            if (aborted_) return 0;
            if (score >= beta) return score;
            if (score > alpha) alpha = score;
        }
        return alpha;
    }

    Evaluator ev_;
    std::vector<TTEntry> tt_;
    size_t tt_mask_ = 0;
    std::chrono::steady_clock::time_point start_time_;
    uint64_t nodes_ = 0;
    uint64_t node_limit_ = 0;
    double   time_limit_ = 0.0;
    bool     aborted_ = false;
    Move     root_best_{0xFFFF, 0xFFFF};
    std::vector<uint64_t>* history_ = nullptr;
};

}  // namespace jungle
