// mcts.hpp -- PUCT Monte-Carlo tree search used as the *teacher*.
//
// The teacher plays the handcrafted evaluation (and optionally the NNUE
// network) and produces, for every position it visits:
//   * a value estimate in [-1,1] from the point of view of the side to move,
//   * a visit distribution over the root moves (policy target).
//
// Positions live inside the tree so the search is self-contained.  Repetition
// detection uses a per-node counter of how many ancestors share the same
// position key, which makes the three-fold repetition test O(1).
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "eval.hpp"
#include "jungle.hpp"
#include "nnue.hpp"

namespace jungle {

struct MctsParams {
    int    simulations     = 800;
    double max_seconds     = 0.0;   // time cap per move (0 => use `simulations`)
    double c_puct          = 1.4;
    double widening_k      = 1.6;   // eligible children = 1 + k*sqrt(visits)
    int    widening_max    = 48;
    double dirichlet_alpha = 0.30;
    double dirichlet_eps   = 0.25;
    double eval_scale      = 2200.0;// handcrafted score -> [-1,1]
    int    rollouts        = 0;     // Monte-Carlo rollouts per fresh leaf
    int    rollout_depth   = 24;
    int    leaf_depth      = 1;     // leaf evaluation: 0 = static eval, >=1 = search
    int    leaf_agg        = 1;     // 0 = max over children, 1 = mean over children
    int    max_ply         = 200;   // draw by length
    bool   use_nnue        = false;
    double leaf_scale      = 1.0;  // scales the neural leaf evaluation
};

enum NodeStatus : int8_t { ST_ONGOING = 0, ST_WIN = 1, ST_LOSS = -1, ST_DRAW = 2 };

struct MctsNode {
    uint8_t  board[NUM_SQ];
    uint8_t  stm    = RED;
    uint16_t ply    = 0;
    uint64_t key    = 0;
    uint8_t  rep    = 0;        // ancestors sharing this position's key
    int8_t   status = ST_ONGOING;
    uint32_t move_list = 0;     // index into move_lists_
    uint32_t children_start = 0;// offset into child_refs_
    uint16_t num_children = 0;  // children created so far
    uint16_t num_moves    = 0;  // total legal moves
    uint16_t visits = 0;
    double   value_sum = 0.0;   // values from *this node's* stm perspective
    double   leaf_value = 0.0;  // static evaluation computed at expansion
    float    prior = 0.0;       // prior of the move leading *to* this node
    uint16_t from = 0xFFFF, to = 0xFFFF;
};

struct RootVisit {
    Move   move;
    int    visits;
    double value;    // average value from the root's stm perspective
};

struct MctsResult {
    Move        best{0xFFFF, 0xFFFF};
    double      value = 0.0;
    int         simulations = 0;
    std::vector<RootVisit> visits;
};

class Mcts {
public:
    Mcts(const MctsParams& params, Evaluator* ev) : p_(params), ev_(ev) {}

    void reset() { nodes_.clear(); move_lists_.clear(); child_refs_.clear(); }

    MctsResult search(const Position& root, std::mt19937_64& rng) {
        nodes_.clear();
        move_lists_.clear();
        child_refs_.clear();
        nodes_.reserve(16384);
        move_lists_.reserve(16384);
        child_refs_.reserve(65536);

        MctsNode n;
        std::memcpy(n.board, root.board, NUM_SQ);
        n.stm  = root.stm;
        n.ply  = root.ply;
        n.key  = root.key;
        n.rep  = 0;
        n.status = ST_ONGOING;
        nodes_.push_back(n);
        root_ = 0;

        expand_node(root_);
        create_children(root_, /*all=*/true, rng);

        const auto t0 = std::chrono::steady_clock::now();
        int done = 0;
        for (int i = 0; i < p_.simulations; ++i) {
            descend(root_, rng);
            ++done;
            if (p_.max_seconds > 0.0 && (i & 31) == 31) {
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (el >= p_.max_seconds) break;
            }
        }

        MctsResult res;
        res.simulations = done;
        res.value = nodes_[root_].visits ? nodes_[root_].value_sum / nodes_[root_].visits : 0.0;

        const MctsNode& r = nodes_[root_];
        for (uint16_t i = 0; i < r.num_children; ++i) {
            const MctsNode& c = nodes_[child_refs_[r.children_start + i]];
            RootVisit rv;
            rv.move   = Move{c.from, c.to};
            rv.visits = c.visits;
            rv.value  = c.visits ? -c.value_sum / c.visits : 0.0;  // root's perspective
            res.visits.push_back(rv);
        }
        std::sort(res.visits.begin(), res.visits.end(),
                  [](const RootVisit& a, const RootVisit& b) { return a.visits > b.visits; });
        if (!res.visits.empty()) res.best = res.visits.front().move;
        return res;
    }

private:
    // --- helpers ---------------------------------------------------------
    Position node_position(uint32_t idx) const {
        Position pos;
        std::memcpy(pos.board, nodes_[idx].board, NUM_SQ);
        pos.stm = nodes_[idx].stm;
        pos.ply = nodes_[idx].ply;
        pos.key = nodes_[idx].key;
        return pos;
    }

    static float move_prior(const Position& pos, Move m) {
        const int captured = pos.board[m.to];
        const int mover    = pos.board[m.from];
        if (terrain().den_owner[m.to] >= 0) return 60.0f;   // den entry
        if (captured) {
            int victim = piece_type(captured);
            if (terrain().trap_owner[m.to] == piece_color(mover)) victim = 0;
            const int attacker = piece_type(mover);
            return 10.0f + victim * 3.0f - attacker * 0.3f;
        }
        const int d = eval_tables().dist_to_enemy[pos.stm][m.to];
        return 1.0f + (14 - d) * 0.25f;
    }

    // Normalise a search score into [-1,1].
    double normalise(int score) const {
        return std::max(-1.0, std::min(1.0, std::tanh(double(score) / p_.eval_scale)));
    }

    // Leaf evaluation.
    //
    // `leaf_depth` controls how much search is done before the static
    // evaluation is used:
    //   0 -> static evaluation only
    //   1 -> evaluate every child position and keep the best (the opponent
    //        replies with a static evaluation)
    //   2 -> full alpha-beta over the children
    //
    // Depth 1 is the sweet spot: it costs about one move generation per
    // leaf but makes the teacher's values far more informative than the raw
    // static evaluation, which is what the network has to learn from.
    // Base evaluation of a position in [-1,1], from the side to move's point
    // of view.
    //
    // When a network is loaded it *replaces* the handcrafted evaluation, but
    // it does not replace the structure of the leaf search: the one-ply
    // averaging below still runs, so the teacher still gets to look one move
    // ahead and to average away the noise of a single static evaluation.
    // That is the difference between a teacher whose values are worth
    // learning from and one that merely echoes the network it was given.
    double base_eval(const Position& pos) {
        if (p_.use_nnue && ev_ && ev_->uses_nnue()) {
            const float v = float(ev_->eval(pos)) / 1000.0f * float(p_.leaf_scale);
            return double(std::max(-1.0f, std::min(1.0f, v)));
        }
        return normalise(evaluate(pos));
    }

    // The same, but for a position the accumulators already describe.
    double base_eval_cached(const Position& pos) {
        if (p_.use_nnue && ev_ && ev_->uses_nnue()) {
            const float v = float(ev_->eval_cached(pos)) / 1000.0f * float(p_.leaf_scale);
            return double(std::max(-1.0f, std::min(1.0f, v)));
        }
        return normalise(evaluate(pos));
    }

    double leaf_value(const Position& pos) {
        if (p_.rollouts > 0) {
            double sum = 0.0;
            for (int i = 0; i < p_.rollouts; ++i) sum += rollout(pos);
            return std::max(-1.0, std::min(1.0, sum / p_.rollouts));
        }
        if (p_.leaf_depth <= 0)
            return base_eval(pos);

        Move moves[128];
        const int n = generate_moves(pos, moves);
        if (n == 0) return -1.0;

        if (p_.leaf_depth == 1) {
            // Averaging the children's evaluations keeps the value
            // discriminative; taking the maximum of ~30 noisy evaluations
            // would bias every position towards zero.
            const bool neural = p_.use_nnue && ev_ && ev_->uses_nnue();
            double sum = 0.0;
            double best = neural ? -1.0 : double(-MATE_SCORE);
            int count = 0;

            // One refresh for the parent; the children are then reached by
            // moving a single piece, which the feature accumulator can
            // follow incrementally.  Refreshing for every child instead
            // makes the teacher four times slower for the same data.
            if (neural) ev_->refresh(pos);

            for (int i = 0; i < n; ++i) {
                if (terrain().den_owner[moves[i].to] == opposite_color(pos.stm))
                    return 1.0;
                const int moved    = pos.board[moves[i].from];
                const int captured = pos.board[moves[i].to];
                Position child = pos;
                make_move(child, moves[i]);
                if (neural) {
                    ev_->on_move(moves[i].from, moves[i].to, moved, captured, true);
                    // The network's value is already in [-1,1], so it is
                    // aggregated directly and never normalised.
                    const double v = -base_eval_cached(child);
                    ev_->on_move(moves[i].from, moves[i].to, moved, captured, false);
                    sum += v;
                    best = std::max(best, v);
                } else {
                    const int v = -evaluate(child);
                    best = std::max(best, double(v));
                    sum += v;
                }
                ++count;
            }
            if (neural)
                return (p_.leaf_agg == 0) ? best : sum / count;
            const int agg = (p_.leaf_agg == 0) ? int(std::lround(best))
                                               : int(std::lround(sum / count));
            return normalise(agg);
        }
        Position work = pos;
        return normalise(leaf_search(work, p_.leaf_depth, -MATE_SCORE - 1, MATE_SCORE + 1));
    }

    // Small alpha-beta used for the leaf evaluation (no TT, no time limit).
    int leaf_search(Position& pos, int depth, int alpha, int beta) {
        Move moves[128];
        const int n = generate_moves(pos, moves);
        if (n == 0) return -MATE_SCORE;
        if (depth <= 0) return evaluate(pos);

        // Order by a cheap heuristic so that alpha-beta cuts early.
        std::array<int, 128> order{};
        for (int i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.begin() + n, [&](int a, int b) {
            return move_prior(pos, moves[a]) > move_prior(pos, moves[b]);
        });

        int best = -MATE_SCORE;
        for (int i = 0; i < n; ++i) {
            const Move m = moves[order[i]];
            if (terrain().den_owner[m.to] == opposite_color(pos.stm)) return MATE_SCORE;
            const int captured = pos.board[m.to];
            make_move(pos, m);
            const int v = -leaf_search(pos, depth - 1, -beta, -alpha);
            undo_move(pos, m, captured);
            best = std::max(best, v);
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
        return best;
    }

    // Random-ish rollout: +1 (stm wins), -1 (stm loses), 0 (draw).
    double rollout(const Position& start) {
        Position pos = start;
        std::mt19937_64 local(0xC0FFEEULL ^ uint64_t(pos.key));
        for (int ply = 0; ply < p_.rollout_depth; ++ply) {
            Move moves[128];
            const int n = generate_moves(pos, moves);
            if (n == 0) return -1.0;
            Move pick = moves[local() % n];
            for (int i = 0; i < n; ++i) {
                if (pos.board[moves[i].to] && (local() & 3) != 0) { pick = moves[i]; break; }
            }
            if (terrain().den_owner[pick.to] == opposite_color(pos.stm)) return 1.0;
            make_move(pos, pick);
            if (pos.ply >= p_.max_ply) return 0.0;
        }
        return std::tanh(double(evaluate(pos)) / p_.eval_scale);
    }

    // --- expansion -------------------------------------------------------
    void expand_node(uint32_t idx) {
        if (nodes_[idx].num_moves || nodes_[idx].status != ST_ONGOING) return;

        Position pos = node_position(idx);
        Move moves[128];
        const int n = generate_moves(pos, moves);
        nodes_[idx].num_moves = uint16_t(n);
        if (n == 0) {
            nodes_[idx].status = ST_LOSS;   // side to move is stuck
            return;
        }

        std::array<int, 128> order{};
        for (int i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.begin() + n,
                         [&](int a, int b) {
                             return move_prior(pos, moves[a]) > move_prior(pos, moves[b]);
                         });

        move_lists_.emplace_back();
        auto& ml = move_lists_.back();
        ml.resize(n);
        for (int i = 0; i < n; ++i) ml[i] = moves[order[i]];
        nodes_[idx].move_list = uint32_t(move_lists_.size() - 1);
        nodes_[idx].leaf_value = leaf_value(pos);

        // Reserve a contiguous slot in the child-reference table for every
        // legal move.  The child *nodes* are created lazily.
        nodes_[idx].children_start = uint32_t(child_refs_.size());
        child_refs_.resize(child_refs_.size() + n, kNoChild);
    }

    int max_children_allowed(uint32_t idx) const {
        const MctsNode& node = nodes_[idx];
        const int budget = 1 + int(p_.widening_k * std::sqrt(double(node.visits)));
        return std::min<int>(std::min(p_.widening_max, int(node.num_moves)), budget);
    }

    // Create child `child_no` of `idx` (children are created in prior order).
    void create_child(uint32_t idx, uint16_t child_no, std::mt19937_64& rng) {
        const Position pos = node_position(idx);
        const Move m = move_lists_[nodes_[idx].move_list][child_no];

        // Prior: normalised heuristic score, with Dirichlet noise at the root.
        float prior = move_prior(pos, m);
        if (idx == root_ && p_.dirichlet_eps > 0.0) {
            std::gamma_distribution<double> gamma(p_.dirichlet_alpha, 1.0);
            double noise = 0.0;
            for (int k = 0; k < child_no; ++k) noise += gamma(rng);
            const double eps = p_.dirichlet_eps;
            prior = float((1.0 - eps) * prior + eps * noise);
        }

        MctsNode child;
        Position cpos = pos;
        make_move(cpos, m);
        std::memcpy(child.board, cpos.board, NUM_SQ);
        child.stm  = cpos.stm;
        child.ply  = cpos.ply;
        child.key  = cpos.key;
        child.rep  = uint8_t(nodes_[idx].rep + (nodes_[idx].key == cpos.key ? 1 : 0));
        child.from = m.from;
        child.to   = m.to;
        child.prior = prior;

        if (terrain().den_owner[m.to] == opposite_color(pos.stm)) {
            child.status = ST_LOSS;   // the mover reached the den: the child loses
        } else if (child.rep >= 2) {
            child.status = ST_DRAW;   // three-fold repetition
        } else if (child.ply >= p_.max_ply) {
            child.status = ST_DRAW;
        } else {
            Move buf[128];
            if (generate_moves(cpos, buf) == 0) child.status = ST_LOSS;
        }

        const uint32_t child_idx = uint32_t(nodes_.size());
        nodes_.push_back(child);
        child_refs_[nodes_[idx].children_start + child_no] = child_idx;
        if (child_no + 1 > nodes_[idx].num_children)
            nodes_[idx].num_children = uint16_t(child_no + 1);
    }

    void create_children(uint32_t idx, bool all, std::mt19937_64& rng) {
        if (nodes_[idx].status != ST_ONGOING) return;
        const uint16_t n = nodes_[idx].num_moves;
        for (uint16_t i = 0; i < n; ++i) {
            if (!all && i >= max_children_allowed(idx)) break;
            create_child(idx, i, rng);
        }
    }

    // --- selection & backup ----------------------------------------------
    double descend(uint32_t idx, std::mt19937_64& rng) {
        if (nodes_[idx].status != ST_ONGOING) {
            const double v = (nodes_[idx].status == ST_WIN)   ? 1.0
                            : (nodes_[idx].status == ST_LOSS) ? -1.0
                                                                : 0.0;
            nodes_[idx].visits++;
            nodes_[idx].value_sum += v;
            return v;
        }

        if (nodes_[idx].visits == 0) {
            expand_node(idx);
            if (nodes_[idx].status != ST_ONGOING) {
                const double v = (nodes_[idx].status == ST_WIN)   ? 1.0
                                : (nodes_[idx].status == ST_LOSS) ? -1.0
                                                                    : 0.0;
                nodes_[idx].visits++;
                nodes_[idx].value_sum += v;
                return v;
            }
            // First visit: evaluate the leaf statically.
            const double v = nodes_[idx].leaf_value;
            nodes_[idx].visits++;
            nodes_[idx].value_sum += v;
            return v;
        }

        // Create another child if the widening budget allows.
        if (nodes_[idx].num_children < max_children_allowed(idx))
            create_child(idx, nodes_[idx].num_children, rng);

        const uint16_t nchild = nodes_[idx].num_children;
        if (nchild == 0) {
            const double v = nodes_[idx].leaf_value;
            nodes_[idx].visits++;
            nodes_[idx].value_sum += v;
            return v;
        }

        const uint32_t base = nodes_[idx].children_start;
        const double sqrt_n = std::sqrt(double(nodes_[idx].visits));
        int best_i = 0;
        double best_score = -1e18;
        for (uint16_t i = 0; i < nchild; ++i) {
            const MctsNode& c = nodes_[child_refs_[base + i]];
            const double q = c.visits ? -(c.value_sum / c.visits) : 0.0;
            const double u = p_.c_puct * c.prior * sqrt_n / (1.0 + c.visits);
            const double s = q + u;
            if (s > best_score) { best_score = s; best_i = i; }
        }

        const double v_child = descend(child_refs_[base + best_i], rng);
        const double v = -v_child;
        nodes_[idx].visits++;
        nodes_[idx].value_sum += v;
        return v;
    }

    MctsParams  p_;
    Evaluator* ev_;
    static constexpr uint32_t kNoChild = 0xFFFFFFFFu;
    std::vector<MctsNode> nodes_;
    std::vector<std::vector<Move>> move_lists_;
    std::vector<uint32_t> child_refs_;
    uint32_t root_ = 0;
};

}  // namespace jungle
