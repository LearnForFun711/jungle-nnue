// engine_api.hpp -- A single interface over the two ways the engine can pick
// a move: iterative-deepening alpha-beta or PUCT MCTS.
//
// Having both behind one interface lets the match harness compare any pair
// of engines, which is how we measure whether the MCTS teacher is actually
// stronger than the handcrafted alpha-beta baseline before training on it.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "eval.hpp"
#include "jungle.hpp"
#include "mcts.hpp"
#include "nnue.hpp"
#include "search.hpp"

namespace jungle {

struct EngineConfig {
    enum Kind { ALPHABETA, MCTS } kind = ALPHABETA;

    bool   use_nnue = false;
    std::string nnue_path;
    float  nnue_scale = 10.0f;  // multiplies the network's value

    // alpha-beta
    double time_limit = 1.0;   // seconds per move
    uint64_t node_limit = 0;

    // MCTS
    int    mcts_sims = 800;
    double mcts_c_puct = 1.4;
    int    mcts_rollouts = 0;
    double mcts_temperature = 0.0;
    int    mcts_leaf_depth = 1;
    int    mcts_leaf_agg = 1;
    double mcts_widening = 1.6;

    int    tt_mb = 32;
};

class Engine {
public:
    explicit Engine(const EngineConfig& cfg)
        : cfg_(cfg) {
        ev_.value_scale = cfg_.nnue_scale;
        if (cfg_.use_nnue && !cfg_.nnue_path.empty()) {
            if (!nnue_load(cfg_.nnue_path, ev_.net)) {
                ev_.net = NnueNetwork();
            }
        }
        if (cfg_.kind == EngineConfig::ALPHABETA) {
            searcher_ = std::make_unique<Searcher>(ev_, size_t(cfg_.tt_mb));
        } else {
            MctsParams mp;
            mp.simulations = cfg_.mcts_sims;
            mp.c_puct = cfg_.mcts_c_puct;
            mp.rollouts = cfg_.mcts_rollouts;
            mp.leaf_depth = cfg_.mcts_leaf_depth;
            mp.leaf_agg = cfg_.mcts_leaf_agg;
            mp.widening_k = cfg_.mcts_widening;
            mp.use_nnue = cfg_.use_nnue && ev_.uses_nnue();
            mcts_ = std::make_unique<Mcts>(mp, &ev_);
            mcts_params_ = mp;
        }
    }

    Move best_move(Position& pos, std::vector<uint64_t>& history) {
        if (cfg_.kind == EngineConfig::ALPHABETA) {
            SearchLimits limits;
            limits.time_limit = cfg_.time_limit;
            limits.node_limit = cfg_.node_limit;
            limits.max_depth = 64;
            SearchResult r = searcher_->search(pos, limits, history);
            last_nodes_ = r.nodes;
            last_depth_ = r.depth;
            return r.move;
        }
        MctsResult r = mcts_->search(pos, rng_);
        if (cfg_.mcts_temperature > 0.0) return sample_by_visits(r, cfg_.mcts_temperature);
        return r.best;
    }

    int last_score() const { return last_score_; }
    uint64_t last_nodes() const { return last_nodes_; }
    int last_depth() const { return last_depth_; }

private:
    struct Visit { Move move; int visits; };

    Move sample_by_visits(const MctsResult& r, double t) {
        double sum = 0.0;
        std::vector<double> w;
        for (const auto& v : r.visits) {
            if (v.visits <= 0) continue;
            double x = std::pow(double(v.visits), 1.0 / std::max(1e-6, t));
            w.push_back(x);
            sum += x;
        }
        if (w.empty()) return r.best;
        double u = std::uniform_real_distribution<double>(0.0, sum)(rng_);
        for (size_t i = 0; i < w.size(); ++i) {
            u -= w[i];
            if (u <= 0) return r.visits[i].move;
        }
        return r.best;
    }

    EngineConfig cfg_;
    Evaluator ev_;
    std::unique_ptr<Searcher> searcher_;
    std::unique_ptr<Mcts> mcts_;
    MctsParams mcts_params_;
    std::mt19937_64 rng_{0x243F6A8885A308D3ULL};
    int last_score_ = 0;
    uint64_t last_nodes_ = 0;
    int last_depth_ = 0;
};

}  // namespace jungle
