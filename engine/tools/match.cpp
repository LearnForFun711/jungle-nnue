// match.cpp -- Engine-vs-engine match harness.
//
//   ./match --games 400 --time 1.0 --nnue net.nnue
//   ./match --games 400 --time 1.0 --engine-a mcts --engine-b ab
//   ./match --games 200 --time 0.2 --nnue a.nnue --opponent-nnue b.nnue
//
// The engine on side A plays the engine on side B; colours alternate from
// game to game so every position is played from both sides.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../src/engine_api.hpp"
#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/nnue.hpp"
#include "../src/search.hpp"

using namespace jungle;

namespace {

struct Options {
    int    games = 200;
    double time_limit = 1.0;
    uint64_t node_limit = 0;
    int    threads = 1;
    uint64_t seed = 987654321;
    int    max_ply = 200;

    // Engine A
    EngineConfig a;
    // Engine B
    EngineConfig b;
};

enum End { END_DEN, END_NO_MOVES, END_REPETITION, END_LENGTH };

struct GameResult {
    int  result;   // +1 A wins (as Red), 0 draw, -1 B wins
    int  plies;
    int  cause;
    int  red_won;  // 1 if the engine playing Red won
};

struct Stats { long long nodes = 0; long long depth = 0; long long moves = 0; };

GameResult play_game(const Options& o, Engine& red_engine, Engine& black_engine,
                     Stats* red_stats = nullptr, Stats* black_stats = nullptr) {
    Position pos;
    std::vector<uint64_t> history;
    history.push_back(pos.key);

    while (true) {
        Engine& engine = (pos.stm == RED) ? red_engine : black_engine;
        Stats* stats = (pos.stm == RED) ? red_stats : black_stats;
        Move m = engine.best_move(pos, history);
        if (m.from == 0xFFFF) break;
        if (stats) {
            stats->nodes += (long long)engine.last_nodes();
            stats->depth += engine.last_depth();
            ++stats->moves;
        }

        const bool den_entry = (terrain().den_owner[m.to] == opposite_color(pos.stm));
        make_move(pos, m);
        history.push_back(pos.key);

        if (den_entry)
            return {pos.stm == RED ? +1 : -1, pos.ply, END_DEN, pos.stm == RED ? 1 : 0};

        int reps = 0;
        for (uint64_t k : history) if (k == pos.key) ++reps;
        if (reps >= 3) return {0, pos.ply, END_REPETITION, 0};
        if (int(pos.ply) >= o.max_ply) return {0, pos.ply, END_LENGTH, 0};

        Move buf[128];
        if (generate_moves(pos, buf) == 0)
            return {pos.stm == RED ? -1 : +1, pos.ply, END_NO_MOVES, pos.stm == RED ? 0 : 1};
    }
    return {0, pos.ply, END_LENGTH, 0};
}

double elo_from_score(double s) {
    if (s <= 0.0) return -800.0;
    if (s >= 1.0) return 800.0;
    return -400.0 * std::log10(1.0 / s - 1.0);
}

EngineConfig parse_engine(const std::string& spec, const Options& o) {
    // spec: "ab" | "mcts" | "mcts:1600:cpuct=1.5" | "ab:nnue=path:scale=3.4"
    EngineConfig c;
    c.time_limit = o.time_limit;
    c.node_limit = o.node_limit;

    std::string kind = spec;
    std::string rest;
    const size_t colon = spec.find(':');
    if (colon != std::string::npos) {
        kind = spec.substr(0, colon);
        rest = spec.substr(colon + 1);
    }

    // The option list after the engine kind, as key/value pairs.
    struct Opt { std::string key; std::string value; bool has_value; };
    std::vector<Opt> opts;
    {
        std::stringstream ss(rest);
        std::string tok;
        while (std::getline(ss, tok, ':')) {
            if (tok.empty()) continue;
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) opts.push_back({tok, std::string(), false});
            else opts.push_back({tok.substr(0, eq), tok.substr(eq + 1), true});
        }
    }

    if (kind == "mcts" || kind == "m") {
        c.kind = EngineConfig::MCTS;
        // The first bare token is the simulation count.
        bool first = true;
        for (const Opt& op : opts) {
            if (first && !op.has_value) {
                c.mcts_sims = std::stoi(op.key);
                first = false;
                continue;
            }
            first = false;
            if (op.key == "cpuct") c.mcts_c_puct = std::stod(op.value);
            else if (op.key == "leaf") c.mcts_leaf_depth = std::stoi(op.value);
            else if (op.key == "widen") c.mcts_widening = std::stod(op.value);
            else if (op.key == "agg") c.mcts_leaf_agg = std::stoi(op.value);
            else if (op.key == "nnue") { c.use_nnue = true; c.nnue_path = op.value; }
            else if (op.key == "scale") c.nnue_scale = std::stof(op.value);
        }
        return c;
    }

    c.kind = EngineConfig::ALPHABETA;
    for (const Opt& op : opts) {
        if (op.key == "nnue") {
            c.use_nnue = true;
            c.nnue_path = op.has_value ? op.value : o.a.nnue_path;
        } else if (op.key == "scale") {
            c.nnue_scale = std::stof(op.value);
        }
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::string engine_a = "ab", engine_b = "ab";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--games") o.games = std::stoi(next());
        else if (a == "--time") o.time_limit = std::stod(next());
        else if (a == "--nodes") o.node_limit = std::stoull(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--max-ply") o.max_ply = std::stoi(next());
        else if (a == "--engine-a") engine_a = next();
        else if (a == "--engine-b") engine_b = next();
        else if (a == "--nnue") o.a.nnue_path = next();
        else if (a == "--opponent-nnue") { o.b.nnue_path = next(); }
    }

    o.a = parse_engine(engine_a, o);
    o.b = parse_engine(engine_b, o);
    // "--nnue path" applies to the A engine, "--opponent-nnue path" to B.
    if (o.a.nnue_path.empty() && !engine_a.empty() && engine_a.find("nnue") != std::string::npos)
        o.a.nnue_path = o.a.nnue_path;
    {
        EngineConfig parsed_a = parse_engine(engine_a, o);
        parsed_a.nnue_path = o.a.nnue_path.empty() ? parsed_a.nnue_path : o.a.nnue_path;
        parsed_a.use_nnue = !parsed_a.nnue_path.empty();
        o.a = parsed_a;
        EngineConfig parsed_b = parse_engine(engine_b, o);
        parsed_b.nnue_path = o.b.nnue_path.empty() ? parsed_b.nnue_path : o.b.nnue_path;
        parsed_b.use_nnue = !parsed_b.nnue_path.empty();
        o.b = parsed_b;
    }

    const int threads = std::max(1, o.threads);
    std::atomic<int> a_wins{0}, b_wins{0}, draws{0}, done{0};
    std::atomic<int> decisive_a{0}, decisive_b{0};
    std::atomic<int> end_den{0}, end_nomoves{0}, end_rep{0}, end_len{0};
    std::atomic<long long> total_plies{0};
    std::atomic<long long> total_nodes{0};
    std::atomic<long long> total_depth{0};
    std::atomic<long long> total_moves{0};
    std::atomic<long long> total_a_nodes{0}, total_a_depth{0}, total_a_moves{0};
    std::atomic<long long> total_b_nodes{0}, total_b_depth{0}, total_b_moves{0};

    auto worker = [&](int tid) {
        Engine ea(o.a), eb(o.b);
        Stats a_stats, b_stats;
        for (int g = tid; g < o.games; g += threads) {
            const bool a_is_red = ((g & 1) == 0);
            Engine& red_engine   = a_is_red ? ea : eb;
            Engine& black_engine = a_is_red ? eb : ea;

            GameResult gr = play_game(o, red_engine, black_engine,
                                       a_is_red ? &a_stats : &b_stats,
                                       a_is_red ? &b_stats : &a_stats);
            int outcome;   // from A's point of view
            if (gr.result == 0) outcome = 0;
            else {
                const bool a_won = (gr.result > 0) == a_is_red;
                outcome = a_won ? 1 : -1;
            }
            if (outcome > 0) { ++a_wins; ++decisive_a; }
            else if (outcome < 0) { ++b_wins; ++decisive_b; }
            else ++draws;

            switch (gr.cause) {
                case END_DEN: ++end_den; break;
                case END_NO_MOVES: ++end_nomoves; break;
                case END_REPETITION: ++end_rep; break;
                default: ++end_len; break;
            }
            total_plies += gr.plies;
            total_nodes += a_stats.nodes + b_stats.nodes;
            total_depth += a_stats.depth + b_stats.depth;
            total_moves += a_stats.moves + b_stats.moves;
            total_a_nodes += a_stats.nodes;
            total_a_depth += a_stats.depth;
            total_a_moves += a_stats.moves;
            total_b_nodes += b_stats.nodes;
            total_b_depth += b_stats.depth;
            total_b_moves += b_stats.moves;

            const int d = ++done;
            if (d % 20 == 0 || d == o.games) {
                std::fprintf(stderr, "\r  games %d/%d  A%d B%d D%d   ", d, o.games,
                             a_wins.load(), b_wins.load(), draws.load());
                std::fflush(stderr);
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    const int n = o.games;
    const int aw = a_wins.load(), bw = b_wins.load(), dr = draws.load();
    const double mean = (double(aw) + 0.5 * double(dr)) / double(n);
    const double se = std::sqrt(std::max(0.0, mean * (1.0 - mean) / double(n)));

    std::fprintf(stderr, "\n");
    std::printf("{\n");
    std::printf("  \"engine_a\": \"%s\",\n", engine_a.c_str());
    std::printf("  \"engine_b\": \"%s\",\n", engine_b.c_str());
    std::printf("  \"games\": %d,\n", n);
    std::printf("  \"a_wins\": %d,\n", aw);
    std::printf("  \"b_wins\": %d,\n", bw);
    std::printf("  \"draws\": %d,\n", dr);
    std::printf("  \"score_a\": %.4f,\n", mean);
    const int dec_total = decisive_a.load() + decisive_b.load();
    if (dec_total > 0) {
        const double dmean = double(decisive_a.load()) / double(dec_total);
        const double dse = std::sqrt(std::max(0.0, dmean * (1.0 - dmean) / double(dec_total)));
        std::printf("  \"decisive_games\": %d,\n", dec_total);
        std::printf("  \"decisive_score_a\": %.4f,\n", dmean);
        std::printf("  \"decisive_score_ci95\": [%.4f, %.4f],\n",
                    dmean - 1.96 * dse, dmean + 1.96 * dse);
    }
    std::printf("  \"score_ci95\": [%.4f, %.4f],\n", mean - 1.96 * se, mean + 1.96 * se);
    std::printf("  \"elo_a\": %.1f,\n", elo_from_score(mean));
    std::printf("  \"endings\": {\"den\": %d, \"no_moves\": %d, \"repetition\": %d, \"length\": %d},\n",
                end_den.load(), end_nomoves.load(), end_rep.load(), end_len.load());
    std::printf("  \"avg_plies\": %.1f,\n", double(total_plies.load()) / double(n));
    std::printf("  \"avg_nodes_per_move\": %.0f,\n",
                double(total_nodes.load()) / std::max(1LL, total_moves.load()));
    std::printf("  \"avg_depth\": %.2f,\n",
                double(total_depth.load()) / std::max(1LL, total_moves.load()));
    // Per-engine numbers: this is what shows whether one side is handicapped
    // by a slower evaluation (fewer nodes per move at the same clock).
    std::printf("  \"speed_a\": {\"avg_nodes_per_move\": %.0f, \"avg_depth\": %.2f},\n",
                double(total_a_nodes.load()) / std::max(1LL, total_a_moves.load()),
                double(total_a_depth.load()) / std::max(1LL, total_a_moves.load()));
    std::printf("  \"speed_b\": {\"avg_nodes_per_move\": %.0f, \"avg_depth\": %.2f},\n",
                double(total_b_nodes.load()) / std::max(1LL, total_b_moves.load()),
                double(total_b_depth.load()) / std::max(1LL, total_b_moves.load()));
    std::printf("  \"time_limit\": %.4f,\n", o.time_limit);
    std::printf("  \"node_limit\": %llu\n", (unsigned long long)o.node_limit);
    std::printf("}\n");
    return 0;
}
