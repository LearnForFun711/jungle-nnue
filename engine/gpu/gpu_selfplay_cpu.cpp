// gpu_selfplay_cpu.cpp -- The self-play generator, on the CPU.
//
// This is the reference implementation of the GPU generator: the same rules,
// the same search, the same records, run one game at a time on the CPU.  It
// exists for three reasons:
//
//   * it is what the test suite runs, so the game loop, the search and the
//     record format are all checked on a machine without a GPU;
//   * it is the fallback when there is no GPU, or when the CUDA build fails;
//   * it is the yardstick the CUDA version is measured against.
//
//   ./gpu_selfplay_cpu --net net/net.nnue --games 4 --sims 800 --out data/cpu
//
// It also plays a league round, where the two sides use different networks:
//
//   ./gpu_selfplay_cpu --nets a.nnue,b.nnue --schedule 0:1,1:0 --games 4
//
// The schedule says which pair of networks plays in which game, and it is
// cycled over the games.  This is the reference implementation of the
// league: the CUDA build has to agree with it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <array>
#include <random>
#include <string>
#include <vector>

#include "jungle_game.cuh"

using namespace jungle_gpu;

namespace {

struct Options {
    std::string net_path;
    std::vector<std::string> net_paths;    // the league
    std::vector<std::pair<int, int>> schedule;
    std::string out = "data/gpu";
    int    games = 4;
    int    sims = 800;
    int    seed = 12345;
    double leaf_scale = 1.0;
    double value_scale = 10.0;
    double c_puct = 1.4;
    double widening = 1.6;
    double dirichlet_alpha = 0.30;
    double dirichlet_eps = 0.25;
    int    max_ply = 200;
    bool   quiet = false;
};

void split(const std::string& s, char sep, std::vector<std::string>& out) {
    size_t start = 0;
    while (true) {
        const size_t p = s.find(sep, start);
        if (p == std::string::npos) {
            if (start < s.size()) out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--net") o.net_path = next();
        else if (a == "--nets") {
            std::vector<std::string> parts;
            split(next(), ',', parts);
            for (const std::string& p : parts)
                if (!p.empty()) o.net_paths.push_back(p);
        }
        else if (a == "--schedule") {
            std::vector<std::string> parts;
            split(next(), ',', parts);
            for (const std::string& p : parts) {
                const size_t colon = p.find(':');
                if (colon == std::string::npos) continue;
                o.schedule.push_back({std::atoi(p.substr(0, colon).c_str()),
                                      std::atoi(p.substr(colon + 1).c_str())});
            }
        }
        else if (a == "--out") o.out = next();
        else if (a == "--games") o.games = std::atoi(next());
        else if (a == "--sims") o.sims = std::atoi(next());
        else if (a == "--seed") o.seed = std::atoi(next());
        else if (a == "--leaf-scale") o.leaf_scale = std::stod(next());
        else if (a == "--value-scale") o.value_scale = std::stod(next());
        else if (a == "--c-puct") o.c_puct = std::stod(next());
        else if (a == "--widening") o.widening = std::stod(next());
        else if (a == "--dirichlet-alpha") o.dirichlet_alpha = std::stod(next());
        else if (a == "--dirichlet-eps") o.dirichlet_eps = std::stod(next());
        else if (a == "--max-ply") o.max_ply = std::atoi(next());
        else if (a == "--quiet") o.quiet = true;
    }
    if (o.net_paths.empty() && !o.net_path.empty()) o.net_paths.push_back(o.net_path);
    if (o.net_paths.empty()) {
        std::fprintf(stderr, "usage: gpu_selfplay_cpu --net <file> [--games n] [--out prefix]\n");
        return 2;
    }
    if (o.schedule.empty()) o.schedule.push_back({0, 0});

    std::vector<Net> nets(o.net_paths.size());
    for (size_t i = 0; i < o.net_paths.size(); ++i) {
        if (!load_net(o.net_paths[i].c_str(), nets[i])) {
            std::fprintf(stderr, "cannot load %s\n", o.net_paths[i].c_str());
            return 1;
        }
        if (nets[i].ft_hidden != nets[0].ft_hidden || nets[i].hidden != nets[0].hidden) {
            std::fprintf(stderr, "every network must have the same shape\n");
            return 1;
        }
    }
    const Geometry geom = build_geometry();

    MctsConfig cfg;
    cfg.sims = o.sims;
    cfg.leaf_scale = o.leaf_scale;
    cfg.value_scale = o.value_scale;
    cfg.c_puct = o.c_puct;
    cfg.widening_k = o.widening;
    cfg.dirichlet_alpha = o.dirichlet_alpha;
    cfg.dirichlet_eps = o.dirichlet_eps;
    cfg.max_ply = o.max_ply;

    std::vector<int32_t> acc(size_t(2) * nets[0].ft_hidden);
    double sum_vals = 0.0, sum_sq = 0.0; int64_t n_vals = 0;
    std::vector<float> child_values(MAX_CHILDREN);
    std::vector<uint8_t> child_board(63);

    // The result of every game, and how many each pair of networks won.
    std::vector<int> results(size_t(o.games), 0);
    std::map<std::pair<int, int>, std::array<int, 3>> pair_results;

    std::FILE* f = std::fopen((o.out + ".bin").c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s.bin\n", o.out.c_str());
        return 1;
    }

    GameMem mem;
    Game game;
    int64_t positions = 0;
    int red_wins = 0, black_wins = 0, draws = 0;

    for (int gi = 0; gi < o.games; ++gi) {
        const auto& pair = o.schedule[size_t(gi) % o.schedule.size()];
        const int red_id = pair.first, black_id = pair.second;
        const Net& red_net = nets[size_t(red_id)];
        const Net& black_net = nets[size_t(black_id)];

        game.start(&mem, &geom, &cfg, uint64_t(o.seed) * 0x9E3779B97F4A7C15ULL + uint64_t(gi) + 1,
                   uint8_t(red_id), uint8_t(black_id));

        while (game.phase != GAME_OVER) {
            if (game.step_search()) {
                // The expansion needs the evaluation of every child.  On the
                // GPU these are collected from every game in flight and
                // evaluated in one batch; here they are evaluated one by one.
                //
                // The leaf is evaluated by the network of the side that will
                // move there, which is the opponent of the side to move in
                // the parent.
                const Board& parent = game.t.board;
                const Net& leaf_net = (parent.stm == RED) ? black_net : red_net;
                for (int i = 0; i < game.t.nchildren; ++i) {
                    std::memcpy(child_board.data(), parent.sq, 63);
                    const int from = game.t.child_from[i];
                    const int to = game.t.child_to[i];
                    child_board[from] = 0;
                    child_board[to] = parent.sq[from];
                    child_values[i] = teacher_leaf_value(
                        nnue_eval(leaf_net, child_board.data(), parent.stm ^ 1, acc.data()),
                        float(cfg.leaf_scale), float(cfg.value_scale));
                    sum_vals += child_values[i];
                    sum_sq += double(child_values[i]) * double(child_values[i]);
                    ++n_vals;
                }
                game.t.finish_expansion(child_values.data());
                game.t.backup();
            }
            if (game.phase == GAME_MOVE) game.step_move();
        }

        game.finish_records();
        std::fwrite(mem.records, 68, size_t(mem.record_count), f);
        positions += mem.record_count;
        results[size_t(gi)] = game.result;
        auto& tally = pair_results[{red_id, black_id}];
        if (game.result > 0) { ++red_wins; ++tally[0]; }
        else if (game.result < 0) { ++black_wins; ++tally[1]; }
        else { ++draws; ++tally[2]; }
    }
    std::fclose(f);

    if (!o.quiet) {
        std::printf("games %d  positions %lld  R%d B%d D%d\n", o.games,
                    (long long)positions, red_wins, black_wins, draws);
        if (n_vals) {
            const double mean = sum_vals / double(n_vals);
            std::printf("raw network value: mean %+.4f  std %.4f  (n=%lld)\n", mean,
                        std::sqrt(sum_sq / double(n_vals) - mean * mean), (long long)n_vals);
        }
        // The league table: one line per pair of networks.
        for (const auto& entry : pair_results) {
            const auto& t = entry.second;
            const int n = t[0] + t[1] + t[2];
            std::printf("pair %d:%d  games %d  red wins %d  black wins %d  draws %d  "
                        "score(red) %.3f\n",
                        entry.first.first, entry.first.second, n, t[0], t[1], t[2],
                        n ? (t[0] + 0.5 * t[2]) / n : 0.0);
        }
        // The result of every game, so that a checker can compare it against
        // the result the records themselves carry.
        std::printf("game results:");
        for (int g = 0; g < o.games; ++g) std::printf(" %d", results[size_t(g)]);
        std::printf("\n");
    }
    return 0;
}
