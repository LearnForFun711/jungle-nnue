// bench.cpp -- Micro-benchmarks for the hot paths.
//
//   ./bench
#include <chrono>
#include <cstdio>
#include <random>

#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/mcts.hpp"
#include "../src/nnue.hpp"

using namespace jungle;
using Clock = std::chrono::steady_clock;

template <typename F>
double per_call(F&& f, int reps) {
    const auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) f();
    const double el = std::chrono::duration<double>(Clock::now() - t0).count();
    return el / reps * 1e6;  // microseconds
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20000;

    // A mid-game position.
    Position pos;
    pos.set_fen("l5t/1c3d1/r1p1w1e/7/7/6R/E1W1P2/1D3C1/T5L b");

    Move moves[128];
    volatile int sink = 0;

    const double t_eval = per_call([&] { sink += evaluate(pos); }, reps);
    const double t_gen = per_call([&] { sink += generate_moves(pos, moves); }, reps);

    // One-ply leaf evaluation (the MCTS leaf_depth=1 path).
    const double t_leaf1 = per_call([&] {
        const int n = generate_moves(pos, moves);
        int best = -MATE_SCORE;
        for (int i = 0; i < n; ++i) {
            Position child = pos;
            make_move(child, moves[i]);
            best = std::max(best, -evaluate(child));
        }
        sink += best;
    }, reps);

    // MCTS simulation throughput.
    {
        MctsParams p;
        p.simulations = 2000;
        p.leaf_depth = 0;
        Evaluator ev;
        Mcts m(p, &ev);
        std::mt19937_64 rng(1234);
        const auto t0 = Clock::now();
        int iters = 20;
        for (int i = 0; i < iters; ++i) m.search(pos, rng);
        const double el = std::chrono::duration<double>(Clock::now() - t0).count();
        const double sims = double(iters) * p.simulations;
        std::printf("mcts (leaf_depth=0): %.2f us/sim  (%.0f sims/s)\n",
                    el / sims * 1e6, sims / el);
    }
    {
        MctsParams p;
        p.simulations = 500;
        p.leaf_depth = 1;
        Evaluator ev;
        Mcts m(p, &ev);
        std::mt19937_64 rng(1234);
        const auto t0 = Clock::now();
        int iters = 10;
        for (int i = 0; i < iters; ++i) m.search(pos, rng);
        const double el = std::chrono::duration<double>(Clock::now() - t0).count();
        const double sims = double(iters) * p.simulations;
        std::printf("mcts (leaf_depth=1): %.2f us/sim  (%.0f sims/s)\n",
                    el / sims * 1e6, sims / el);
    }

    // NNUE evaluation speed.
    {
        NnueNetwork net;
        if (nnue_load("../build/net_512_256.nnue", net)) {
            const double t_nn = per_call([&] {
                nnue_refresh(net, pos);
                sink += int(nnue_forward(net, pos) * 1000.0f);
            }, reps);
            std::printf("nnue eval (512/256): %.3f us\n", t_nn);
        }
        if (nnue_load("../build/net_512_128.nnue", net)) {
            const double t_nn = per_call([&] {
                nnue_refresh(net, pos);
                sink += int(nnue_forward(net, pos) * 1000.0f);
            }, reps);
            std::printf("nnue eval (512/128): %.3f us\n", t_nn);
        }
        if (nnue_load("../build/net_256_128.nnue", net)) {
            const double t_nn = per_call([&] {
                nnue_refresh(net, pos);
                sink += int(nnue_forward(net, pos) * 1000.0f);
            }, reps);
            std::printf("nnue eval (256/128): %.3f us\n", t_nn);
        }
    }

    std::printf("evaluate():      %.3f us\n", t_eval);
    std::printf("generate_moves(): %.3f us\n", t_gen);
    std::printf("1-ply leaf eval: %.3f us\n", t_leaf1);
    (void)sink;
    return 0;
}
