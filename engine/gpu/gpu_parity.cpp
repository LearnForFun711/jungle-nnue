// gpu_parity.cpp -- Check the GPU core against the engine.
//
// Two things are checked, on positions taken from real games:
//
//   * the move generator: the bitboard version must produce exactly the same
//     set of legal moves as engine/src/jungle.hpp's, on every position;
//
//   * the network: the GPU's forward pass must produce the same value as
//     engine/src/nnue.hpp's, to within the rounding of the last layer.
//
// The positions come from random games played with the engine's own rules,
// which is the same way scripts/verify_rules.py checks the engine against
// the Python reference.
//
//   ./gpu_parity --net net/net.nnue --games 40 --positions 200000
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "../src/jungle.hpp"
#include "../src/nnue.hpp"
#include "jungle_gpu.cuh"

using jungle::Move;
using jungle::Position;
using jungle::generate_moves;
using jungle::make_move;
using jungle::nnue_forward;
using jungle::nnue_refresh;
using jungle::nnue_load;
using jungle::NnueNetwork;
using jungle_gpu::Board;
using jungle_gpu::Geometry;
using jungle_gpu::MoveBuf;
using jungle_gpu::Net;
using jungle_gpu::build_geometry;
using jungle_gpu::load_net;
using jungle_gpu::nnue_eval;

namespace {

// One position, in both representations.
struct Sample {
    uint8_t board[jungle::NUM_SQ];
    uint8_t stm;
    uint16_t ply;
};

std::vector<Sample> collect_samples(uint64_t seed, int games, int wanted) {
    std::vector<Sample> out;
    std::mt19937_64 rng(seed);
    Position pos;
    Move moves[128];
    for (int g = 0; g < games && int(out.size()) < wanted; ++g) {
        pos.reset();
        for (int ply = 0; ply < 200 && int(out.size()) < wanted; ++ply) {
            Sample s;
            std::memcpy(s.board, pos.board, jungle::NUM_SQ);
            s.stm = pos.stm;
            s.ply = pos.ply;
            out.push_back(s);
            const int n = generate_moves(pos, moves);
            if (n == 0) break;
            const Move m = moves[rng() % n];
            make_move(pos, m);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    uint64_t seed = 12345;
    int games = 40;
    int wanted = 200000;
    const char* net_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--seed") seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--games") games = std::atoi(next());
        else if (a == "--positions") wanted = std::atoi(next());
        else if (a == "--net") net_path = next();
    }

    const Geometry geom = build_geometry();

    std::vector<Sample> samples = collect_samples(seed, games, wanted);
    std::printf("sampled %zu positions\n", samples.size());

    // ---- the move generator -------------------------------------------
    int mismatches = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const Sample& s = samples[i];

        Position pos;
        std::memcpy(pos.board, s.board, jungle::NUM_SQ);
        pos.stm = s.stm;
        pos.ply = s.ply;
        pos.key = pos.compute_key();

        Move ref[128];
        const int n_ref = generate_moves(pos, ref);
        std::set<std::pair<int, int>> ref_set;
        for (int k = 0; k < n_ref; ++k)
            ref_set.insert({ref[k].from, ref[k].to});

        Board gb;
        std::memcpy(gb.sq, s.board, jungle::NUM_SQ);
        gb.stm = s.stm;
        MoveBuf buf;
        const int n_gpu = jungle_gpu::generate_moves(gb, geom, buf);
        std::set<std::pair<int, int>> gpu_set;
        for (int k = 0; k < n_gpu; ++k)
            gpu_set.insert({buf.from[k], buf.to[k]});

        if (ref_set != gpu_set) {
            ++mismatches;
            if (mismatches <= 5) {
                std::printf("MISMATCH at sample %zu\n  reference only:", i);
                for (const auto& m : ref_set)
                    if (!gpu_set.count(m)) std::printf(" %d->%d", m.first, m.second);
                std::printf("\n  gpu only:     ");
                for (const auto& m : gpu_set)
                    if (!ref_set.count(m)) std::printf(" %d->%d", m.first, m.second);
                std::printf("\n");
            }
        }
    }
    if (mismatches) {
        std::printf("FAIL: %d of %zu positions disagree\n", mismatches, samples.size());
        return 1;
    }
    std::printf("PASS: the move generator agrees on all %zu positions\n", samples.size());

    // ---- the network ---------------------------------------------------
    if (net_path) {
        NnueNetwork net;
        if (!nnue_load(net_path, net)) {
            std::printf("FAIL: cannot load %s\n", net_path);
            return 1;
        }
        Net gnet;
        if (!load_net(net_path, gnet)) {
            std::printf("FAIL: cannot load %s into the GPU network\n", net_path);
            return 1;
        }

        std::vector<int32_t> acc(size_t(2) * gnet.ft_hidden);
        double worst = 0.0;
        int differing = 0;
        const int check = std::min<int>(samples.size(), 20000);
        for (int i = 0; i < check; ++i) {
            const Sample& s = samples[i];
            Position pos;
            std::memcpy(pos.board, s.board, jungle::NUM_SQ);
            pos.stm = s.stm;
            pos.ply = s.ply;
            pos.key = pos.compute_key();
            // The engine refreshes the accumulators from the position before
            // the forward pass; without that the accumulator still holds
            // whatever the previous position left in it.
            nnue_refresh(net, pos);
            const float a = nnue_forward(net, pos.stm);

            Board gb;
            std::memcpy(gb.sq, s.board, jungle::NUM_SQ);
            gb.stm = s.stm;
            const float b = nnue_eval(gnet, gb.sq, gb.stm, acc.data());

            const double d = std::abs(double(a) - double(b));
            worst = std::max(worst, d);
            if (d > 1e-5) ++differing;
        }
        std::printf("network: %d positions, worst difference %.3g, differing %d\n",
                    check, worst, differing);
        if (worst > 1e-4) {
            std::printf("FAIL: the forward passes disagree\n");
            return 1;
        }
        std::printf("PASS: the forward passes agree\n");
    }
    return 0;
}
