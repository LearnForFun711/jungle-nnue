// nnue_ref.cpp -- The CPU's forward pass, on positions the engine's own rules
// produced.
//
// The GPU's forward pass has to give the same numbers as the engine's, and
// the engine's forward pass is the one the network was trained against.  To
// check the GPU, it needs positions to check it on, and the positions have
// to be real Jungle positions -- a random board would not exercise the
// features the way a game does.
//
// So this tool plays random games with the engine's rules, writes the
// positions out, and prints the value the CPU's forward pass gives for
// each one:
//
//   ./nnue_ref --net net/net.nnue --positions 20000 --seed 7 --out positions.bin
//
// The file holds one record per position: 63 bytes of board, one byte of
// side to move, and the four bytes of the value the CPU gave it.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../src/jungle.hpp"
#include "../src/nnue.hpp"

using namespace jungle;

int main(int argc, char** argv) {
    const char* net_path = nullptr;
    const char* out_path = nullptr;
    int wanted = 20000;
    uint64_t seed = 7;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--net") net_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--positions") wanted = std::atoi(next());
        else if (a == "--seed") seed = std::strtoull(next(), nullptr, 10);
    }

    NnueNetwork net;
    bool have_net = false;
    if (net_path) {
        if (!nnue_load(net_path, net)) {
            std::fprintf(stderr, "cannot load %s\n", net_path);
            return 1;
        }
        have_net = true;
    }

    std::FILE* out = out_path ? std::fopen(out_path, "wb") : nullptr;
    if (out_path && !out) {
        std::fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }

    std::mt19937_64 rng(seed);
    Position pos;
    Move moves[128];
    int produced = 0;
    std::vector<uint8_t> record(68);

    while (produced < wanted) {
        pos.reset();
        for (int ply = 0; ply < 200 && produced < wanted; ++ply) {
            std::memcpy(record.data(), pos.board, NUM_SQ);
            record[63] = pos.stm;

            float value = 0.f;
            if (have_net) {
                nnue_refresh(net, pos);
                value = nnue_forward(net, pos.stm);
            }
            std::memcpy(record.data() + 64, &value, 4);

            if (out) std::fwrite(record.data(), 1, record.size(), out);
            std::printf("%.9g\n", double(value));
            ++produced;

            const int n = generate_moves(pos, moves);
            if (n == 0) break;
            make_move(pos, moves[rng() % n]);
        }
    }

    if (out) std::fclose(out);
    std::fprintf(stderr, "wrote %d positions\n", produced);
    return 0;
}
