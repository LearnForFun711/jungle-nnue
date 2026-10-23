// eval_nnue.cpp -- Print the NNUE evaluation of positions.
//
// Reads FENs (one per line) on stdin and prints the network's value for each,
// from the point of view of the side to move, scaled by 1000.  Used by
// scripts/verify_nnue.py to prove that the PyTorch forward pass and the C++
// inference agree exactly.
//
//   cat positions.txt | ./eval_nnue net.nnue
#include <cstdio>
#include <iostream>
#include <string>

#include "../src/jungle.hpp"
#include "../src/nnue.hpp"

using namespace jungle;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: eval_nnue <net.nnue> [positions]\n");
        return 2;
    }
    // "--handcrafted" prints the static evaluation instead of the network's.
    const bool handcrafted = (argc > 1 && std::string(argv[1]) == "--handcrafted");

    NnueNetwork net;
    if (!handcrafted && !nnue_load(argv[1], net)) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Position pos;
        if (!pos.set_fen(line)) {
            std::printf("ERROR\n");
            continue;
        }
        if (handcrafted) {
            std::printf("%d\n", evaluate(pos));
            continue;
        }
        nnue_refresh(net, pos);
        const float v = nnue_forward(net, pos);   // side to move's perspective
        std::printf("%.6f\n", double(v));
    }
    return 0;
}
