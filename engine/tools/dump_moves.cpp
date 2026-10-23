// dump_moves.cpp -- Print the legal moves of a position.
//
// Reads FENs (one per line) on stdin and prints, for each, the sorted list of
// legal moves in UCI notation.  Used to cross-check the C++ rules against the
// Python reference implementation in backend/engine.py.
//
//   echo "E8L/9/9/9/9/9/9/9/9 r" | ./dump_moves
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../src/jungle.hpp"

using namespace jungle;

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Position pos;
        if (!pos.set_fen(line)) {
            std::printf("ERROR\n");
            continue;
        }
        Move moves[128];
        const int n = generate_moves(pos, moves);
        std::vector<std::string> names;
        for (int i = 0; i < n; ++i) names.push_back(move_to_uci(pos, moves[i]));
        std::sort(names.begin(), names.end());
        std::printf("%s |", line.c_str());
        for (const auto& s : names) std::printf(" %s", s.c_str());
        std::printf("\n");
    }
    return 0;
}
