// repro.cpp -- the position where the engine reported a mate it did not play.
#include <cstdio>
#include <cstring>
#include <vector>
#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/nnue.hpp"
#include "../src/search.hpp"
using namespace jungle;

int main(int argc, char** argv) {
    Position pos;
    std::memset(pos.board, 0, NUM_SQ);
    const char* rows[9] = {"b2......",
                           ".....b3b6",
                           "b1..b4..",
                           "...b5..b8",
                           ".......",
                           ".......",
                           ".......",
                           "r8r3..r5r2r1",
                           "r6.b7...r7"};
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 7; ++c) {
            const char p = rows[r][c];
            if (p == '.') continue;
            const int rank = p - '0';
            pos.board[sq_index(r, c)] = uint8_t(p == 'b' ? rank + 8 : rank);
        }
    }
    pos.stm = BLACK;
    pos.key = pos.compute_key();

    Evaluator ev;
    ev.value_scale = 10.0f;
    if (argc > 1 && !nnue_load(argv[1], ev.net)) { std::printf("no net\n"); return 1; }

    Searcher searcher(ev, 32);
    std::vector<uint64_t> history{pos.key};
    SearchLimits limits;
    limits.time_limit = 1.0;
    limits.max_depth = 64;
    SearchResult r = searcher.search(pos, limits, history);
    std::printf("move (%d,%d)->(%d,%d)  score %d  depth %d  nodes %llu\n",
                r.move.from / 7, r.move.from % 7, r.move.to / 7, r.move.to % 7,
                r.score, r.depth, (unsigned long long)r.nodes);
    return 0;
}
