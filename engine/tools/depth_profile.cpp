// depth_profile.cpp -- where does the search's time go?
#include <cstdio>
#include <vector>
#include "../src/engine_api.hpp"
#include "../src/jungle.hpp"
#include "../src/nnue.hpp"
using namespace jungle;

int main(int argc, char** argv) {
    const char* net = argc > 1 ? argv[1] : "";
    Evaluator ev;
    ev.value_scale = 10.0f;
    if (*net && !nnue_load(net, ev.net)) { std::printf("cannot load %s\n", net); return 1; }
    Searcher searcher(ev, 32);
    for (double t : {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}) {
        Position pos;
        std::vector<uint64_t> history{pos.key};
        SearchLimits limits;
        limits.time_limit = t;
        limits.max_depth = 64;
        SearchResult r = searcher.search(pos, limits, history);
        std::printf("time %5.2fs -> depth %2d  nodes %8llu  last-iteration %8llu  aborted %8llu\n",
                    t, r.depth, (unsigned long long)r.nodes,
                    (unsigned long long)r.nodes_last_iteration,
                    (unsigned long long)r.nodes_aborted);
    }
    return 0;
}
