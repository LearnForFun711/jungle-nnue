// game_check.cpp -- does the harness report the winner, or the loser?
//
//   ./game_check --games 30 --time 0.2 [--nnue net.nnue]
//
// The match harness used to invert every result: the side that entered the
// opponent's den won, but the harness recorded the *other* side as the
// winner.  Every match number the project produced before this tool existed
// was therefore upside down, and nothing in the output said so.
//
// This tool plays games through the very same code the harness uses --
// `play_game()` in src/match_game.hpp, the same function `match` calls --
// and compares the result it reports against the winner taken from the
// board: the side that moved into the opponent's den.  The two are computed
// from different sources on purpose, so the comparison is a real check and
// not a restatement.
//
// The den-entry rule is not a matter of interpretation, so the number is
// either 100% or the harness is wrong.
#include <cstdio>
#include <string>
#include <vector>

#include "../src/engine_api.hpp"
#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/match_game.hpp"
#include "../src/nnue.hpp"
#include "../src/search.hpp"

using namespace jungle;

int main(int argc, char** argv) {
    int games = 30;
    double time_limit = 0.2;
    std::string nnue;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--games") games = std::stoi(next());
        else if (a == "--time") time_limit = std::stod(next());
        else if (a == "--nnue") nnue = next();
    }

    EngineConfig cfg;
    cfg.time_limit = time_limit;
    if (!nnue.empty()) cfg.nnue_path = nnue;

    Engine red_engine(cfg), black_engine(cfg);

    int agree = 0, disagree = 0, draws = 0;
    int den = 0, no_moves = 0;

    for (int g = 0; g < games; ++g) {
        Position pos;
        std::vector<uint64_t> history;
        GameResult r = play_game(red_engine, black_engine, history, pos);

        if (r.cause == END_REPETITION || r.cause == END_LENGTH) { ++draws; continue; }
        if (r.cause == END_DEN) ++den;
        if (r.cause == END_NO_MOVES) ++no_moves;

        // The board says Red won; the harness reports Red winning as +1.
        const int from_the_board = r.red_won ? +1 : -1;
        if (r.result == from_the_board) ++agree;
        else ++disagree;
    }

    std::printf("games                              %d\n", games);
    std::printf("decisive                           %d   (den %d, no-moves %d)\n",
                agree + disagree, den, no_moves);
    std::printf("draws                              %d\n", draws);
    std::printf("harness agrees with the board      %d\n", agree);
    std::printf("harness disagrees with the board  %d\n", disagree);
    if (disagree) {
        std::printf("\n==> FAIL: the harness records the loser as the winner\n");
        return 1;
    }
    std::printf("\n==> PASS: the harness records the winner as the winner\n");
    return 0;
}
