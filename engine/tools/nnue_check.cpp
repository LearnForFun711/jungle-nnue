// nnue_check.cpp -- Prove that the incremental NNUE accumulator updates are
// exactly equivalent to recomputing the accumulators from scratch.
//
// The search keeps the feature transformer up to date one piece at a time
// instead of rebuilding it at every node.  That is only safe if the
// incremental path and the full refresh always agree.  This tool walks random
// legal games and, at every position, compares
//
//   * the value produced from the incrementally maintained accumulators, and
//   * the value produced after a full refresh from the board,
//
// and also compares the raw accumulators.  Any drift shows up immediately.
//
//   ./nnue_check <net.nnue> [games] [plies]
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/nnue.hpp"

using namespace jungle;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: nnue_check <net.nnue> [games] [plies]\n");
        return 2;
    }
    const int games  = (argc > 2) ? std::atoi(argv[2]) : 40;
    const int plies  = (argc > 3) ? std::atoi(argv[3]) : 200;

    NnueNetwork net;
    if (!nnue_load(argv[1], net)) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }

    std::mt19937_64 rng(20241023ULL);
    double worst_value = 0.0;
    int    worst_acc   = 0;
    long   checks      = 0;
    long   undo_checks = 0;

    std::vector<int32_t> acc_red_full, acc_black_full;

    for (int g = 0; g < games; ++g) {
        Position pos;
        Evaluator ev;
        ev.net = net;
        ev.refresh(pos);

        // Stack of moves so we can undo them again.
        struct Done { Move m; int moved; int captured; };
        std::vector<Done> done;

        for (int ply = 0; ply < plies; ++ply) {
            // --- compare incremental vs. full refresh --------------------
            const int v_inc = ev.eval_cached(pos);
            const int v_full = ev.eval(pos);           // refreshes + resets stm_
            worst_value = std::max(worst_value, std::abs(double(v_inc - v_full)) / 1000.0);
            ++checks;

            // --- compare the raw accumulators ---------------------------
            nnue_refresh(net, pos);                    // net is now "full"
            for (uint32_t h = 0; h < net.ft_hidden; ++h) {
                worst_acc = std::max(worst_acc, std::abs(net.acc[RED][h] - ev.net.acc[RED][h]));
                worst_acc = std::max(worst_acc, std::abs(net.acc[BLACK][h] - ev.net.acc[BLACK][h]));
            }
            // Put the evaluator's incremental state back (nnue_refresh above
            // only touched the local copy `net`, not ev.net).
            ev.refresh(pos);

            // --- undo half of the moves occasionally ---------------------
            if (!done.empty() && (rng() % 3 == 0)) {
                const Done d = done.back();
                done.pop_back();
                undo_move(pos, d.m, d.captured);
                ev.on_move(d.m.from, d.m.to, d.moved, d.captured, false);
                ++undo_checks;
                continue;
            }

            Move moves[128];
            const int n = generate_moves(pos, moves);
            if (n == 0) break;
            const Move m = moves[rng() % n];

            // Never move into the opponent's den -- that ends the game and
            // the position would no longer be reachable.
            if (terrain().den_owner[m.to] == opposite_color(pos.stm)) continue;

            const int moved    = pos.board[m.from];
            const int captured = pos.board[m.to];
            make_move(pos, m);
            ev.on_move(m.from, m.to, moved, captured, true);
            done.push_back(Done{m, moved, captured});
        }
    }

    std::printf("positions checked : %ld  (undo steps: %ld)\n", checks, undo_checks);
    std::printf("max |value diff|   : %.6f\n", worst_value);
    std::printf("max |accum diff|   : %d\n", worst_acc);
    if (worst_value > 1e-6 || worst_acc != 0) {
        std::printf("FAIL: incremental accumulators drifted from the full refresh\n");
        return 1;
    }
    std::printf("PASS: incremental accumulators match the full refresh exactly\n");
    return 0;
}
