// match_game.hpp -- playing one game, and saying who won.
//
// The winner of a Jungle Chess game is not a matter of interpretation: the
// side that moves into the opponent's den wins.  This header keeps that
// fact separate from the *sign convention* used to report it, because the
// two are easy to confuse and getting them backwards silently inverts
// every match result.
//
// `GameResult::red_won` is the truth, taken from the board: it is 1 when
// the side that moved into the opponent's den was Red.  `GameResult::result`
// is the reported value, from Red's point of view: +1 Red won, 0 draw,
// -1 Black won.
//
// The two are computed from different sources on purpose -- one from the
// side that made the winning move, the other from the side to move after
// the move -- so that `engine/tools/game_check.cpp` can compare them and
// prove that the harness reports the winner as the winner.
#pragma once

#include <vector>

#include "engine_api.hpp"
#include "jungle.hpp"
#include "search.hpp"

namespace jungle {

enum End { END_DEN, END_NO_MOVES, END_REPETITION, END_LENGTH };

struct GameResult {
    int  result;   // +1 Red won, 0 draw, -1 Black won (Red's point of view)
    int  plies;
    int  cause;
    int  red_won;  // 1 if Red won -- from the board, not from a sign
};

struct Stats { long long nodes = 0; long long depth = 0; long long moves = 0; };

inline GameResult play_game(Engine& red_engine,
                             Engine& black_engine,
                             std::vector<uint64_t>& history,
                             Position& pos,
                             Stats* red_stats = nullptr,
                             Stats* black_stats = nullptr,
                             int max_ply = 200) {
    history.clear();
    history.push_back(pos.key);

    while (true) {
        Engine& engine = (pos.stm == RED) ? red_engine : black_engine;
        Stats* stats = (pos.stm == RED) ? red_stats : black_stats;
        Move m = engine.best_move(pos, history);
        if (m.from == 0xFFFF) break;
        if (stats) {
            stats->nodes += (long long)engine.last_nodes();
            stats->depth += engine.last_depth();
            ++stats->moves;
        }

        // The side that is about to move.  This is the only fact needed to
        // decide who won, and it is captured before make_move() flips the
        // side to move.
        const int mover = pos.stm;

        const bool den_entry = (terrain().den_owner[m.to] == opposite_color(pos.stm));
        make_move(pos, m);
        history.push_back(pos.key);

        if (den_entry) {
            // The mover entered the opponent's den, so the mover won.
            const int red_won = (mover == RED) ? 1 : 0;
            // Reported from Red's point of view: after the move the side to
            // move is the loser, so Red won exactly when it is Black's turn.
            const int result = (pos.stm == RED) ? -1 : +1;
            return {result, pos.ply, END_DEN, red_won};
        }

        int reps = 0;
        for (uint64_t k : history) if (k == pos.key) ++reps;
        if (reps >= 3) return {0, pos.ply, END_REPETITION, 0};
        if (int(pos.ply) >= max_ply) return {0, pos.ply, END_LENGTH, 0};
        Move buf[128];
        if (generate_moves(pos, buf) == 0) {
            // The side to move has no legal move and loses.
            const int red_won = (pos.stm == BLACK) ? 1 : 0;
            const int result = (pos.stm == RED) ? -1 : +1;
            return {result, pos.ply, END_NO_MOVES, red_won};
        }
    }
    return {0, pos.ply, END_LENGTH, 0};
}

}  // namespace jungle
