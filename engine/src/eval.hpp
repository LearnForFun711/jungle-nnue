// eval.hpp -- Handcrafted evaluation for Jungle Chess.
//
// This is both
//   (a) the *baseline* engine the neural network has to beat, and
//   (b) the leaf evaluator used by the MCTS teacher.
//
// The score is returned from the point of view of the side to move
// (positive == good for the side to move), in the same "centipawn-ish"
// units the search uses.
#pragma once

#include <algorithm>
#include <cmath>

#include "jungle.hpp"

namespace jungle {

// Material values, scaled x100.  A Rat is worth more than a Dog because it
// can capture the Elephant, and the Elephant is the strongest piece.
constexpr int VALUE[RAT - 1 + 9] = {0, 350, 250, 300, 400, 500, 700, 800, 900};

// How much each piece type wants to advance towards the enemy den.
constexpr int ADVANCE[RAT - 1 + 9] = {0, 10, 6, 8, 10, 14, 26, 30, 34};

struct EvalTables {
    int  dist_to_den[2][NUM_SQ];   // [color][square] -> manhattan distance to own den
    int  dist_to_enemy[2][NUM_SQ]; // [color][square] -> manhattan distance to enemy den
    int  adv_rank[2][NUM_SQ];      // rows advanced into enemy territory

    EvalTables() {
        for (int s = 0; s < NUM_SQ; ++s) {
            const int r = sq_row(s), c = sq_col(s);
            dist_to_den[RED][s]    = std::abs(r - 8) + std::abs(c - 3);
            dist_to_den[BLACK][s]  = std::abs(r - 0) + std::abs(c - 3);
            dist_to_enemy[RED][s]   = std::abs(r - 0) + std::abs(c - 3);
            dist_to_enemy[BLACK][s] = std::abs(r - 8) + std::abs(c - 3);
            adv_rank[RED][s]        = 8 - r;
            adv_rank[BLACK][s]      = r;
        }
    }
};

inline const EvalTables& eval_tables() {
    static const EvalTables t;
    return t;
}

// Can the piece on `atk_sq` capture the piece on `def_sq`?
inline bool can_capture_square(const Position& pos, int atk_sq, int atk_piece,
                               int def_sq, int def_piece, int atk_color) {
    if (!atk_piece || !def_piece) return false;
    if (piece_color(def_piece) == atk_color) return false;

    int d_rank = piece_type(def_piece);
    if (terrain().trap_owner[def_sq] == atk_color) d_rank = 0;

    const bool from_river = terrain().river[atk_sq];
    const bool to_river   = terrain().river[def_sq];
    if (from_river && !to_river) return false;
    if (!from_river && to_river) return false;

    const int a_rank = piece_type(atk_piece);
    if (a_rank == RAT && d_rank == ELEPHANT) return true;
    if (a_rank == ELEPHANT && d_rank == RAT) return false;
    return a_rank >= d_rank;
}

// ---------------------------------------------------------------------------
// Main evaluation
// ---------------------------------------------------------------------------

inline int evaluate(const Position& pos) {
    const Terrain& T = terrain();
    const EvalTables& E = eval_tables();
    const int stm = pos.stm;

    struct PieceRef { int sq, piece, color, type; };
    PieceRef mine[16], theirs[16];
    int n_mine = 0, n_theirs = 0;

    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = pos.board[s];
        if (!p) continue;
        const int c = piece_color(p);
        PieceRef& ref = (c == stm) ? mine[n_mine++] : theirs[n_theirs++];
        ref.sq = s; ref.piece = p; ref.color = c; ref.type = piece_type(p);
    }

    // A side with no pieces has no legal move.  It is scored the same way
    // the search scores "no legal move", so that the two agree on how bad
    // it is -- and so that it is not mistaken for a mate.
    if (n_mine == 0) return -STUCK_SCORE;
    if (n_theirs == 0) return STUCK_SCORE;

    int score = 0;

    // -- 1. Material ------------------------------------------------------
    for (int i = 0; i < n_mine; ++i)   score += VALUE[mine[i].type];
    for (int i = 0; i < n_theirs; ++i) score -= VALUE[theirs[i].type];

    // -- 2. Trap danger ---------------------------------------------------
    // A piece standing on the opponent's trap has rank 0: literally anything
    // may take it.  This is the single biggest tactical factor in the game
    // and the original handcrafted engine ignored it completely.
    for (int side = 0; side < 2; ++side) {
        const PieceRef* list = (side == 0) ? mine : theirs;
        const int n = (side == 0) ? n_mine : n_theirs;
        const int sign = (side == 0) ? +1 : -1;
        for (int i = 0; i < n; ++i) {
            const int trap_owner = T.trap_owner[list[i].sq];
            if (trap_owner < 0 || trap_owner == list[i].color) continue;
            int danger = VALUE[list[i].type] * 6 / 10 + 150;
            // Extra urgency when an enemy piece is adjacent (it can take us now).
            for (int d = 0; d < 4; ++d) {
                const int nb = T.neighbor[list[i].sq][d];
                if (nb < 0) continue;
                const int q = pos.board[nb];
                if (!q || piece_color(q) == list[i].color) continue;
                if (can_capture_square(pos, nb, q, list[i].sq, list[i].piece,
                                       piece_color(q))) {
                    danger += 120;
                    break;
                }
            }
            score -= sign * danger;
        }
    }

    // -- 3. Attacked-but-undefended ---------------------------------------
    // For every piece we look at its (at most four) orthogonal neighbours and
    // at the enemy Tigers/Lions that could jump onto it.  Keeping this
    // neighbour-based rather than all-pairs is what makes the evaluation
    // cheap enough to run inside the MCTS inner loop.
    for (int side = 0; side < 2; ++side) {
        const PieceRef* list = (side == 0) ? mine : theirs;
        const PieceRef* opp  = (side == 0) ? theirs : mine;
        const int n = (side == 0) ? n_mine : n_theirs;
        const int no = (side == 0) ? n_theirs : n_mine;
        const int sign = (side == 0) ? +1 : -1;

        // Collect the enemy jumpers once instead of scanning every enemy
        // piece for every one of our pieces.
        const PieceRef* jumpers[4];
        int n_jumpers = 0;
        for (int k = 0; k < no && n_jumpers < 4; ++k)
            if (opp[k].type == TIGER || opp[k].type == LION) jumpers[n_jumpers++] = &opp[k];

        for (int i = 0; i < n; ++i) {
            const PieceRef& me = list[i];
            int attackers = 0, defenders = 0;

            // Orthogonal neighbours.
            for (int d = 0; d < 4; ++d) {
                const int nb = T.neighbor[me.sq][d];
                if (nb < 0) continue;
                const int q = pos.board[nb];
                if (!q) continue;
                const int qcolor = piece_color(q);
                if (qcolor == me.color) {
                    if (can_capture_square(pos, nb, q, me.sq, me.piece, qcolor))
                        ++defenders;
                } else {
                    if (can_capture_square(pos, nb, q, me.sq, me.piece, qcolor))
                        ++attackers;
                }
            }

            // Enemy Tigers and Lions that could jump onto this piece.
            for (int k = 0; k < n_jumpers; ++k) {
                const PieceRef& op = *jumpers[k];
                for (const auto& j : T.jumps[op.sq]) {
                    if (j.to != me.sq) continue;
                    bool clear = true;
                    for (int q = 0; q < j.path_len; ++q)
                        if (pos.board[j.path[q]] != 0) { clear = false; break; }
                    if (clear && can_capture_square(pos, op.sq, op.piece, me.sq, me.piece,
                                                   op.color))
                        ++attackers;
                    break;
                }
            }

            if (attackers > defenders) {
                const int exposure = (attackers - defenders);
                int penalty = VALUE[me.type] * exposure / 3 + 40 * exposure;
                // Being trapped makes any single attack lethal.
                if (T.trap_owner[me.sq] >= 0 && T.trap_owner[me.sq] != me.color)
                    penalty = std::max(penalty, VALUE[me.type] / 2);
                score -= sign * penalty;
            }
        }
    }

    // -- 4. Advancement / aggression -------------------------------------
    // The win condition is reaching the opponent's den, so a capable piece
    // gets dramatically more valuable as it approaches that den.  This is
    // what stops the engine from shuffling material back and forth.
    for (int i = 0; i < n_mine; ++i) {
        const PieceRef& p = mine[i];
        const int d = E.dist_to_enemy[stm][p.sq];
        int adv = (14 - d) * ADVANCE[p.type] / 2;
        if (p.type >= TIGER || p.type == RAT) {
            if (d <= 1)      adv += 420;   // one move from winning outright
            else if (d == 2) adv += 190;
            else if (d == 3) adv += 80;
        }
        score += adv;
    }
    for (int i = 0; i < n_theirs; ++i) {
        const PieceRef& p = theirs[i];
        const int d = E.dist_to_enemy[opposite_color(stm)][p.sq];
        int adv = (14 - d) * ADVANCE[p.type] / 2;
        if (p.type >= TIGER || p.type == RAT) {
            if (d <= 1)      adv += 420;
            else if (d == 2) adv += 190;
            else if (d == 3) adv += 80;
        }
        score -= adv;
    }

    // -- 5. Den race ------------------------------------------------------
    int my_best = 99, opp_best = 99;
    for (int i = 0; i < n_mine; ++i)
        my_best = std::min(my_best, E.dist_to_enemy[stm][mine[i].sq]);
    for (int i = 0; i < n_theirs; ++i)
        opp_best = std::min(opp_best, E.dist_to_enemy[opposite_color(stm)][theirs[i].sq]);
    if (my_best < opp_best) {
        score += (opp_best - my_best) * 70;      // we are closer: press
    } else if (opp_best < my_best) {
        score -= (my_best - opp_best) * 50;      // they are closer: defend
    }

    // -- 6. Water play ----------------------------------------------------
    for (int i = 0; i < n_mine; ++i) {
        const PieceRef& p = mine[i];
        if (p.type != RAT || !T.river[p.sq]) continue;
        // A rat in the river is immune to the enemy elephant on land.
        bool enemy_ele = false;
        for (int k = 0; k < n_theirs; ++k)
            if (theirs[k].type == ELEPHANT) { enemy_ele = true; break; }
        score += enemy_ele ? 220 : 60;
        // A rat in the river can grab an enemy rat that strays onto land next
        // to it, and it blocks the river crossing.
        for (int d = 0; d < 4; ++d) {
            const int nb = T.neighbor[p.sq][d];
            if (nb < 0) continue;
            const int q = pos.board[nb];
            if (q && piece_color(q) != stm && piece_type(q) == RAT && !T.river[nb])
                score += 90;
        }
    }
    for (int i = 0; i < n_theirs; ++i) {
        const PieceRef& p = theirs[i];
        if (p.type != RAT || !T.river[p.sq]) continue;
        bool my_ele = false;
        for (int k = 0; k < n_mine; ++k)
            if (mine[k].type == ELEPHANT) { my_ele = true; break; }
        score -= my_ele ? 220 : 60;
    }

    // -- 7. Mobility ------------------------------------------------------
    for (int i = 0; i < n_mine; ++i) {
        const PieceRef& p = mine[i];
        int mob = 0;
        for (int d = 0; d < 4; ++d) {
            const int nb = T.neighbor[p.sq][d];
            if (nb < 0) continue;
            if (step_is_legal(pos, p.sq, nb, p.piece, p.color)) ++mob;
        }
        score += mob * 8;
    }
    for (int i = 0; i < n_theirs; ++i) {
        const PieceRef& p = theirs[i];
        int mob = 0;
        for (int d = 0; d < 4; ++d) {
            const int nb = T.neighbor[p.sq][d];
            if (nb < 0) continue;
            if (step_is_legal(pos, p.sq, nb, p.piece, p.color)) ++mob;
        }
        score -= mob * 8;
    }

    // -- 8. Den defenders -------------------------------------------------
    for (int i = 0; i < n_mine; ++i) {
        const PieceRef& p = mine[i];
        const int d = E.dist_to_den[stm][p.sq];
        if (d <= 2) score += (3 - d) * 30;
    }
    for (int i = 0; i < n_theirs; ++i) {
        const PieceRef& p = theirs[i];
        const int d = E.dist_to_den[opposite_color(stm)][p.sq];
        if (d <= 2) score -= (3 - d) * 30;
    }

    return score;
}

}  // namespace jungle
