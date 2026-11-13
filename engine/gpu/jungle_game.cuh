// jungle_game.cuh -- Playing one game with the MCTS teacher.
//
// The loop is the one in engine/tools/selfplay.cpp: record the position,
// search, play the best move, and stop when the game is won, drawn or out
// of moves.  The records are written in the same 68-byte format the CPU
// generator writes, so the training pipeline does not know or care which
// generator produced the data.
//
// Like the search, the loop is a state machine: it is driven one round at a
// time, and a round may end with the game waiting for the evaluation of a
// batch of positions.
#pragma once

#include "jungle_mcts.cuh"

namespace jungle_gpu {

enum GamePhase : int {
    GAME_SEARCH = 0,   // running the MCTS
    GAME_MOVE = 1,     // the search is over: record the value and play
    GAME_OVER = 2,     // the game is finished
};

struct Game {
    Teacher t;
    GameMem* mem = nullptr;
    const Geometry* geom = nullptr;
    const MctsConfig* cfg = nullptr;

    // Which network plays which side.  In a league the two sides play with
    // different networks, and each side evaluates the positions it is about
    // to move in with its own network -- the same way the engine's own
    // evaluation works, where the value is always from the point of view of
    // the side to move.
    uint8_t red_net = 0;
    uint8_t black_net = 0;

    int phase = GAME_SEARCH;
    int result = 0;       // the result from Red's point of view
    int end_cause = END_LENGTH;

    JGPU_DEVICE void start(GameMem* memory, const Geometry* geometry, const MctsConfig* config,
               uint64_t seed, uint8_t red = 0, uint8_t black = 0) {
        mem = memory;
        geom = geometry;
        cfg = config;
        red_net = red;
        black_net = black;
        t.geom = geometry;
        t.cfg = config;
        t.mem = memory;
        t.rng = seed ? seed : 1;
        t.board.setup_start();
        mem->record_count = 0;
        mem->history_count = 0;
        mem->history[mem->history_count++] = t.board.key;
        phase = GAME_SEARCH;
        result = 0;
        end_cause = END_LENGTH;
        record_position();
        t.begin_search();
    }

    // The network that evaluates a child of the current node: the one that
    // plays the side to move in the child, which is the opponent of the side
    // to move in the parent.
    JGPU_DEVICE uint8_t leaf_net() const {
        return (t.board.stm == RED) ? black_net : red_net;
    }

    // Fill in one leaf evaluation request.  The CUDA kernel and the CPU
    // reference both go through here, so they cannot drift apart.
    JGPU_DEVICE void fill_request(Request& r, uint32_t game_index,
                                  uint16_t child_index) const {
        const Board& parent = t.board;
        r.game = game_index;
        r.child = child_index;
        r.stm = uint16_t(parent.stm ^ 1);
        r.net_id = leaf_net();
        const int from = t.child_from[child_index];
        const int to = t.child_to[child_index];
        memcpy(r.board, parent.sq, 63);
        r.board[from] = 0;
        r.board[to] = parent.sq[from];
    }

    JGPU_DEVICE void record_position() {
        if (mem->record_count >= 256) return;
        uint8_t* rec = mem->records[mem->record_count++];
        memset(rec, 0, 68);
        memcpy(rec, t.board.sq, 63);
        rec[63] = t.board.stm;          // the result bits are filled in later
        rec[64] = 0;                    // the value is filled in later
        rec[65] = uint8_t(t.board.ply & 0xFF);
        rec[66] = uint8_t((t.board.ply >> 8) & 0xFF);
        rec[67] = 0;
    }

    // Run one simulation.  Returns true when the search needs the value of
    // the expanded node, which is the part that gets batched.
    JGPU_DEVICE bool step_search() {
        if (phase != GAME_SEARCH) return false;
        if (t.sim >= cfg->sims) { phase = GAME_MOVE; return false; }
        const bool needs_batch = t.descend();
        if (!needs_batch) t.backup();
        return needs_batch;
    }

    // The search is over: record its value and play the move.
    JGPU_DEVICE void step_move() {
        if (phase != GAME_MOVE) return;
        const GpuNode& r = mem->nodes[t.root];

        // The teacher's value for the position.
        const double v = r.visits ? double(r.value_sum) / double(r.visits) : 0.0;
        int q = int(round((v + 1.0) * 127.5));
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        mem->records[mem->record_count - 1][64] = uint8_t(q);

        // The move: the child with the most visits.
        uint32_t best = kNoChild;
        uint32_t best_visits = 0;
        for (uint32_t c = r.first_child; c != kNoChild; c = mem->nodes[c].next_sibling) {
            if (mem->nodes[c].visits > best_visits) {
                best_visits = mem->nodes[c].visits;
                best = c;
            }
        }
        if (best == kNoChild) {
            // The side to move has no legal move: it loses.
            result = (t.board.stm == RED) ? -1 : +1;
            end_cause = END_NO_MOVES;
            phase = GAME_OVER;
            return;
        }
        const GpuNode& child = mem->nodes[best];
        const int from = child.from;
        const int to = child.to;

        if (geom->den_owner[to] == (t.board.stm ^ 1)) {
            // The mover reached the opponent's den: it wins.
            result = (t.board.stm == RED) ? +1 : -1;
            end_cause = END_DEN;
            phase = GAME_OVER;
            return;
        }

        apply_move(t.board, from, to);
        if (mem->history_count < 300) mem->history[mem->history_count++] = t.board.key;

        int reps = 0;
        for (int i = 0; i < mem->history_count; ++i)
            if (mem->history[i] == t.board.key) ++reps;
        if (reps >= 3) {
            result = 0;
            end_cause = END_REPETITION;
            phase = GAME_OVER;
            return;
        }
        if (int(t.board.ply) >= cfg->max_ply) {
            result = 0;
            end_cause = END_LENGTH;
            phase = GAME_OVER;
            return;
        }

        MoveBuf buf;
        if (generate_moves(t.board, *geom, buf) == 0) {
            result = (t.board.stm == RED) ? -1 : +1;
            end_cause = END_NO_MOVES;
            phase = GAME_OVER;
            return;
        }

        record_position();
        t.begin_search();
        phase = GAME_SEARCH;
    }

    // Fill in the results of the game, now that it is known.
    JGPU_DEVICE void finish_records() {
        for (int i = 0; i < mem->record_count; ++i) {
            uint8_t* rec = mem->records[i];
            const int stm = rec[63] & 1;
            const int stm_result = (stm == RED) ? result : -result;
            const uint8_t code = uint8_t(stm_result > 0 ? 2 : (stm_result == 0 ? 1 : 0));
            rec[63] = uint8_t(stm | (code << 2));
            rec[67] = uint8_t(end_cause);
        }
    }
};

}  // namespace jungle_gpu
