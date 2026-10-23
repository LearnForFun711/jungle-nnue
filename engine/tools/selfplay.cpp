// selfplay.cpp -- Parallel self-play data generator.
//
// Each thread plays complete games against itself using the MCTS teacher and
// appends one record per decision position to its own shard file.  The record
// layout is shared with train/dataset.py:
//
//   offset  size  meaning
//   0       63    board: 0 empty, 1..8 Red piece (type), 9..16 Black piece
//   63      1     meta: bit0 = side to move (0 Red, 1 Black)
//                   bits 2..3 = game result from the side-to-move's view
//                                  (0 loss, 1 draw, 2 win)
//   64      1     MCTS value, quantised from [-1,1] to [0,255]
//   65      2     ply (uint16, little endian)
//   66      1     reserved
//   67      1     reserved
//
// Usage:
//   ./selfplay --games 200 --threads 8 --sims 800 --out data/data
//   ./selfplay --games 2000 --threads 8 --sims 800 --out data/data --policy
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../src/eval.hpp"
#include "../src/jungle.hpp"
#include "../src/mcts.hpp"
#include "../src/nnue.hpp"

using namespace jungle;

namespace {

constexpr size_t RECORD_SIZE = 68;

struct Options {
    int         games = 100;
    int         threads = 8;
    int         sims = 800;
    std::string out = "data/data";
    std::string nnue;
    uint64_t    seed = 12345;
    int         max_ply = 200;
    double      temperature = 0.0;
    int         temperature_moves = 0;
    bool        policy = false;
    int         rollouts = 0;
    int         leaf_depth = 1;
    double      c_puct = 1.4;
    double      widening = 1.6;
    int         leaf_agg = 1;
};

struct GameRecord {
    uint8_t board[63];
    uint8_t stm;
    uint8_t value_q;
    uint16_t ply;
};

// Sample a move from the visit distribution with temperature `t`.
Move sample_move(const MctsResult& r, double t, std::mt19937_64& rng) {
    if (r.visits.empty()) return r.best;
    std::vector<double> w;
    double sum = 0.0;
    for (const auto& v : r.visits) {
        double x = double(v.visits);
        if (t > 0) x = std::pow(x, 1.0 / t);
        if (x <= 0) continue;
        w.push_back(x);
        sum += x;
    }
    if (w.empty()) return r.best;
    double u = std::uniform_real_distribution<double>(0.0, sum)(rng);
    for (size_t i = 0; i < w.size(); ++i) {
        u -= w[i];
        if (u <= 0) return r.visits[i].move;
    }
    return r.visits.back().move;
}

// Play one game.  Returns the result from Red's point of view
// (+1 Red wins, 0 draw, -1 Black wins) and fills `records`.
// Why a game ended.
enum EndCause : uint8_t { END_DEN = 0, END_NO_MOVES = 1, END_REPETITION = 2, END_LENGTH = 3 };

int play_game(const MctsParams& params, Evaluator& ev, std::mt19937_64& rng,
              std::vector<GameRecord>& records, int max_ply, double temperature,
              int temperature_moves, std::vector<std::vector<std::pair<Move,int>>>& policy,
              uint8_t* end_cause) {
    Position pos;
    std::vector<uint64_t> history;
    history.push_back(pos.key);

    Mcts mcts(params, &ev);

    while (true) {
        // Record the decision position; the teacher's value is filled in
        // after the search has run.
        GameRecord rec;
        std::memcpy(rec.board, pos.board, 63);
        rec.stm = pos.stm;
        rec.value_q = 0;
        rec.ply = pos.ply;
        records.push_back(rec);

        MctsResult r = mcts.search(pos, rng);
        records.back().value_q = uint8_t(std::lround((r.value + 1.0) * 127.5));

        Move chosen;
        if (temperature > 0.0 && int(pos.ply) < temperature_moves)
            chosen = sample_move(r, temperature, rng);
        else
            chosen = r.best;

        if (chosen.from == 0xFFFF) break;   // no legal move: should not happen

        if (policy.size() < records.size()) policy.resize(records.size());
        policy[records.size() - 1].clear();
        for (const auto& v : r.visits)
            if (v.visits > 0) policy[records.size() - 1].push_back({v.move, v.visits});

        const bool den_entry = (terrain().den_owner[chosen.to] == opposite_color(pos.stm));
        make_move(pos, chosen);
        history.push_back(pos.key);

        // After the move the side to move is the loser of that move.
        if (den_entry) {
            *end_cause = END_DEN;
            return pos.stm == RED ? -1 : +1;   // the mover won
        }

        // Repetition?
        int reps = 0;
        for (uint64_t k : history) if (k == pos.key) ++reps;
        if (reps >= 3) { *end_cause = END_REPETITION; return 0; }

        if (int(pos.ply) >= max_ply) { *end_cause = END_LENGTH; return 0; }

        Move buf[128];
        if (generate_moves(pos, buf) == 0) {
            // The side to move has no legal move: the previous mover wins.
            *end_cause = END_NO_MOVES;
            return pos.stm == RED ? -1 : +1;
        }
    }
    *end_cause = END_LENGTH;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--games") o.games = std::stoi(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--sims") o.sims = std::stoi(next());
        else if (a == "--out") o.out = next();
        else if (a == "--nnue") o.nnue = next();
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--max-ply") o.max_ply = std::stoi(next());
        else if (a == "--temperature") o.temperature = std::stod(next());
        else if (a == "--temperature-moves") o.temperature_moves = std::stoi(next());
        else if (a == "--policy") o.policy = true;
        else if (a == "--rollouts") o.rollouts = std::stoi(next());
        else if (a == "--leaf-depth") o.leaf_depth = std::stoi(next());
        else if (a == "--c-puct") o.c_puct = std::stod(next());
        else if (a == "--widening") o.widening = std::stod(next());
        else if (a == "--leaf-agg") o.leaf_agg = std::stoi(next());
    }

    MctsParams params;
    params.simulations = o.sims;
    params.rollouts = o.rollouts;
    params.max_ply = o.max_ply;
    params.leaf_depth = o.leaf_depth;
    params.c_puct = o.c_puct;
    params.widening_k = o.widening;
    params.leaf_agg = o.leaf_agg;

    Evaluator ev;
    if (!o.nnue.empty()) {
        if (!nnue_load(o.nnue, ev.net)) {
            std::fprintf(stderr, "failed to load nnue: %s\n", o.nnue.c_str());
            return 1;
        }
        params.use_nnue = true;
    }

    const int threads = std::max(1, o.threads);
    std::atomic<int> games_done{0};
    std::atomic<int> red_wins{0}, black_wins{0}, draws{0};
    std::atomic<long long> positions{0};

    auto worker = [&](int tid) {
        std::mt19937_64 rng(o.seed + uint64_t(tid) * 0x9E3779B97F4A7C15ULL);
        const std::string shard = o.out + ".t" + std::to_string(tid) + ".bin";
        const std::string polshard = o.out + ".policy.t" + std::to_string(tid) + ".bin";
        std::FILE* f = std::fopen(shard.c_str(), "wb");
        std::FILE* pf = o.policy ? std::fopen(polshard.c_str(), "wb") : nullptr;
        if (!f) { std::fprintf(stderr, "cannot open %s\n", shard.c_str()); return; }

        std::vector<GameRecord> records;
        std::vector<std::vector<std::pair<Move,int>>> policy;
        std::vector<uint8_t> buffer;

        for (int g = 0; g < o.games; ++g) {
            records.clear();
            policy.clear();
            uint8_t end_cause = END_LENGTH;
            const int result = play_game(params, ev, rng, records, o.max_ply,
                                         o.temperature, o.temperature_moves, policy,
                                         &end_cause);

            // Now that the result is known, write the records out.
            buffer.clear();
            buffer.reserve(records.size() * RECORD_SIZE);
            for (const auto& r : records) {
                uint8_t rec[RECORD_SIZE];
                std::memset(rec, 0, RECORD_SIZE);
                std::memcpy(rec, r.board, 63);
                // The stored result must be relative to the side to move.
                const int stm_result = (r.stm == RED) ? result : -result;
                const uint8_t code = uint8_t(stm_result > 0 ? 2 : (stm_result == 0 ? 1 : 0));
                rec[63] = uint8_t(r.stm | (code << 2));
                rec[64] = r.value_q;
                rec[65] = uint8_t(r.ply & 0xFF);
                rec[66] = uint8_t((r.ply >> 8) & 0xFF);
                rec[67] = end_cause;
                buffer.insert(buffer.end(), rec, rec + RECORD_SIZE);
            }
            std::fwrite(buffer.data(), 1, buffer.size(), f);

            if (pf) {
                for (const auto& pl : policy) {
                    uint16_t n = uint16_t(pl.size());
                    std::fwrite(&n, 2, 1, pf);
                    for (const auto& mv : pl) {
                        uint16_t m = uint16_t((mv.first.from << 6) | mv.first.to);
                        uint16_t v = uint16_t(mv.second);
                        std::fwrite(&m, 2, 1, pf);
                        std::fwrite(&v, 2, 1, pf);
                    }
                }
            }

            positions += int64_t(records.size());
            if (result > 0) ++red_wins;
            else if (result < 0) ++black_wins;
            else ++draws;
            const int done = ++games_done;
            if (done % 10 == 0 || done == o.games) {
                std::fprintf(stderr,
                             "\r[thread %d] games %d/%d  positions %lld  R%d B%d D%d   ",
                             tid, done, o.games, positions.load(), red_wins.load(),
                             black_wins.load(), draws.load());
                std::fflush(stderr);
            }
        }
        std::fclose(f);
        if (pf) std::fclose(pf);
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    std::fprintf(stderr, "\ndone: %lld positions, %d red wins, %d black wins, %d draws\n",
                 positions.load(), red_wins.load(), black_wins.load(), draws.load());
    return 0;
}
