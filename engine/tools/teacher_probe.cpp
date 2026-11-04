// teacher_probe.cpp -- Measure the quality of the MCTS teacher.
//
// The teacher produces the values the network is trained on, so the quality of
// those values is the ceiling on the quality of the network.  This tool takes
// positions that already came out of a self-play game -- positions whose game
// result is known -- runs the teacher on them with a given configuration, and
// prints one row per position:
//
//     value,leaf_value,result,ply,stored
//
//   value      the teacher's root value: the visit-weighted average over the
//              tree, in [-1,1]
//   leaf_value the root's own leaf evaluation -- the position evaluated where
//              it stands, without the tree's sign flips and averaging
//   result  the result of the game the position came from, from the side to
//           move's point of view: -1 loss, 0 draw, +1 win
//   ply     how deep into the game the position was
//   stored  the value the teacher gave the position when the game was played;
//           it is only printed so that a re-run with the same configuration can
//           be checked against the original
//
// The correlation between `value` and `result` is the number that says whether
// a teacher configuration is worth generating data with.
//
// Usage:
//   ./teacher_probe --shard data/round1/data.t0.bin --positions 4000 \
//                   --sims 800 --leaf-depth 1 --leaf-agg 1 --out probe.csv
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
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
    std::vector<std::string> shards;
    int    positions   = 2000;
    int    threads     = 4;
    int    sims        = 800;
    int    leaf_depth  = 1;
    int    leaf_agg    = 1;
    double leaf_scale  = 1.0;
    double hybrid      = 0.0;
    double c_puct      = 1.4;
    double widening    = 1.6;
    double dirichlet_alpha = 0.30;
    double dirichlet_eps   = 0.25;
    uint64_t seed      = 7;
    std::string nnue;
    std::string out;
    std::string sample_out;   // the sampled records, verbatim
    std::string value_out;    // the same records, with the teacher's value
                               // replaced by the one measured here
};

struct Probe {
    float    value;       // the root's visit-weighted average
    float    leaf_value;  // the root's own leaf evaluation
    float    result;
    uint16_t ply;
    float    stored;
};

// Read one shard and return the records it holds.
std::vector<uint8_t> read_shard(const std::string& path, size_t& n_records) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> raw(size > 0 ? size : 0);
    if (size > 0 && std::fread(raw.data(), 1, size, f) != size_t(size)) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
    }
    std::fclose(f);
    n_records = raw.size() / RECORD_SIZE;
    raw.resize(n_records * RECORD_SIZE);
    return raw;
}

// Rebuild the position a record was taken from.
Position position_of(const uint8_t* rec) {
    Position pos;
    std::memset(pos.board, 0, NUM_SQ);
    std::memcpy(pos.board, rec, NUM_SQ);
    pos.stm = uint8_t(rec[63] & 1);
    pos.ply = uint16_t(rec[65] | (rec[66] << 8));
    pos.key = pos.compute_key();
    return pos;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--shard") o.shards.push_back(next());
        else if (a == "--positions") o.positions = std::stoi(next());
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--sims") o.sims = std::stoi(next());
        else if (a == "--leaf-depth") o.leaf_depth = std::stoi(next());
        else if (a == "--leaf-agg") o.leaf_agg = std::stoi(next());
        else if (a == "--leaf-scale") o.leaf_scale = std::stod(next());
        else if (a == "--hybrid") o.hybrid = std::stod(next());
        else if (a == "--sample-out") o.sample_out = next();
        else if (a == "--value-out") o.value_out = next();
        else if (a == "--c-puct") o.c_puct = std::stod(next());
        else if (a == "--widening") o.widening = std::stod(next());
        else if (a == "--dirichlet-alpha") o.dirichlet_alpha = std::stod(next());
        else if (a == "--dirichlet-eps") o.dirichlet_eps = std::stod(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--nnue") o.nnue = next();
        else if (a == "--out") o.out = next();
    }
    if (o.shards.empty()) {
        std::fprintf(stderr, "usage: teacher_probe --shard <file> [--shard <file> ...]\n");
        return 2;
    }

    // Load every shard into one flat array of records.
    std::vector<uint8_t> all;
    for (const std::string& s : o.shards) {
        size_t n = 0;
        std::vector<uint8_t> raw = read_shard(s, n);
        std::fprintf(stderr, "shard %s: %zu records\n", s.c_str(), n);
        all.insert(all.end(), raw.begin(), raw.end());
    }
    const size_t total = all.size() / RECORD_SIZE;
    if (total == 0) { std::fprintf(stderr, "no records\n"); return 1; }

    // Sample the positions to probe.  A fixed stride keeps the sample spread
    // over the whole game instead of clustering in the opening, and the seed
    // keeps the sample the same between configurations.
    std::vector<size_t> pick;
    const int want = std::min<int>(o.positions, int(total));
    if (want >= int(total)) {
        pick.resize(total);
        for (size_t i = 0; i < total; ++i) pick[i] = i;
    } else {
        std::mt19937_64 rng(o.seed);
        std::vector<size_t> all_idx(total);
        for (size_t i = 0; i < total; ++i) all_idx[i] = i;
        std::shuffle(all_idx.begin(), all_idx.end(), rng);
        pick.assign(all_idx.begin(), all_idx.begin() + want);
        std::sort(pick.begin(), pick.end());
    }

    MctsParams params;
    params.simulations = o.sims;
    params.leaf_depth  = o.leaf_depth;
    params.leaf_agg    = o.leaf_agg;
    params.leaf_scale  = o.leaf_scale;
    params.hybrid      = o.hybrid;
    params.c_puct      = o.c_puct;
    params.widening_k  = o.widening;
    params.dirichlet_alpha = o.dirichlet_alpha;
    params.dirichlet_eps   = o.dirichlet_eps;

    Evaluator ev_template;
    if (!o.nnue.empty()) {
        if (!nnue_load(o.nnue, ev_template.net)) {
            std::fprintf(stderr, "failed to load nnue: %s\n", o.nnue.c_str());
            return 1;
        }
        params.use_nnue = true;
    }

    const int threads = std::max(1, o.threads);
    std::vector<Probe> probes(pick.size());
    std::atomic<size_t> next_index{0};

    auto worker = [&](int) {
        std::mt19937_64 rng(o.seed * 0x9E3779B97F4A7C15ULL);
        // Each thread needs its own evaluator.  The network's feature
        // accumulators are changed while a position is evaluated, so one
        // evaluator shared between threads hands out values that belong to
        // another thread's position -- and the measurement then says more
        // about the race than about the teacher.
        Evaluator ev = ev_template;
        Mcts mcts(params, &ev);
        while (true) {
            const size_t i = next_index.fetch_add(1);
            if (i >= pick.size()) break;
            const uint8_t* rec = all.data() + pick[i] * RECORD_SIZE;
            Position pos = position_of(rec);

            const int code = (rec[63] >> 2) & 3;
            const float result = (code == 0) ? -1.0f : (code == 1 ? 0.0f : 1.0f);
            const float stored = float(rec[64]) / 127.5f - 1.0f;

            const MctsResult r = mcts.search(pos, rng);
            probes[i].value  = float(r.value);
            probes[i].leaf_value = float(r.leaf_value);
            probes[i].result = result;
            probes[i].ply    = pos.ply;
            probes[i].stored = stored;
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    // Write the sampled positions out as a dataset, so that two teachers can
    // be compared on exactly the same positions: the sample keeps the board,
    // the side to move, the result of the game and the ply, and only the
    // teacher's value differs.
    auto write_shard = [&](const std::string& path, bool with_new_value) {
        if (path.empty()) return;
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return; }
        std::vector<uint8_t> buf;
        buf.reserve(pick.size() * RECORD_SIZE);
        for (size_t i = 0; i < pick.size(); ++i) {
            const uint8_t* rec = all.data() + pick[i] * RECORD_SIZE;
            buf.insert(buf.end(), rec, rec + RECORD_SIZE);
            if (with_new_value) {
                const int q = int(std::lround((double(probes[i].value) + 1.0) * 127.5));
                buf[buf.size() - RECORD_SIZE + 64] = uint8_t(q < 0 ? 0 : (q > 255 ? 255 : q));
            }
        }
        if (!buf.empty()) std::fwrite(buf.data(), 1, buf.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s (%zu records)\n", path.c_str(), pick.size());
    };
    write_shard(o.sample_out, false);
    write_shard(o.value_out, true);

    std::FILE* out = stdout;
    if (!o.out.empty()) {
        out = std::fopen(o.out.c_str(), "w");
        if (!out) { std::fprintf(stderr, "cannot write %s\n", o.out.c_str()); return 1; }
    }
    std::fprintf(out, "value,leaf_value,result,ply,stored\n");
    for (const Probe& p : probes)
        std::fprintf(out, "%.6f,%.6f,%.1f,%u,%.6f\n", p.value, p.leaf_value, p.result,
                     unsigned(p.ply), p.stored);
    if (out != stdout) std::fclose(out);

    std::fprintf(stderr, "probed %zu positions\n", probes.size());
    return 0;
}
