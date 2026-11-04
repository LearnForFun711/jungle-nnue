// gpu_teacher_probe.cpp -- The portable teacher's values on the positions the
// engine's teacher_probe uses.
//
// The GPU teacher is a port of the CPU teacher, and a port is only worth
// anything if it produces the same numbers.  This tool probes the same
// positions, in the same order, with the same options, and prints the same
// columns, so that the two teachers can be compared value by value:
//
//     value,leaf_value,result,ply,stored
//
// With the noise at the root turned off (--dirichlet-eps 0) both teachers
// are deterministic, and the values should agree to within the difference
// between the engine's double arithmetic and the GPU's single.
//
//   ./gpu_teacher_probe --shard data/round1/data.t0.bin --positions 2000 \
//                        --sims 800 --net net/net.nnue --out gpu.csv
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "jungle_game.cuh"

using namespace jungle_gpu;

namespace {

constexpr size_t RECORD_SIZE = 68;

struct Options {
    std::vector<std::string> shards;
    int    positions      = 2000;
    int    sims           = 800;
    double leaf_scale     = 1.0;
    double value_scale    = 10.0;
    double c_puct         = 1.4;
    double widening       = 1.6;
    double dirichlet_alpha = 0.30;
    double dirichlet_eps  = 0.25;
    int    max_ply        = 200;
    uint64_t seed         = 7;
    std::string net;
    std::string out;
};

std::vector<uint8_t> read_shard(const std::string& path, size_t& n_records) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> raw(size > 0 ? size : 0);
    if (size > 0 && std::fread(raw.data(), 1, size, f) != size_t(size))
        std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::fclose(f);
    n_records = raw.size() / RECORD_SIZE;
    raw.resize(n_records * RECORD_SIZE);
    return raw;
}

// The same sampling the engine's teacher_probe uses, so that the two tools
// probe the same positions.
std::vector<size_t> pick_positions(size_t total, int want, uint64_t seed) {
    std::vector<size_t> pick;
    if (want >= int(total)) {
        pick.resize(total);
        for (size_t i = 0; i < total; ++i) pick[i] = i;
        return pick;
    }
    std::mt19937_64 rng(seed);
    std::vector<size_t> all_idx(total);
    for (size_t i = 0; i < total; ++i) all_idx[i] = i;
    std::shuffle(all_idx.begin(), all_idx.end(), rng);
    pick.assign(all_idx.begin(), all_idx.begin() + want);
    std::sort(pick.begin(), pick.end());
    return pick;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--shard") o.shards.push_back(next());
        else if (a == "--positions") o.positions = std::atoi(next());
        else if (a == "--sims") o.sims = std::atoi(next());
        else if (a == "--leaf-scale") o.leaf_scale = std::stod(next());
        else if (a == "--value-scale") o.value_scale = std::stod(next());
        else if (a == "--c-puct") o.c_puct = std::stod(next());
        else if (a == "--widening") o.widening = std::stod(next());
        else if (a == "--dirichlet-alpha") o.dirichlet_alpha = std::stod(next());
        else if (a == "--dirichlet-eps") o.dirichlet_eps = std::stod(next());
        else if (a == "--max-ply") o.max_ply = std::atoi(next());
        else if (a == "--seed") o.seed = std::strtoull(next(), nullptr, 10);
        else if (a == "--net") o.net = next();
        else if (a == "--out") o.out = next();
    }
    if (o.shards.empty()) {
        std::fprintf(stderr, "usage: gpu_teacher_probe --shard <file> [--net <file>]\n");
        return 2;
    }

    Net net;
    if (!o.net.empty() && !load_net(o.net.c_str(), net)) {
        std::fprintf(stderr, "cannot load %s\n", o.net.c_str());
        return 1;
    }

    std::vector<uint8_t> all;
    for (const std::string& s : o.shards) {
        size_t n = 0;
        std::vector<uint8_t> raw = read_shard(s, n);
        std::fprintf(stderr, "shard %s: %zu records\n", s.c_str(), n);
        all.insert(all.end(), raw.begin(), raw.end());
    }
    const size_t total = all.size() / RECORD_SIZE;
    if (total == 0) { std::fprintf(stderr, "no records\n"); return 1; }

    const std::vector<size_t> pick = pick_positions(total, o.positions, o.seed);

    const Geometry geom = build_geometry();
    MctsConfig cfg;
    cfg.sims = o.sims;
    cfg.leaf_scale = o.leaf_scale;
    cfg.value_scale = o.value_scale;
    cfg.c_puct = o.c_puct;
    cfg.widening_k = o.widening;
    cfg.dirichlet_alpha = o.dirichlet_alpha;
    cfg.dirichlet_eps = o.dirichlet_eps;
    cfg.max_ply = o.max_ply;

    std::vector<int32_t> acc(size_t(2) * net.ft_hidden);
    std::vector<float> child_values(MAX_CHILDREN);
    std::vector<uint8_t> child_board(63);

    GameMem mem;
    Teacher teacher;
    teacher.geom = &geom;
    teacher.cfg = &cfg;
    teacher.mem = &mem;

    struct Row { double value, leaf_value, result, stored; uint16_t ply; };
    std::vector<Row> rows;
    rows.reserve(pick.size());

    for (size_t idx : pick) {
        const uint8_t* rec = all.data() + idx * RECORD_SIZE;

        Board board;
        std::memset(board.sq, 0, sizeof(board.sq));
        std::memcpy(board.sq, rec, 63);
        board.stm = uint8_t(rec[63] & 1);
        board.ply = uint16_t(rec[65] | (rec[66] << 8));
        board.key = board.compute_key();

        teacher.board = board;
        teacher.rng = o.seed * 0x9E3779B97F4A7C15ULL + idx + 1;
        teacher.begin_search();

        for (int s = 0; s < o.sims; ++s) {
            if (!teacher.descend()) { teacher.backup(); continue; }
            const Board& parent = teacher.board;
            for (int i = 0; i < teacher.nchildren; ++i) {
                std::memcpy(child_board.data(), parent.sq, 63);
                const int from = teacher.child_from[i];
                const int to = teacher.child_to[i];
                child_board[from] = 0;
                child_board[to] = parent.sq[from];
                child_values[i] = teacher_leaf_value(
                    nnue_eval(net, child_board.data(), parent.stm ^ 1, acc.data()),
                    float(cfg.leaf_scale), float(cfg.value_scale));
            }
            teacher.finish_expansion(child_values.data());
            teacher.backup();
        }

        const GpuNode& r = mem.nodes[teacher.root];
        Row row;
        row.value = r.visits ? double(r.value_sum) / double(r.visits) : 0.0;
        row.leaf_value = double(r.leaf);
        const int code = (rec[63] >> 2) & 3;
        row.result = (code == 0) ? -1.0 : (code == 1 ? 0.0 : 1.0);
        row.stored = double(rec[64]) / 127.5 - 1.0;
        row.ply = board.ply;
        rows.push_back(row);
    }

    std::FILE* out = stdout;
    if (!o.out.empty()) {
        out = std::fopen(o.out.c_str(), "w");
        if (!out) { std::fprintf(stderr, "cannot write %s\n", o.out.c_str()); return 1; }
    }
    std::fprintf(out, "value,leaf_value,result,ply,stored\n");
    for (const Row& r : rows)
        std::fprintf(out, "%.6f,%.6f,%.1f,%u,%.6f\n", r.value, r.leaf_value, r.result,
                     unsigned(r.ply), r.stored);
    if (out != stdout) std::fclose(out);

    std::fprintf(stderr, "probed %zu positions\n", rows.size());
    return 0;
}
