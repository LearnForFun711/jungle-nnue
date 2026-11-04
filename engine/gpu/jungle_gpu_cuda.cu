// jungle_gpu_cuda.cu -- The self-play generator, on the GPU.
//
// One thread plays one game.  The trees live in the games' own memory, so
// the threads do not fight over the same cache lines, and the network's leaf
// evaluations are batched across every game a warp is playing: when a game
// expands a node it needs the value of every child, and the whole warp
// collects its requests, evaluates them together and picks the answers up.
//
// The network's forward pass is the engine's own arithmetic, arranged so
// that a warp evaluates one position at a time: the feature accumulators
// live in the lanes' registers, and the two hidden layers read the
// activations out of shared memory.  That is what makes it fast -- a warp
// evaluating a position on its own would leave thirty-one lanes idle.
//
// Everything inside a warp is synchronised with __syncwarp(), never with
// __syncthreads(): the warps in a block are working on different positions
// and reach their barriers at different times.
//
// A lane that finishes its game starts the next one straight away, so no
// warp waits for its slowest game.
//
// The generated records are the same 68 bytes the CPU generator writes, so
// nothing downstream has to know where the data came from.
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "jungle_game.cuh"

using namespace jungle_gpu;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err = (expr);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
        }                                                                       \
    } while (0)

// One request: whose child it is, which child, whose turn it is, and the
// board itself.
struct Request {
    uint32_t game;
    uint16_t child;
    uint16_t stm;
    uint8_t  board[63];
};

// The requests a warp can hold in one round.  A warp has thirty-two lanes
// and a position has at most ninety-six legal moves, so 32 * 128 is enough.
constexpr int WARP_QUEUE = 32 * 128;

// The records one game can hold.  The length limit is two hundred plies.
constexpr int MAX_RECORDS = 256;
constexpr int RECORD_SIZE = 68;

// ---------------------------------------------------------------------------
// The network, one position at a time, with a warp
// ---------------------------------------------------------------------------

// The number of accumulator slots each lane holds: the feature transformer's
// width divided by the width of a warp, rounded up.  It has to be a compile
// time constant for the accumulator to stay in the registers.
template <int CHUNKS>
__device__ __forceinline__ float nnue_eval_warp(const Net& net, const uint8_t* board,
                                                 int stm, float* shared_ft,
                                                 float* shared_h0, float* shared_h1) {
    const int FH = net.ft_hidden;
    const int H = net.hidden;
    const int lane = threadIdx.x & 31;

    int32_t acc_r[CHUNKS], acc_b[CHUNKS];
#pragma unroll
    for (int k = 0; k < CHUNKS; ++k) { acc_r[k] = 0; acc_b[k] = 0; }

    // The feature accumulators.  The lane owns the hidden units
    // lane, lane + 32, ... and the piece loop is the same for every lane,
    // because they all read the same board.
    for (int s = 0; s < NUM_SQ; ++s) {
        const int p = board[s];
        if (!p) continue;
        const int f_red = feature_index(s, p, RED);
        const int f_black = feature_index(s, p, BLACK);
        const int8_t* row_red = net.ft_w + size_t(f_red) * FH;
        const int8_t* row_black = net.ft_w + size_t(f_black) * FH;
#pragma unroll
        for (int k = 0; k < CHUNKS; ++k) {
            const int h = lane + 32 * k;
            if (h < FH) {
                acc_r[k] += row_red[h];
                acc_b[k] += row_black[h];
            }
        }
    }

    // The feature transformer, into shared memory.
    __syncwarp();
#pragma unroll
    for (int k = 0; k < CHUNKS; ++k) {
        const int h = lane + 32 * k;
        if (h < FH) {
            shared_ft[h] = clip_to_127(float(acc_r[k]) * net.ft_scale[h] + net.ft_bias[h]);
            shared_ft[FH + h] = clip_to_127(float(acc_b[k]) * net.ft_scale[h] + net.ft_bias[h]);
        }
    }
    __syncwarp();

    // The first hidden layer.  The lane owns the outputs lane, lane + 32, ...
    constexpr int MAX_OUT = 8;
    const int h0_per_lane = (H + 31) / 32;
    int8_t h0_r[MAX_OUT], h0_b[MAX_OUT];
#pragma unroll
    for (int j = 0; j < MAX_OUT; ++j) {
        const int o = lane + 32 * j;
        const bool mine = (j < h0_per_lane) && (o < H);
        int32_t d_r = 0, d_b = 0;
        if (mine) {
            const int8_t* w = net.h_w[0] + size_t(o) * FH;
            for (int i = 0; i < FH; ++i) {
                d_r += int32_t(w[i]) * int32_t(shared_ft[i]);
                d_b += int32_t(w[i]) * int32_t(shared_ft[FH + i]);
            }
        }
        // The scaling is only read for the outputs the lane owns: reading
        // past the end of the scale array would be a read outside the
        // network, and the network is not that big.
        const float sc = mine ? net.h_scale[0][o] : 0.0f;
        const float bi = mine ? net.h_bias[0][o] : 0.0f;
        h0_r[j] = int8_t(clip_to_127(float(d_r) * sc + bi));
        h0_b[j] = int8_t(clip_to_127(float(d_b) * sc + bi));
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < MAX_OUT; ++j) {
        const int o = lane + 32 * j;
        if ((j < h0_per_lane) && (o < H)) {
            shared_h0[o] = float(h0_r[j]);
            shared_h0[H + o] = float(h0_b[j]);
        }
    }
    __syncwarp();

    // The second hidden layer.
    const int h1_per_lane = (H + 31) / 32;
    int8_t h1_r[MAX_OUT], h1_b[MAX_OUT];
#pragma unroll
    for (int j = 0; j < MAX_OUT; ++j) {
        const int o = lane + 32 * j;
        const bool mine = (j < h1_per_lane) && (o < H);
        int32_t d_r = 0, d_b = 0;
        if (mine) {
            const int8_t* w = net.h_w[1] + size_t(o) * H;
            for (int i = 0; i < H; ++i) {
                d_r += int32_t(w[i]) * int32_t(shared_h0[i]);
                d_b += int32_t(w[i]) * int32_t(shared_h0[H + i]);
            }
        }
        const float sc = mine ? net.h_scale[1][o] : 0.0f;
        const float bi = mine ? net.h_bias[1][o] : 0.0f;
        h1_r[j] = int8_t(clip_to_127(float(d_r) * sc + bi));
        h1_b[j] = int8_t(clip_to_127(float(d_b) * sc + bi));
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < MAX_OUT; ++j) {
        const int o = lane + 32 * j;
        if ((j < h1_per_lane) && (o < H)) {
            shared_h1[o] = float(h1_r[j]);
            shared_h1[H + o] = float(h1_b[j]);
        }
    }
    __syncwarp();

    // The output.  The two perspectives are ordered by the side to move, the
    // same way the engine does it, so the value comes out from the side to
    // move's point of view.
    const float* first = (stm == RED) ? &shared_h1[0] : &shared_h1[H];
    const float* second = (stm == RED) ? &shared_h1[H] : &shared_h1[0];
    float sum = 0.f;
    for (int i = lane; i < H; i += 32) {
        sum += float(net.out_w[i]) * first[i];
        sum += float(net.out_w[H + i]) * second[i];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    return sum;
}

// The value of one position, from the side to move's point of view.  The
// whole warp works on it; the answer is in lane 0.
__device__ __forceinline__ float nnue_eval_batch(const Net& net, const uint8_t* board,
                                                  int stm, float* shared_ft, float* shared_h0,
                                                  float* shared_h1) {
    const int FH = net.ft_hidden;
    const int chunks = (FH + 31) / 32;
    float sum = 0.f;
    switch (chunks) {
        case 1:  sum = nnue_eval_warp<1>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 2:  sum = nnue_eval_warp<2>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 3:  sum = nnue_eval_warp<3>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 4:  sum = nnue_eval_warp<4>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 5:  sum = nnue_eval_warp<5>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 6:  sum = nnue_eval_warp<6>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 7:  sum = nnue_eval_warp<7>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 8:  sum = nnue_eval_warp<8>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 9:  sum = nnue_eval_warp<9>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 10: sum = nnue_eval_warp<10>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 11: sum = nnue_eval_warp<11>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 12: sum = nnue_eval_warp<12>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 13: sum = nnue_eval_warp<13>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 14: sum = nnue_eval_warp<14>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 15: sum = nnue_eval_warp<15>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        case 16: sum = nnue_eval_warp<16>(net, board, stm, shared_ft, shared_h0, shared_h1); break;
        default: sum = 0.f; break;
    }
    return sum * net.out_scale + net.out_bias;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

__global__ void selfplay_kernel(GameMem* mems, const Net* net, const Geometry* geom,
                                 const MctsConfig* cfg, int num_games, uint64_t seed,
                                 Request* queue, float* values,
                                 uint8_t* out_records, int* out_counts) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    // Two different warps: the one inside the block picks the slice of the
    // shared memory, the one across the whole grid picks the slice of the
    // request queue.  Using the grid-wide warp number for the shared memory
    // would put every block's warps past the end of its own shared memory.
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    // The shared memory the network's forward pass needs, one slice per warp:
    // the feature transformer output and the two hidden layers.
    extern __shared__ char smem[];
    const int FH = net->ft_hidden;
    const int H = net->hidden;
    float* slice = reinterpret_cast<float*>(smem) + size_t(warp_in_block) * (2 * FH + 4 * H);
    float* shared_ft = slice;
    float* shared_h0 = slice + 2 * FH;
    float* shared_h1 = slice + 2 * FH + 2 * H;

    Game game;
    int game_index = gid;
    bool have_game = game_index < num_games;
    if (have_game)
        game.start(&mems[game_index], geom, cfg,
                   seed * 0x9E3779B97F4A7C15ULL + uint64_t(game_index) + 1);

    Request* my_queue = queue + size_t(warp_global) * WARP_QUEUE;

    while (true) {
        // The loop has to be the same length for every lane of the warp: a
        // game that ends early simply stops doing work, but it still comes
        // to every barrier.
        const bool active = have_game && (game.phase != GAME_OVER);
        if (!__any_sync(0xffffffff, active)) break;

        // ---- the tree work: one simulation
        const bool need = active && game.step_search();

        // ---- the batch: what this lane needs evaluated
        const int my_count = need ? game.t.nchildren : 0;
        // An exclusive scan over the warp: every lane needs to know where
        // its own requests start in the queue.
        int offset = my_count;
        for (int step = 1; step < 32; step <<= 1) {
            const int other = __shfl_up_sync(0xffffffff, offset, step);
            if (lane >= step) offset += other;
        }
        const int total = __shfl_sync(0xffffffff, offset, 31);
        const int base = offset - my_count;

        if (need) {
            const Board& parent = game.t.board;
            for (int i = 0; i < my_count; ++i) {
                Request& r = my_queue[base + i];
                r.game = uint32_t(game_index);
                r.child = uint16_t(i);
                r.stm = uint16_t(parent.stm ^ 1);
                const int from = game.t.child_from[i];
                const int to = game.t.child_to[i];
                memcpy(r.board, parent.sq, 63);
                r.board[from] = 0;
                r.board[to] = parent.sq[from];
            }
        }
        __syncwarp();

        // ---- the network, one position at a time, the whole warp on each
        for (int p = 0; p < total; ++p) {
            const Request& r = my_queue[p];
            const float value = nnue_eval_batch(*net, r.board, int(r.stm),
                                                shared_ft, shared_h0, shared_h1);
            if (lane == 0) values[size_t(r.game) * MAX_CHILDREN + r.child] = value;
            __syncwarp();
        }
        __syncwarp();

        // ---- the answer
        if (need) {
            game.t.finish_expansion(&values[size_t(game_index) * MAX_CHILDREN]);
            game.t.backup();
        }
        if (active && game.phase == GAME_MOVE) game.step_move();

        // ---- the game is over: write the records out and start the next one
        if (active && game.phase == GAME_OVER) {
            game.finish_records();
            GameMem* m = &mems[game_index];
            const int n = m->record_count;
            memcpy(out_records + size_t(game_index) * MAX_RECORDS * RECORD_SIZE,
                   m->records[0], size_t(n) * RECORD_SIZE);
            out_counts[game_index] = n;
            game_index += 32;
            have_game = game_index < num_games;
            if (have_game)
                game.start(&mems[game_index], geom, cfg,
                           seed * 0x9E3779B97F4A7C15ULL + uint64_t(game_index) + 1);
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// A batch evaluation on its own, so that the GPU's forward pass can be
// checked against the engine's.
// ---------------------------------------------------------------------------

__global__ void nnue_batch_kernel(const Net* net, const uint8_t* boards, const int* stms,
                                   int count, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;

    extern __shared__ char smem[];
    const int FH = net->ft_hidden;
    const int H = net->hidden;
    float* slice = reinterpret_cast<float*>(smem) + size_t(warp_in_block) * (2 * FH + 4 * H);
    float* shared_ft = slice;
    float* shared_h0 = slice + 2 * FH;
    float* shared_h1 = slice + 2 * FH + 2 * H;

    // The lanes past the end of the batch must not leave the kernel: the
    // forward pass synchronises the warp, and a warp whose lanes have gone
    // away is a warp whose barriers are not defined.  So the extra lanes
    // evaluate a position that is there and their answers are dropped.
    const bool valid = (i < count);
    const int slot = valid ? i : 0;
    const float value = nnue_eval_batch(*net, boards + size_t(slot) * 63, stms[slot],
                                        shared_ft, shared_h0, shared_h1);
    if (lane == 0 && valid) out[i] = value;
}

// ---------------------------------------------------------------------------
// The host side
// ---------------------------------------------------------------------------

namespace {

// How much shared memory one warp needs for the network's forward pass.
size_t smem_per_warp(const Net& net) {
    return size_t(2 * net.ft_hidden + 4 * net.hidden) * sizeof(float);
}

// The largest block size that still fits in the shared memory the hardware
// gives a block without asking for more.
int choose_threads(const Net& net) {
    const size_t per_warp = smem_per_warp(net);
    for (int warps = 16; warps >= 1; --warps) {
        if (per_warp * size_t(warps) <= 48 * 1024) return warps * 32;
    }
    return 32;
}

}  // namespace

// The memory model: how much the generator needs, and how many games fit in
// a given amount of video memory.  The notebook asks for this before it
// decides how many games to run at once.
py::dict memory_model(int threads) {
    const size_t per_game = sizeof(GameMem);
    const size_t per_warp_queue = size_t(WARP_QUEUE) * sizeof(Request);
    const size_t per_game_values = size_t(MAX_CHILDREN) * sizeof(float);
    const size_t per_warp = per_warp_queue / 32;   // the queue is per warp
    py::dict d;
    d["bytes_per_game"] = (long long)per_game;
    d["bytes_per_game_total"] =
        (long long)(per_game + per_warp + per_game_values);
    d["queue_bytes_per_warp"] = (long long)per_warp_queue;
    d["records_bytes_per_game"] = (long long)(MAX_RECORDS * RECORD_SIZE);
    d["max_records"] = MAX_RECORDS;
    d["threads"] = threads;
    return d;
}

py::dict run_gpu(const std::string& net_path, const std::string& out_path, int games,
                  int sims, uint64_t seed, double leaf_scale, double value_scale,
                  double c_puct, double widening, double dirichlet_alpha,
                  double dirichlet_eps, int max_ply, int threads) {
    py::dict stats;
    Net net;
    if (!load_net(net_path.c_str(), net)) {
        throw std::runtime_error("cannot load " + net_path);
    }
    if ((net.hidden + 31) / 32 > 8) {
        throw std::runtime_error("hidden layer too wide for the warp kernel");
    }
    Geometry geom = build_geometry();
    MctsConfig cfg;
    cfg.sims = sims;
    cfg.leaf_scale = leaf_scale;
    cfg.value_scale = value_scale;
    cfg.c_puct = c_puct;
    cfg.widening_k = widening;
    cfg.dirichlet_alpha = dirichlet_alpha;
    cfg.dirichlet_eps = dirichlet_eps;
    cfg.max_ply = max_ply;

    if (threads <= 0) threads = choose_threads(net);
    if (threads % 32 != 0) threads = (threads / 32) * 32;
    if (threads < 32) threads = 32;

    const int blocks = (games + threads - 1) / threads;
    const int warps = blocks * (threads / 32);

    GameMem* d_mems = nullptr;
    Net* d_net = nullptr;
    Geometry* d_geom = nullptr;
    MctsConfig* d_cfg = nullptr;
    Request* d_queue = nullptr;
    float* d_values = nullptr;
    uint8_t* d_records = nullptr;
    int* d_counts = nullptr;

    CUDA_CHECK(cudaMalloc(&d_mems, size_t(games) * sizeof(GameMem)));
    CUDA_CHECK(cudaMalloc(&d_net, sizeof(Net)));
    CUDA_CHECK(cudaMalloc(&d_geom, sizeof(Geometry)));
    CUDA_CHECK(cudaMalloc(&d_cfg, sizeof(MctsConfig)));
    CUDA_CHECK(cudaMalloc(&d_queue, size_t(warps) * WARP_QUEUE * sizeof(Request)));
    CUDA_CHECK(cudaMalloc(&d_values, size_t(games) * MAX_CHILDREN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_records, size_t(games) * MAX_RECORDS * RECORD_SIZE));
    CUDA_CHECK(cudaMalloc(&d_counts, size_t(games) * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_net, &net, sizeof(Net), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_geom, &geom, sizeof(Geometry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cfg, &cfg, sizeof(MctsConfig), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_mems, 0, size_t(games) * sizeof(GameMem)));
    CUDA_CHECK(cudaMemset(d_values, 0, size_t(games) * MAX_CHILDREN * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_counts, 0, size_t(games) * sizeof(int)));

    // The Zobrist table has to reach the device as well.
    CUDA_CHECK(cudaMemcpyToSymbol(jungle_gpu::g_zobrist, &host_zobrist(),
                                  sizeof(ZobristTable)));

    const size_t smem = size_t(threads / 32) * smem_per_warp(net);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    selfplay_kernel<<<blocks, threads, smem>>>(d_mems, d_net, d_geom, d_cfg, games, seed,
                                                d_queue, d_values, d_records, d_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));
    float seconds = 0.f;
    cudaEventElapsedTime(&seconds, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    // The records, back to the host.
    std::vector<uint8_t> records(size_t(games) * MAX_RECORDS * RECORD_SIZE);
    std::vector<int> counts(size_t(games), 0);
    CUDA_CHECK(cudaMemcpy(records.data(), d_records, records.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, counts.size(), cudaMemcpyDeviceToHost));

    // Write the shard, and count what came out of it.
    long long positions = 0;
    int red_wins = 0, black_wins = 0, draws = 0, played = 0;
    std::FILE* f = std::fopen(out_path.c_str(), "wb");
    if (f) {
        for (int g = 0; g < games; ++g) {
            const int n = counts[g];
            if (n <= 0) continue;
            const uint8_t* rec = records.data() + size_t(g) * MAX_RECORDS * RECORD_SIZE;
            std::fwrite(rec, RECORD_SIZE, size_t(n), f);
            positions += n;
            ++played;
            const int code = (rec[63] >> 2) & 3;
            if (code == 2) ++red_wins;
            else if (code == 0) ++black_wins;
            else ++draws;
        }
        std::fclose(f);
    }

    CUDA_CHECK(cudaFree(d_mems));
    CUDA_CHECK(cudaFree(d_net));
    CUDA_CHECK(cudaFree(d_geom));
    CUDA_CHECK(cudaFree(d_cfg));
    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_records));
    CUDA_CHECK(cudaFree(d_counts));

    stats["games"] = played;
    stats["positions"] = positions;
    stats["red_wins"] = red_wins;
    stats["black_wins"] = black_wins;
    stats["draws"] = draws;
    stats["seconds"] = seconds / 1000.0;
    stats["positions_per_second"] = seconds > 0 ? positions / (seconds / 1000.0) : 0.0;
    stats["games_per_second"] = seconds > 0 ? played / (seconds / 1000.0) : 0.0;
    stats["threads"] = threads;
    stats["blocks"] = blocks;
    stats["bytes_per_game"] = (long long)sizeof(GameMem);
    stats["vram_bytes"] = (long long)(size_t(games) * sizeof(GameMem) +
                                       size_t(warps) * WARP_QUEUE * sizeof(Request) +
                                       size_t(games) * MAX_CHILDREN * sizeof(float) +
                                       size_t(games) * MAX_RECORDS * RECORD_SIZE);
    return stats;
}

// The GPU's forward pass on a batch of positions, so that it can be compared
// with the engine's.
torch::Tensor nnue_batch(torch::Tensor boards, torch::Tensor stms, const std::string& net_path) {
    Net net;
    if (!load_net(net_path.c_str(), net)) {
        throw std::runtime_error("cannot load " + net_path);
    }
    const int count = int(boards.size(0));
    auto out = torch::zeros({count}, boards.options().dtype(torch::kFloat32));

    Net* d_net = nullptr;
    uint8_t* d_boards = nullptr;
    int* d_stms = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_net, sizeof(Net)));
    CUDA_CHECK(cudaMalloc(&d_boards, size_t(count) * 63));
    CUDA_CHECK(cudaMalloc(&d_stms, size_t(count) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, size_t(count) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_net, &net, sizeof(Net), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boards, boards.data_ptr<uint8_t>(), size_t(count) * 63,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stms, stms.data_ptr<int>(), size_t(count) * sizeof(int),
                           cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    const size_t smem = size_t(threads / 32) * smem_per_warp(net);
    nnue_batch_kernel<<<blocks, threads, smem>>>(d_net, d_boards, d_stms, count, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data_ptr<float>(), d_out, size_t(count) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_net));
    CUDA_CHECK(cudaFree(d_boards));
    CUDA_CHECK(cudaFree(d_stms));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run_gpu, "run the GPU self-play generator",
          py::arg("net_path"), py::arg("out_path"), py::arg("games") = 1024,
          py::arg("sims") = 800, py::arg("seed") = 12345, py::arg("leaf_scale") = 0.25,
          py::arg("value_scale") = 10.0, py::arg("c_puct") = 1.4, py::arg("widening") = 1.6,
          py::arg("dirichlet_alpha") = 0.30, py::arg("dirichlet_eps") = 0.25,
          py::arg("max_ply") = 200, py::arg("threads") = 0);
    m.def("nnue_batch", &nnue_batch, "evaluate a batch of positions on the GPU",
          py::arg("boards"), py::arg("stms"), py::arg("net_path"));
    m.def("memory_model", &memory_model, "how much memory the generator needs",
          py::arg("threads") = 256);
}
