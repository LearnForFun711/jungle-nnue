// jungle_mcts.cuh -- The MCTS teacher, on the GPU.
//
// This is the teacher from engine/src/mcts.hpp, with one difference that
// matters on a GPU: it is written as a state machine.  A simulation can stop
// in the middle, when a node is expanded and the value of the position has
// to be looked up, and carry on once the value arrives.  That is what lets a
// thousand games ask for their leaf evaluations at the same time and have
// them answered in one batch.
//
// The tree lives in the game's own memory, so a thousand games in flight do
// not touch the same cache lines.
//
// The search itself is the same as the CPU teacher's: PUCT, progressive
// widening, a one-ply leaf that averages the evaluations of the opponent's
// replies, and noise at the root's priors.  The rules that decide whether a
// position is won, drawn or lost -- a move into the opponent's den, a
// repetition, the length limit -- are the CPU teacher's rules too, and they
// are applied when the child is created, exactly as the CPU teacher does it.
// A tree that does not know that a den entry wins is a different game.
#pragma once

#include "jungle_gpu.cuh"

namespace jungle_gpu {

constexpr uint32_t kNoChild = 0xFFFFFFFFu;

// How many nodes one game's tree may hold.  A search of `sims` simulations
// creates at most one node per simulation, plus the root's children, so a
// thousand simulations fit.  When the tree is full no more children are
// created and the search falls back on the node's leaf value, which is what
// the CPU teacher does when it runs out of nodes as well.
constexpr int MAX_NODES = 1024;

struct GpuNode {
    float    value_sum = 0.f;   // the values backed up, from this node's view
    float    leaf = 0.f;        // the leaf evaluation, computed once
    float    prior = 0.f;       // the prior of the move that led here
    uint32_t visits = 0;
    uint32_t first_child = kNoChild;  // the children are a list, not a block:
                                     // a node may have ninety legal moves and
                                     // the widening only ever creates a few
                                     // of them, so reserving a slot for each
                                     // would waste most of the memory.
    uint32_t last_child = kNoChild;
    uint32_t next_sibling = kNoChild;
    uint32_t parent = 0;
    uint64_t key = 0;           // the key of the position at this node
    uint16_t num_children = 0;  // the children created so far
    uint16_t num_moves = 0;     // the legal moves, known after the expansion
    uint16_t from = 0xFFFF, to = 0xFFFF;
    uint16_t ply = 0;           // the ply of the position at this node
    uint8_t  rep = 0;           // how many times the position has repeated
    uint8_t  captured = 0;      // the piece that stood on `to` before the move
    int8_t   status = ST_ONGOING;
};

constexpr int MAX_PATH = 256;
constexpr int MAX_CHILDREN = 128;

// The memory one game needs: the tree, the records of the decisions taken so
// far, and the history of the positions for the repetition test.
struct GameMem {
    GpuNode  nodes[MAX_NODES];
    uint8_t  records[256][68];
    uint32_t history[300];
    int      record_count = 0;
    int      history_count = 0;
};

// ---------------------------------------------------------------------------
// The configuration
// ---------------------------------------------------------------------------

struct MctsConfig {
    int    sims = 800;
    double c_puct = 1.4;
    double widening_k = 1.6;
    int    widening_max = 48;
    double dirichlet_alpha = 0.30;
    double dirichlet_eps = 0.25;
    int    max_ply = 200;
    double leaf_scale = 1.0;
    double value_scale = 10.0;   // the network's value is multiplied by this
    uint64_t seed = 1;
};

// The phases of one simulation.
enum MctsPhase : int {
    PHASE_DESCEND = 0,   // walking down the tree
    PHASE_WAIT = 1,      // the leaf's value has been asked for
    PHASE_BACKUP = 2,    // sending the value back up
};

// The state of one game
struct Teacher {
    // the shared, read-only inputs
    const Geometry* geom = nullptr;
    const MctsConfig* cfg = nullptr;

    // the tree
    GameMem* mem = nullptr;
    uint32_t node_count = 0;
    uint32_t root = 0;

    // the search
    Board    board;          // the position, moved along with the descent
    uint32_t path[MAX_PATH];
    int      depth = 0;
    int      phase = PHASE_DESCEND;
    float    value = 0.f;
    int      sim = 0;

    // the expansion that is waiting for its values
    uint32_t expand_node = 0;
    uint8_t  child_from[MAX_CHILDREN];
    uint8_t  child_to[MAX_CHILDREN];
    float    child_prior[MAX_CHILDREN];
    int      nchildren = 0;

    // the noise at the root: the sum of the exponentials drawn so far
    double root_noise_sum = 0.0;

    // the random numbers
    uint64_t rng = 1;

    // ---- random numbers ------------------------------------------------
    // A small xorshift: the GPU has no room for a Mersenne twister, and the
    // teacher only needs noise for the root's priors and the move choice.
    uint64_t next_random() {
        uint64_t x = rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        rng = x;
        return x;
    }
    double next_uniform() {
        return double(next_random() >> 11) * (1.0 / 9007199254740992.0);
    }

    // A standard normal sample.
    double next_gaussian() {
        const double u1 = fmax(1e-12, next_uniform());
        const double u2 = next_uniform();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    }

    // A gamma sample with the shape `alpha` and the scale one -- the same
    // distribution the CPU teacher's Dirichlet noise draws from.
    //
    // The first version of the GPU teacher stood in for the gamma samples
    // with exponentials of the same mean.  The means matched but the shapes
    // did not, and it showed: the engine's noise is much more spread out, so
    // the engine's teacher wanders and ends a lot of its games in a
    // repetition draw, while the GPU teacher followed the priors so closely
    // that it ran into the opponent's den almost every game.  The same
    // distribution gives the same teacher.
    double next_gamma(double alpha) {
        if (alpha < 1.0) {
            // Boost: a gamma of a small shape is a gamma of a larger shape
            // scaled down by a power of a uniform sample.
            const double u = fmax(1e-12, next_uniform());
            return next_gamma(alpha + 1.0) * pow(u, 1.0 / alpha);
        }
        const double d = alpha - 1.0 / 3.0;
        const double c = 1.0 / sqrt(9.0 * d);
        for (;;) {
            const double x = next_gaussian();
            const double v = 1.0 + c * x;
            if (v <= 0.0) continue;
            const double vv = v * v * v;
            const double u = fmax(1e-12, next_uniform());
            if (log(u) < 0.5 * x * x + d * (1.0 - vv + log(vv))) return d * vv;
        }
    }

    // ---- the priors ----------------------------------------------------
    // The same heuristic the CPU teacher uses: a move into a den is worth
    // everything, a capture is worth more the bigger the victim, and a quiet
    // move is worth more the closer it gets to the enemy den.
    float move_prior(int from, int to) const {
        const int mover = board.sq[from];
        const int captured = board.sq[to];
        if (geom->den_owner[to] >= 0) return 60.0f;
        if (captured) {
            int victim = piece_type(captured);
            if (geom->trap_owner[to] == piece_color(mover)) victim = 0;
            const int attacker = piece_type(mover);
            return 10.0f + victim * 3.0f - attacker * 0.3f;
        }
        const int r = sq_row(to), c = sq_col(to);
        // Red's den is at (8,3) and Black's at (0,3).
        const int er = (board.stm == RED) ? 0 : 8;
        const int d = (r > er ? r - er : er - r) + (c > 3 ? c - 3 : 3 - c);
        return 1.0f + (14 - d) * 0.25f;
    }

    // The moves of the current position, ordered by their prior, the way the
    // CPU teacher orders them.  The order matters: the progressive widening
    // creates the children in this order, so the moves worth exploring get
    // explored first.
    int sorted_moves(uint8_t* from, uint8_t* to, float* prior) const {
        MoveBuf buf;
        const int n = generate_moves(board, *geom, buf);
        for (int i = 0; i < n; ++i) {
            from[i] = buf.from[i];
            to[i] = buf.to[i];
            prior[i] = move_prior(buf.from[i], buf.to[i]);
        }
        // A stable insertion sort by descending prior.
        for (int i = 1; i < n; ++i) {
            const uint8_t f = from[i], t = to[i];
            const float p = prior[i];
            int j = i - 1;
            while (j >= 0 && prior[j] < p) {
                from[j + 1] = from[j];
                to[j + 1] = to[j];
                prior[j + 1] = prior[j];
                --j;
            }
            from[j + 1] = f;
            to[j + 1] = t;
            prior[j + 1] = p;
        }
        return n;
    }

    // ---- the tree ------------------------------------------------------

    uint32_t new_node(uint32_t parent) {
        const uint32_t idx = node_count++;
        mem->nodes[idx] = GpuNode();
        mem->nodes[idx].parent = parent;
        return idx;
    }

    int max_children_allowed(uint32_t node) const {
        const GpuNode& n = mem->nodes[node];
        const int budget = 1 + int(cfg->widening_k * sqrt(double(n.visits)));
        int limit = budget < cfg->widening_max ? budget : cfg->widening_max;
        if (limit > int(n.num_moves)) limit = int(n.num_moves);
        return limit;
    }

    // Start a new search from the current position.
    void begin_search() {
        node_count = 1;
        root = 0;
        mem->nodes[0] = GpuNode();
        mem->nodes[0].parent = 0xFFFFFFFFu;
        mem->nodes[0].key = board.key;
        mem->nodes[0].ply = board.ply;
        depth = 0;
        phase = PHASE_DESCEND;
        sim = 0;
        value = 0.f;
        nchildren = 0;
        root_noise_sum = 0.0;
    }

    // Create the next child of a node.  The move list is regenerated and
    // re-sorted rather than stored, which keeps the tree small; the order is
    // deterministic, so the k-th child is always the same move.
    uint32_t create_child(uint32_t node) {
        GpuNode& parent = mem->nodes[node];
        uint8_t from[MAX_CHILDREN], to[MAX_CHILDREN];
        float prior[MAX_CHILDREN];
        const int nmoves = sorted_moves(from, to, prior);
        parent.num_moves = uint16_t(nmoves);
        if (nmoves == 0) {
            parent.status = ST_LOSS;   // the side to move is stuck
            return 0xFFFFFFFFu;
        }
        const int k = parent.num_children;
        if (k >= nmoves) return 0xFFFFFFFFu;
        if (node_count >= uint32_t(MAX_NODES)) return 0xFFFFFFFFu;

        const uint32_t child = new_node(node);
        GpuNode& c = mem->nodes[child];
        c.from = from[k];
        c.to = to[k];
        c.prior = prior[k];
        c.captured = board.sq[to[k]];
        c.key = board.key_after_move(from[k], to[k]);
        c.ply = uint16_t(board.ply + 1);
        c.rep = uint8_t(parent.rep + (parent.key == c.key ? 1 : 0));

        // The status of the child, with the CPU teacher's rules.  A move into
        // the opponent's den wins at once, so the child -- the opponent to
        // move -- is lost.  A position that has repeated three times, or one
        // that has reached the length limit, is drawn.
        if (geom->den_owner[to[k]] == (board.stm ^ 1)) {
            c.status = ST_LOSS;
        } else if (c.rep >= 2) {
            c.status = ST_DRAW;
        } else if (int(c.ply) >= cfg->max_ply) {
            c.status = ST_DRAW;
        }

        // The children hang off their parent in the order they were created.
        // The CPU teacher picks the first child with the best PUCT score and
        // the first child with the most visits, and both of those are ties
        // that the creation order decides, so the order has to be the same.
        c.next_sibling = kNoChild;
        if (parent.last_child == kNoChild) parent.first_child = child;
        else mem->nodes[parent.last_child].next_sibling = child;
        parent.last_child = child;

        if (node == root) {
            // The root's priors get noise, so that the first move chosen is
            // not always the same one.  The noise is the running sum of
            // gamma samples, which is what the CPU teacher adds up.
            //
            // The order matters: the CPU teacher gives the k-th child the
            // sum of the first k gamma samples, so the first child -- the
            // one with the best prior -- gets no noise at all and its prior
            // is only scaled down.  Adding the sample before using the sum
            // gives every child a different noise, and the teacher then
            // follows the priors so closely that it runs into the
            // opponent's den almost every game.
            const double eps = cfg->dirichlet_eps;
            c.prior = float((1.0 - eps) * c.prior + eps * root_noise_sum);
            root_noise_sum += next_gamma(cfg->dirichlet_alpha);
        }
        parent.num_children = uint16_t(k + 1);
        return child;
    }

    // ---- the descent ---------------------------------------------------

    // Walk down the tree.  Returns true when the search has to stop and wait
    // for the value of an expanded node.
    bool descend() {
        depth = 0;
        path[0] = root;
        while (true) {
            uint32_t node = path[depth];
            GpuNode& n = mem->nodes[node];

            if (n.status != ST_ONGOING) {
                value = (n.status == ST_WIN) ? 1.f : (n.status == ST_LOSS ? -1.f : 0.f);
                phase = PHASE_BACKUP;
                return false;
            }

            if (n.visits == 0) {
                // The node is expanded: its leaf value needs the evaluation
                // of every child, which is the batched part.
                expand_node = node;
                if (!prepare_expansion()) {
                    value = (n.status == ST_WIN) ? 1.f : (n.status == ST_LOSS ? -1.f : 0.f);
                    phase = PHASE_BACKUP;
                    return false;
                }
                if (node == root) {
                    // The CPU teacher creates every one of the root's
                    // children before the search starts, so the widening
                    // never applies to the root and the noise is drawn for
                    // all of them.
                    while (mem->nodes[root].num_children < mem->nodes[root].num_moves)
                        if (create_child(root) == kNoChild) break;
                }
                phase = PHASE_WAIT;
                return true;
            }

            if (int(n.num_children) < max_children_allowed(node)) create_child(node);
            if (n.num_children == 0) {
                value = n.leaf;
                phase = PHASE_BACKUP;
                return false;
            }

            // PUCT.
            const float sqrt_n = sqrtf(float(n.visits));
            uint32_t best = kNoChild;
            float best_score = -1e18f;
            for (uint32_t c = n.first_child; c != kNoChild; c = mem->nodes[c].next_sibling) {
                const GpuNode& cn = mem->nodes[c];
                const float q = cn.visits ? -(cn.value_sum / float(cn.visits)) : 0.f;
                const float u = float(cfg->c_puct) * cn.prior * sqrt_n / (1.0f + float(cn.visits));
                const float s = q + u;
                if (s > best_score) { best_score = s; best = c; }
            }
            if (best == kNoChild) {
                value = n.leaf;
                phase = PHASE_BACKUP;
                return false;
            }
            const uint32_t child = best;
            // Apply the move so that the next node sees its own position.
            const uint16_t from = mem->nodes[child].from;
            const uint16_t to = mem->nodes[child].to;
            apply_move(board, from, to);
            path[++depth] = child;
            if (depth >= MAX_PATH - 1) {
                value = 0.f;
                phase = PHASE_BACKUP;
                return false;
            }
        }
    }

    // Prepare the expansion of a node: work out its legal moves, in the order
    // the children will be created in.  The child boards are written by the
    // caller (they go into the batch), and the values are expected back in
    // the same order.
    bool prepare_expansion() {
        GpuNode& n = mem->nodes[expand_node];
        const int nmoves = sorted_moves(child_from, child_to, child_prior);
        n.num_moves = uint16_t(nmoves);
        if (nmoves == 0) {
            n.status = ST_LOSS;    // the side to move has no legal move
            return false;
        }
        nchildren = nmoves;
        return true;
    }

    // The children's values have arrived: average them, which is what the
    // one-ply leaf does.
    void finish_expansion(const float* child_values) {
        GpuNode& n = mem->nodes[expand_node];

        // A move into the opponent's den wins at once.  The CPU teacher's
        // leaf evaluation returns 1.0 the moment it sees such a move, and
        // that is worth copying exactly: a teacher that cannot see the win
        // is a teacher that plays on.
        for (int i = 0; i < nchildren; ++i) {
            if (geom->den_owner[child_to[i]] == (board.stm ^ 1)) {
                n.leaf = 1.0f;
                value = 1.0f;
                phase = PHASE_BACKUP;
                return;
            }
        }

        double sum = 0.0;
        for (int i = 0; i < nchildren; ++i) {
            // The child is the opponent to move, so its value is the other
            // side's.
            sum += -double(child_values[i]);
        }
        n.leaf = float(sum / double(nchildren));
        value = n.leaf;           // the first visit backs up the leaf value
        phase = PHASE_BACKUP;
    }

    // Send the value back up the tree and restore the board.
    void backup() {
        for (int i = depth; i >= 0; --i) {
            GpuNode& n = mem->nodes[path[i]];
            n.visits++;
            n.value_sum += value;
            value = -value;
        }
        // Put the board back where the search started.  The captured piece
        // has to be put back as well: a search that forgets it leaves the
        // board with pieces missing, and the game after the search is then
        // a different game.
        for (int i = depth; i >= 1; --i) {
            const GpuNode& n = mem->nodes[path[i]];
            board.sq[n.from] = board.sq[n.to];
            board.sq[n.to] = n.captured;
            board.stm = uint8_t(board.stm ^ 1);
            board.ply = uint16_t(board.ply - 1);
            board.key = mem->nodes[path[i - 1]].key;   // the parent's key
        }
        depth = 0;
        sim++;
        phase = PHASE_DESCEND;
    }
};

}  // namespace jungle_gpu
