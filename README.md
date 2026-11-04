# Jungle Chess NNUE

An open-source Jungle Chess (鬥獸棋 / 斗兽棋) engine built around a
Stockfish-style neural network.

The project is a complete, reproducible pipeline:

```
   MCTS teacher  ──►  self-play data  ──►  PyTorch NNUE  ──►  int8 binary
   (PUCT search)      (compact shards)     (quantisation-aware)   (engine reads it)
                                                        │
                                                        ▼
                                            alpha-beta + NNUE  ──►  plays Jungle
```

Everything is in this repository: the C++ engine, the teacher, the dataset
format, the training code and the evaluation harness.  There are no external
dependencies for the engine itself and no external datasets — the network is
trained entirely from self-play.

---

## The game

Standard 7x9 Jungle Chess.

| | |
|---|---|
| Board | 9 rows x 7 columns; Red at the bottom, Red moves first |
| Pieces | Rat, Cat, Dog, Wolf, Leopard, Tiger, Lion, Elephant (rank 1..8) |
| Dens | Red (8,3), Black (0,3) — entering the opponent's den wins at once |
| Traps | Black (0,2) (0,4) (1,3); Red (8,2) (8,4) (7,3) — a piece on the opponent's trap loses its rank and can be taken by anything, even a Rat |
| River | Rows 3..5, columns 1,2,4,5 — only the Rat may enter |
| Jumps | Tiger and Lion jump the river (3 squares across, 4 down) if the path is clear |
| Water | A Rat in the river cannot capture a piece on land, and land pieces cannot capture a Rat in the river |
| End | No legal move loses; three-fold repetition is a draw |

The implementation is verified against the reference engine of the original
project:

```bash
make -C engine
python3 scripts/verify_rules.py        # 1200 random positions, exact match
```

---

## Building

```bash
cd engine
make                 # add NATIVE=1 for -march=native
```

Produces:

| binary | purpose |
|---|---|
| `build/selfplay` | MCTS self-play data generator |
| `build/match` | engine-vs-engine match harness with Elo |
| `build/dump_moves` | prints the legal moves of a position (rule checking) |
| `build/eval_nnue` | prints the network's value for a position (parity checking) |
| `build/nnue_check` | proves the incremental NNUE accumulators match a full refresh |
| `build/bench` | micro-benchmarks of the hot paths |
| `build/libjungle.so` | shared library with a flat C API for embedding |

Only a C++17 compiler is required.

Everything the CI checks can also be run locally:

```bash
scripts/verify_all.sh
```

---

## The teacher: MCTS

`engine/src/mcts.hpp` implements PUCT MCTS with progressive widening,
Dirichlet noise at the root and a search-based leaf evaluation.

The leaf evaluation is the interesting part.  A plain static evaluation gives
the teacher values that are no better than the handcrafted evaluation it is
meant to replace — training on it would not produce a network worth having.
Evaluating every child position and averaging (rather than taking the
maximum) gives a teacher that is measurably better.

Measured as the Pearson correlation between the teacher's value and the actual
outcome of the game the position came from, both from the side-to-move's point
of view.  `engine/tools/teacher_probe.cpp` runs the teacher over a fixed
sample of positions from finished games, so two configurations are always
compared on the same positions.  4,000 positions from a fresh shard, 800
simulations a move, the network at `net/net.nnue`:

| teacher's leaf | scale | correlation with the outcome | correlation on decisive games | spread of the values |
|---|---|---|---|---|
| handcrafted static evaluation (the thing to beat) | — | +0.189 | +0.238 | 0.192 |
| network, mean over the opponent's replies | 1.0 | +0.150 | +0.223 | 0.563 |
| network, mean over the opponent's replies | 0.5 | +0.173 | +0.247 | 0.466 |
| network, mean over the opponent's replies | 0.25 | +0.210 | +0.276 | 0.334 |
| network, mean over the opponent's replies | 0.1875 | +0.224 | +0.285 | 0.275 |
| **network, mean over the opponent's replies** | **0.125** | **+0.236** | **+0.290** | 0.202 |
| network, mean over the opponent's replies | 0.0625 | +0.241 | +0.277 | 0.121 |
| network with a quarter of the handcrafted evaluation mixed in | 0.125 | +0.227 | +0.279 | 0.198 |

(The absolute numbers depend on the dataset; the *ordering* is what matters.
The same measurement over the positions late in a game, ply 120+, is far
higher, because the signal is concentrated where the game is actually
decided.)

Three things in that table are worth keeping in mind.

**The scale of the network is a lever of its own.**  The engine multiplies
the network's value by ten before it is used as a search score, because the
network only uses a small part of its range.  Used at that scale the values
saturate: 74% of them come out at the clamp of [-1,1], their spread is
0.563 instead of 0.192, and the correlation with the outcome is *worse* than
the handcrafted teacher's.  Used at an eighth of that scale the same network
is the best leaf measured: +0.236 against the handcrafted teacher's +0.189.
The difference between the two settings is not a detail -- it is the
difference between a teacher that is worth more than the handcrafted one and
one that is worth less.

**The search is not what makes the teacher's values good -- the leaf is.**
With the handcrafted evaluation, one ply of search over the opponent's
replies moves the correlation by three thousandths.  With the network as the
leaf, the same search is worth six hundredths, and the teacher's values
become 39% more informative than the handcrafted ones.

**Mixing the handcrafted evaluation back in makes it worse.**  A teacher
whose leaf is three quarters network and a quarter handcrafted scores
+0.227, below the network's own +0.236.  The handcrafted evaluation is not
a useful correction to a network that has been trained on the same game; it
is noise added to a signal.

### A bug that was worth more than any of the tuning

The numbers above were measured after a bug was found: `selfplay` and
`teacher_probe` shared **one** evaluator between all their threads, and the
evaluator holds the network's feature accumulators, which every evaluation
changes.  Two threads evaluating at the same time therefore read each
other's half-finished positions, and the teacher's values came out as a
mixture of several positions' worth of arithmetic.  Each thread has its own
evaluator now.

The bug had been quietly making the network look worse than it is.  Before the
fix, the network's leaf and the handcrafted leaf were indistinguishable; after
it, the network's leaf at a scale of 0.125 is worth **+0.236** against the
handcrafted teacher's +0.189 -- a 25% improvement in the quality of the data
the whole project is made of.  The handcrafted teacher was never affected
(its evaluation is a pure function of the position), which is why the dataset
already on disk is still good.

### The teacher on the GPU

The teacher is the bottleneck of this project: every node the search expands
needs the network evaluated once for every legal move, and the CPU does that
one evaluation at a time.  The whole teacher can run on the GPU instead.

`engine/gpu/` holds the teacher in a form that compiles both as CUDA and as
plain C++:

* `jungle_gpu.cuh` — the move generator, written with bitboards.  Every legal
  move of every piece of a type comes out of one shift and a few masks, and
  the code is the same whether it runs on one thread or a thousand.
* `jungle_gpu.cuh` — the network's forward pass, which is the engine's own
  arithmetic.  It is bit-identical to the engine's: the dot products are
  sums of products of small integers, which are exact in fp32, and the
  clipping and rounding steps are the same ones.
* `jungle_mcts.cuh` — the search, written as a state machine so that a
  simulation can stop in the middle, ask for the value of a leaf, and carry
  on when the answer comes back.  That is what lets a thousand games share
  one batched evaluation.
* `jungle_game.cuh` — the game loop, writing the same 68-byte records the CPU
  generator writes.
* `jungle_gpu_cuda.cu` — the CUDA launcher: one thread per game, the leaf
  evaluations of a warp's games batched together, `__syncwarp()` and never
  `__syncthreads()`.

The port is checked against the engine, not assumed to be right:

| check | how | result |
|---|---|---|
| the move generator | 80,000 positions from real games, against `engine/src/jungle.hpp` | all agree |
| the forward pass | 20,000 positions, against the engine's int8 path | worst difference 0 |
| the teacher's values | 300 positions, the CPU teacher's values against the engine's | 300/300 identical |
| the CUDA source | type-checked against a shim for the toolkit (`scripts/check_cuda_syntax.py`) | passes |

The values are checked value by value with the root's noise turned off, which
makes both teachers deterministic: the GPU teacher's root value matches the
engine's to the last decimal on every position.

How many games to run at once is measured, not chosen.  `engine/gpu/gpu_selfplay.py`
asks the compiled code how much video memory a game needs, works out how many
fit in the card, then runs each candidate number and measures the positions
per second -- the games finish at different times, so the batch waits for the
slowest lane, and past a point more games stop helping.  The number it
recommends is written to `runs/gpu/autotune.json` and the generator reads it
from there.  Nothing in the notebook hard-codes a batch size.

Generate data:

```bash
cd engine
./build/selfplay --games 5000 --threads 8 --sims 800 \
                 --leaf-depth 1 --leaf-agg 1 --c-puct 1.4 \
                 --out ../data/data
```

With a network as the teacher's leaf:

```bash
./build/selfplay --games 5000 --threads 8 --sims 800 \
                 --nnue ../net/net.nnue --leaf-scale 0.125 \
                 --leaf-depth 1 --leaf-agg 1 --c-puct 1.4 \
                 --out ../data/round1/data
```

On the GPU, the same teacher:

```bash
python3 engine/gpu/gpu_selfplay.py --net net/net.nnue --out data/gpu/data.t0.bin \
                                    --games 4096 --sims 800
```

Each thread writes its own shard, `data/data.t<N>.bin`, 68 bytes per
position, so generation can be stopped at any time and whatever is on disk is
already usable.

---

## The dataset

One record per decision position:

```
offset  size  meaning
0       63    board: 0 empty, 1..8 Red piece (type), 9..16 Black piece (type)
63      1     meta: bit0 = side to move (0 Red, 1 Black)
                     bits 2..3 = game result from the side-to-move's view
                                (0 loss, 1 draw, 2 win)
64      1     MCTS value, quantised from [-1,1] to [0,255]
65      2     ply (uint16)
67      1     ending cause: 0 den entry, 1 no legal moves, 2 repetition, 3 length
```

At 68 bytes a position, the 3.7 million positions used here are 253 MB raw and
**29 MB compressed** — small enough to move in a single upload, which is why the
dataset is generated on the machine with the most cores and uploaded once.

Inspect a dataset with:

```bash
python3 scripts/analyze_data.py --data "data/data.t*.bin"
```

which reports the value/outcome correlation, the calibration curve and the
value distribution — the numbers to watch when changing the teacher.

---

## The network

`train/model.py` implements a Stockfish-style NNUE:

```
features (2 perspectives) -> FT(1008 -> ft_hidden) -> H1(ft_hidden -> hidden)
                                                 -> H2(hidden -> hidden)
                                                 -> OUT(concat -> 1)
```

* **Two perspectives.**  Red is "own" with the board as it is; Black is "own"
  with the board flipped vertically, so each side's own den is always at
  (8,3).  The board is vertically symmetric, so the flip is a legal
  relabelling and it halves the work the network has to do.
* **Side-to-move ordering.**  The two perspectives are concatenated in
  side-to-move order, exactly as Stockfish does, so the network can learn the
  tempo effect.
* **Quantisation aware.**  The forward pass uses the same int8 weights and the
  same rounding and clipping as the C++ inference, using a straight-through
  estimator.  The exported network therefore plays *exactly* the network that
  was trained — `scripts/verify_nnue.py` asserts this to within 1e-6.
* **Fast enough to matter.**  A network that is stronger per evaluation but
  three times slower is not stronger at all.  The feature transformer is stored
  transposed so that a single piece update touches one contiguous run of
  weights, and the search updates the accumulators incrementally as it
  traverses the tree instead of rebuilding them at every node.  An evaluation
  is then the forward pass and nothing else.  `engine/tools/nnue_check.cpp`
  proves the two paths agree exactly.

Per-layer quantisation, per output row:

```
M[o]     = max_i |W[o][i]|
Wq[o][i] = round(W[o][i] / M[o] * 127)        (int8)
scale[o] = M[o] / 127
forward: acc[o] = sum_i Wq[o][i] * x[i]        (int32)
         y[o]   = clamp(round(acc[o] * scale[o]), 0, 127)
```

Every layer carries a learnable gain, because the weight magnitude needed for
the activations to land inside the [0,127] clipping range differs by five
orders of magnitude between the feature transformer (binary inputs) and the
output layer.  The gain is folded into the weights at export time; because
the per-row quantisation normalises each row to its own maximum, folding it
in leaves the int8 weights untouched.

Train:

```bash
python3 train/train.py --data "data/data.t*.bin" --out runs/exp \
        --epochs 40 --batch 8192 --lr 2e-3 --ft-hidden 128 --hidden 64
```

or, to spend a wall-clock budget rather than a number of epochs:

```bash
python3 train/train_budget.py --data "data/data.t*.bin" --out runs/budget \
        --minutes 480 --configs "128:64:0.25,96:48:0.35"
```

The learning-rate schedule is re-aimed at the budget after every epoch, so
the rate decays across the whole run instead of stopping part-way through a
schedule sized for fewer epochs.

Export and verify:

```bash
python3 train/export_nnue.py --checkpoint runs/exp/best.pt --out net.nnue
python3 scripts/verify_nnue.py --checkpoint runs/exp/best.pt --net build/verify.nnue
./engine/build/nnue_check net.nnue
```

`verify_nnue.py` proves the PyTorch and C++ forward passes agree exactly;
`nnue_check` proves the incremental accumulators the search maintains match a
full refresh exactly.

### How well does it work?

Two numbers matter.  The first says whether the network is a better
*evaluation* than the handcrafted one; the second says whether that makes it a
better *player*.

**As an evaluation.**  Measured on a held-out shard of the dataset, as the
correlation between the network's value and the *outcome of the game* the
position came from, from the side-to-move's point of view:

| | correlation with the game outcome |
|---|---|
| the handcrafted evaluation | +0.161 |
| the MCTS teacher the network is distilled from | +0.150 |
| **the distilled network** | **+0.356** |

The network predicts who wins considerably better than the evaluation it
replaces.  The teacher does not: its value is a noisy estimate of the same
quantity, and the network regressing on it removes that noise.

The network's number is a blend, and it is worth knowing why.  The training
target is `0.75 * teacher_value + 0.25 * game_result`, and the blend's own
correlation with the outcome is `0.75 * 0.150 + 0.25 * 1.000 = 0.362`.  The
network reaches +0.356, which is what it looks like when a network has
learned its target as well as the target can be learned.  So a third of the
information comes from the teacher's search and two thirds from the game's
result — and the result is the only ground truth there is.

**As a player.**  Against the handcrafted engine, the colours alternating from
game to game:

| seconds per move | games | neural engine | score | Elo |
|---|---|---|---|---|
| 1 | 400 | 202 W, 13 L, 185 D | 0.736 (95% CI [0.693, 0.779]) | **+178** |
| 3 | 200 | **170 W, 0 L, 30 D** | 0.925 (95% CI [0.889, 0.962]) | **+436** |

At one second per move the neural engine wins 93.9% of the decisive games
(95% CI [0.908, 0.971]).  At three seconds it wins *every* decisive game —
170 out of 170, with no losses at all — and the gap widens rather than
narrowing, because the extra time lets the network's evaluation pay off while
the handcrafted engine has nothing more to gain from it.

Every decisive game in both matches was won by entering the opponent's den,
which is how the game is supposed to be won; the rest ended in a three-fold
repetition.

The network width is a trade-off, and it is worth being explicit about it.
A wider network predicts slightly better but evaluates more slowly, and at a
fixed clock a slower engine searches less deeply.  The widths measured here:

| network | MACs per evaluation | correlation with the outcome |
|---|---|---|
| 96 / 48 | 14k | +0.179 |
| 256 / 128 | 99k | +0.187 |
| 512 / 256 | 394k | — |

A 96/48 network is nearly as good and seven times cheaper, which matters
much more than the last few points of accuracy: at 1s per move the neural
engine's evaluation cost is what decides how many nodes it can search.

---

## Training on Colab

`colab/train_colab.ipynb` is the whole training story in one notebook: fetch
the project, build the engine, find the dataset, train on the GPU, export the
network and download it.

Open it from the repository page (**Open in Colab**), or directly:

<https://colab.research.google.com/github.com/LearnForFun711/jungle-nnue/blob/main/colab/train_colab.ipynb>

or clone this repository into a Colab runtime and open the notebook from
there.  Then:

1. Runtime ▸ Change runtime type ▸ **A100 GPU**, then restart.
2. Run the cells in order.  The one that needs you is the dataset upload: drop
   `jungle-dataset.tar.gz` (29 MB) into the runtime with the file pane.
3. The last cell downloads `net.nnue` (about 300 KB).

The dataset is the only thing that crosses the network and it is uploaded
once; training happens entirely on the GPU.  For this model size an A100
trains in a few minutes, so the GPU time is spent on *many* short runs rather
than one long one.

```bash
python3 scripts/make_colab_notebook.py     # regenerate the notebook
```

---

## Playing and evaluating

```bash
cd engine
# neural network vs handcrafted evaluation
./build/match --games 400 --time 1.0 --engine-a "ab:nnue=../net.nnue" --engine-b ab
# MCTS teacher vs handcrafted evaluation
./build/match --games 400 --time 0.05 --engine-a "mcts:800:leaf=1" --engine-b ab
```

The harness alternates colours from game to game and reports the score with a
95% confidence interval and the corresponding Elo difference.

Embedding the engine in another program:

```c
extern int jungle_get_best_move(const char* board_flat, int turn, int move_count,
                                 int time_ms, int* out_move);
extern int jungle_get_best_move_nnue(const char* board_flat, int turn, int move_count,
                                     int time_ms, const char* nnue_path, int* out_move);
```

`board_flat` is 63 bytes: `0` empty, `1..8` Red piece of that type, `9..16`
Black piece.

---

## Layout

```
engine/
  src/jungle.hpp      board, rules, move generation, Zobrist hashing
  src/eval.hpp        handcrafted evaluation
  src/mcts.hpp        PUCT MCTS teacher
  src/nnue.hpp        int8 network: format, loading, inference
  src/search.hpp      iterative-deepening alpha-beta with a transposition table
  src/engine_api.hpp  one interface over alpha-beta and MCTS
  tools/              selfplay, match, dump_moves, eval_nnue, nnue_check, bench,
                      teacher_probe, nnue_ref
  gpu/                the teacher, written so that it compiles as CUDA and as C++
    jungle_gpu.cuh    bitboard move generator, the network's forward pass
    jungle_mcts.cuh   the search, as a resumable state machine
    jungle_game.cuh   the game loop, writing the engine's 68-byte records
    jungle_gpu_cuda.cu  the CUDA launcher
    gpu_selfplay.py   the driver and the auto-tuner
train/
  dataset.py          shard reader
  model.py            quantisation-aware NNUE
  train.py            training loop
  export_nnue.py      export to the engine's binary format
scripts/
  verify_rules.py     C++ vs. reference rule check
  verify_golden.py    C++ vs. recorded golden positions
  verify_nnue.py      torch vs. C++ parity check
  verify_notebook.py  every cell of the Colab notebook parses
  analyze_data.py     dataset quality report
  package_data.py     pack the shards for a single upload
  eval_teacher_quality.py  how well the teacher's values predict the outcome
  eval_teacher_configs.py  compare teacher configurations on the same positions
  eval_net_quality.py      how well a network predicts the outcome
  check_cuda_syntax.py     type-check the CUDA source without a CUDA toolkit
  gpu_selfplay_loop.sh     the GPU generator, in the background
  verify_all.sh       everything the CI checks
  make_colab_notebook.py
colab/
  train_colab.ipynb
```

---

## Licence

MIT — see [LICENSE](LICENSE).
