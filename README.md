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
compared on the same positions:

| teacher's leaf | how the replies are aggregated | correlation with the game outcome |
|---|---|---|
| handcrafted static evaluation (the thing to beat) | — | +0.147 |
| handcrafted | mean over the opponent's replies | +0.150 |
| handcrafted | the opponent's best reply | +0.148 |
| handcrafted | depth-2 alpha-beta | +0.153 |
| **network** | **mean over the opponent's replies** | **+0.209** |
| network | the opponent's best reply | +0.118 … +0.129 |

(The absolute numbers depend on the dataset; the *ordering* is what matters.
The same measurement over the positions late in a game, ply 120+, is far
higher, because the signal is concentrated where the game is actually
decided.)

Two things in that table are worth keeping in mind.

The search is not what makes the teacher's values good — the leaf is.  With
the handcrafted evaluation, one ply of search over the opponent's replies
moves the correlation by three thousandths.  With the network as the leaf, the
same search is worth six hundredths, and the teacher's values become 39% more
informative than the handcrafted ones.  This is the largest lever the project
has, and it is why the Colab notebook puts a network into the teacher half an
hour into the session rather than after it.

The "best reply" aggregation is worse than the mean, and with a network it is
much worse: the network's value uses only a small part of the [-1,1] range,
so the maximum over thirty replies saturates at the clamp and the twenty
per cent of values it produces outside [-0.5, 0.5] say more about the tail
of the distribution than about the position.

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
                 --nnue ../net/net.nnue --leaf-scale 1.0 \
                 --leaf-depth 1 --leaf-agg 1 --c-puct 1.4 \
                 --out ../data/round1/data
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

**As a player.**  400 games against the handcrafted engine, one second per
move, the colours alternating from game to game:

| | |
|---|---|
| neural engine | **202 wins, 13 losses, 185 draws** |
| score | **0.736**, 95% confidence interval **[0.693, 0.779]** |
| Elo | **+178** |
| decisive games | 93.9% won, 95% confidence interval [0.908, 0.971] |

Every one of the 215 decisive games was won by entering the opponent's den,
which is how the game is supposed to be won; the other 185 ended in a
three-fold repetition.

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
                      teacher_probe
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
  verify_all.sh       everything the CI checks
  make_colab_notebook.py
colab/
  train_colab.ipynb
```

---

## Licence

MIT — see [LICENSE](LICENSE).
