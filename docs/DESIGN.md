# Design notes

## Why the teacher is MCTS

The brief was to use MCTS as the teacher and distil the result into an
NNUE-like network.  The alternative — distilling an alpha-beta engine — was
rejected because MCTS gives two things alpha-beta does not: a visit
distribution over the root moves (a policy signal, available for free) and a
root value that is already an average over many continuations rather than a
single forced line.  That second property is what makes the MCTS value a
better regression target than a depth-limited search score.

## The bottleneck is data generation, not training

The network has roughly 300k parameters and the forward pass is about 100k
multiply-accumulates.  On an A100 that is a few million positions per second
of training, so a dataset of a few million positions can be trained on
repeatedly in minutes.  Generating that dataset takes hours on eight CPU
cores.

This inverts the usual assumption and drives the whole design:

* The generator is written in C++ and parallelised over threads, because a
  Python teacher would be two orders of magnitude slower.
* The dataset is 68 bytes per position so that moving it is cheap and can
  happen once.
* The teacher is tuned for *value quality per second of CPU*, not for playing
  strength.  This is why the leaf evaluation matters so much.

## The leaf evaluation

The first version of the teacher used the static evaluation directly at the
leaf.  Measured against the game outcome, that teacher was *worse* than the
static evaluation alone — training on it would have produced a network worse
than the handcrafted engine it was supposed to beat.

The fix was to evaluate every child position and average the results.  The
first attempt took the maximum instead, which made things worse again: the
maximum of ~30 noisy evaluations is biased upwards, which compressed every
value towards zero and destroyed the signal.  Averaging keeps the value
discriminative.

With the mean-aggregated one-ply leaf the teacher is measurably better than
the static evaluation it replaces, which is the precondition for the whole
distillation to be worth anything.

## Two bugs worth remembering

* **The game driver's den-entry result was inverted.**  After a move the side
  to move is the *loser* of that move, so returning the side-to-move's sign
  flipped the result of every game that ended with a piece reaching the den.
  This showed up as the teacher's values being *anti-correlated* with the
  game outcome.  The diagnostic that caught it was a per-ending-cause
  breakdown of the correlation, not the aggregate number.

* **The feature transformer was indexed as a flat array.**  The weights are
  stored as `[FT_INPUT][ft_hidden]` (transposed, see below), and the accumulator
  update walked the flat array instead of the matrix row, so it added the wrong
  weights for every hidden unit.  The network still "worked" — it produced
  numbers — which is why the only thing that caught it was asserting that the
  PyTorch and C++ forward passes agree.  `scripts/verify_nnue.py` exists for
  exactly this.

## Making the network fast enough to matter

A network that is stronger per evaluation but three times slower is not
stronger at all: the search simply visits a third of the nodes.  The first
working version rebuilt the feature transformer from scratch at every node,
and the layout made that as slow as it could possibly be.

The feature transformer is stored **transposed**: feature `f` owns a
contiguous run of `ft_hidden` weights instead of the hidden unit owning a
contiguous run of `FT_INPUT` weights.  In the natural layout a single piece
update strides through the entire 250 KiB matrix and touches one cache line per
hidden unit; transposed, it touches two contiguous 256-byte runs.

On top of that the search no longer rebuilds the accumulators at all.  It
updates them incrementally as it descends and un-descends the tree, so an
evaluation is the forward pass and nothing else.  The search therefore pays
about one piece update per edge instead of one full rebuild per node.

This is the kind of change that is easy to get subtly wrong, so it is checked
rather than trusted: `engine/tools/nnue_check.cpp` walks random legal games
and compares the incrementally maintained accumulators against a full refresh
at every position, including after unmaking moves.  The two must agree
*exactly* — not approximately, not within a tolerance.

## Quantisation

The network is trained with the int8 weights and the rounding and clipping
that the C++ inference uses, via a straight-through estimator.  Training a
plain float network and quantising afterwards loses a noticeable amount of
accuracy, and it also means the exported network is not quite the network
that was trained.

The activation scale needed careful handling.  Because the activations are
clipped to [0,127], the *magnitude* of the weights decides where they land,
and the magnitude needed differs by five orders of magnitude between the
feature transformer (whose input is binary) and the output layer.  Each layer
therefore carries a learnable gain, folded into the weights at export time.
