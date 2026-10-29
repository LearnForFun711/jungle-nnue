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

## Three bugs worth remembering

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

## The training pipeline could not learn at all

The most expensive bug in the project, and the hardest to see, because the
symptom looks like an ordinary "the model is not very good": the loss
plateaus at the variance of the target and the network stays flat.  Three
separate faults each on their own were enough to do that.

**The straight-through estimator was broken.**  The estimator that carries the
gradient through the int8 rounding used the *rounded* value in its
straight-through term.  `torch.round` has no derivative, so the gradient was
identically zero for every hidden layer — only the output layer trained at
all.  The fix is to carry the *pre-round* value:

```python
r = torch.round(x)
r = r + (x - x.detach())     # forward: round(x), gradient: 1
```

**One learning rate could not train all four layers.**  The weights are
quantised to int8 per output row, so the smallest change that affects the
forward pass is `max|W|/127`.  The weight magnitudes in this network span
five orders of magnitude — the feature transformer needs large weights
because its input is binary, the output layer needs tiny ones because its
output is in [-1,1] — and the quantisation steps span the same factor.
Adam moves every parameter by roughly `lr` per step regardless of its
gradient, so a single learning rate left the feature transformer crawling
and the output layer thrashing violently.  Scaling the learning rate with
the layer's weight magnitude makes `lr/step` the same everywhere, which is
what makes one `--lr` meaningful at all.

**Half of every layer was dead on arrival.**  The hidden activations are
clipped to [0,127] and the pre-activation of a zero-mean weight matrix is
zero-mean, so about 51% of the units fell below zero and were clipped away.
A clipped unit produces no gradient, so it could never come back.  Every
layer now has a learnable per-unit bias, and the initial weight scales are
set from the expected input statistics so the activations start in the
middle of the range.

The effect, measured on a held-out shard of the dataset:

| | correlation with the game outcome |
|---|---|
| the handcrafted evaluation (the thing to beat) | +0.161 |
| the MCTS teacher the network was distilled from | +0.160 |
| the network, before these three fixes | **−0.02** |
| the network, after them | **+0.187** |

The distilled network now predicts game outcomes *better than the teacher it
was distilled from*, which is what a value function is supposed to do: the
teacher's estimate of a position's value is noisy, and regressing on it
removes that noise.

`train/train.py` now refuses to start if any parameter receives no gradient.
That check costs a few milliseconds and would have caught this in seconds
instead of hours.

## The clamp needs a real gradient too

The straight-through estimator has two halves and both of them matter.  The
*rounding* has to pass the gradient through; the *clamping* must not.

Letting the gradient through the clamp as well looks harmless — the forward
value is still clamped — but it is not: a unit that is pinned at the top of
the range keeps being pushed upwards, the layer's weights grow to
compensate, and because the learning rate is scaled with the weight
magnitude the whole thing runs away.  In one run the feature transformer's
weights grew by a factor of 200 and every unit in the layer was clipped:
the validation loss climbed from 0.0244 to 0.0352 and kept going.

With a proper clamp a saturated unit produces no gradient, so it cannot be
pushed any further, and the runaway cannot start.

## The network should be small

The obvious instinct is that a bigger network is a better network.  Measured
on a held-out shard, that is not what happens here:

| network | MACs per evaluation | correlation with the game outcome |
|---|---|---|
| 96 / 48 | 14k | **+0.179** |
| 128 / 64 | 25k | +0.178 |
| 256 / 128 | 99k | +0.175 |

The smallest network is the most accurate *and* seven times cheaper.  With
three million training positions and a hundred thousand parameters there is
enough data to support a small network and not enough to justify a large
one.

That matters far more than the accuracy, because of what happens at the
search level.  At one second per move the neural engine spends essentially
all of its time evaluating: with the 256/128 network it visits 134,000 nodes
per move against the handcrafted engine's 1,400,000 — a ten-fold handicap
that no amount of evaluation accuracy can overcome.  The same measurement
with a small network has a handicap of under two, which the network's better
prediction of the outcome can pay for.

This is the same trade-off Stockfish makes, and it is why the width of a
network is chosen by playing strength rather than by validation loss.

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

## A side with no legal move is not a mate

This one cost more playing strength than anything the network did.

The search scored "the side to move has no legal move" as a mate.  In this
game that is a loss, but it is usually a loss the opponent has to work
several moves for, and the two consequences of scoring it as a mate were
both fatal.

The first was mechanical: the iterative deepening stops as soon as it finds
a mate score, so the search gave up several plies early.  The symptom was
that *more time made the search weaker*.  The depth reached fell from 16.9
at 0.1 seconds per move to 11.9 at 1 second, while the node count went up
sevenfold — the opposite of what a longer clock should do.

The second was that the two engines disagreed about how bad the position
is.  The handcrafted evaluation returns -30000 for a side with no pieces,
which is a mate-sized score; the network returns about -120, which is
nothing at all next to a repetition worth exactly zero.  The network
engine was therefore content to take a draw rather than fight for its
pieces, and the handcrafted engine was not.

Both now score it the same way: a very large loss, but below the mate
score, so the search keeps thinking and the two engines agree on how bad it
is.  At 0.1 seconds per move the score against the handcrafted engine went
from 0.438 to 0.667 and the depth from 14.9 to 16.9.

## The network's value is compressed, and the search notices

The network's value is defined on [-1,1] and uses about a tenth of it: its
standard deviation over real positions is 0.06, against the handcrafted
evaluation's 0.19.  The search compares scores with a one-unit window and a
repetition is worth exactly zero, so a compressed evaluation reads "slightly
behind" where a wide one reads "clearly behind" — and the engine gives up on
positions it thinks it is only slightly losing in.  The value is multiplied
by ten before it is used as a search score, and the multiplier is a command
line option so it can be tuned per network.

## The teacher is only as good as its leaf

The teacher's values correlate with the result of the game by about 0.14,
whether the leaf is the handcrafted evaluation or a network.  The network's
own value correlates by 0.36.  So the teacher dilutes the signal it is
given: the search averages the values of the children, and that average is
worse than the network's own opinion of the position.

This is the ceiling of the whole project.  More GPU time does not raise it.
What raises it is a network that predicts the outcome better, and the
training target blends the teacher's value with the result of the game for
exactly that reason — the result is the only ground truth there is.

## The search found mates it did not play

The search returns its move through `root_best_`, a member that remembers the
best move at the root.  It was set on one path through the move loop and not
on the other, so a mate found the other way — a piece that can step into the
opponent's den — set the *score* to a mate and left `root_best_` holding
whatever move the previous turn preferred.

So the engine would report a mate and play a stale move, which could be
illegal in the current position.  In the games this looked like the neural
engine making illegal moves and the game getting stuck; in the match harness
it looked like the engine throwing away wins it had already found: no game
in the four hundred game match ended with a piece entering the opponent's
den, because every one of those mates was found and then not played.

Two things had to be fixed: the root's best move is recorded on both paths,
and it is reset at the start of every search, because a move belongs to the
position it was found in.

## A mate three plies down is still a mate

The score of a mate is `MATE_SCORE - ply - 1`, so a mate found three plies
down scores 29997 rather than 29999.  The iterative deepening stopped when
`abs(score) >= MATE_SCORE - 1`, so it did not stop for that mate: the search
spent its entire time budget proving the same mate over and over.

That is why *more time made the search weaker*.  The depth reached fell from
16.9 at 0.1 seconds per move to 11.9 at 1 second, while the node count went
up sevenfold.  The iterations after the mate was found were all wasted, and
the more time there was, the more of them there were.  The threshold now
allows for a mate at any depth.

With the two search bugs fixed and the value scaled, the same network that
lost every decisive game at one second per move wins them:

| seconds per move | before | after |
|---|---|---|
| 0.1 | 3 W / 6 L / 15 D | **12 W / 0 L / 12 D** |
| 0.3 | 0 W / 1 L / 23 D | **17 W / 0 L / 7 D** |

and the games are now won by entering the opponent's den, which is how the
game is supposed to be won.
