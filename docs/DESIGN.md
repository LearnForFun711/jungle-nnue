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

* **The match harness had the same inversion, and nobody noticed.**  The bug
  above was in the data generator; the same mistake was made again, years of
  work later, in `engine/tools/match.cpp` -- and it survived because a match
  result has no internal cross-check.  The teacher's values were
  anti-correlated with the outcome, which is obvious.  A match that says
  "the neural engine won 202 and lost 13" looks exactly like a match in
  which the neural engine won 202 and lost 13.

  The fix and the guard are described in
  [the match harness reported the loser](#the-match-harness-reported-the-loser)
  below.  The short version: the game loop now lives in one place
  (`engine/src/match_game.hpp`), and `engine/tools/game_check.cpp` plays
  games through the same function the harness calls and compares what it
  reports against the winner taken from the board.

* **The feature transformer was indexed as a flat array.**  The weights are
  stored as `[FT_INPUT][ft_hidden]` (transposed, see below), and the accumulator
  update walked the flat array instead of the matrix row, so it added the wrong
  weights for every hidden unit.  The network still "worked" — it produced
  numbers — which is why the only thing that caught it was asserting that the
  PyTorch and C++ forward passes agree.  `scripts/verify_nnue.py` exists for
  exactly this.

## The match harness reported the loser

Every match result this project produced before 2026-11-05 was upside down.

`engine/tools/match.cpp` decided who had won a game like this:

```cpp
const bool den_entry = (terrain().den_owner[m.to] == opposite_color(pos.stm));
make_move(pos, m);
if (den_entry)
    return {pos.stm == RED ? +1 : -1, pos.ply, END_DEN, pos.stm == RED ? 1 : 0};
```

`make_move()` flips the side to move, so after the move `pos.stm` is the side
that *lost* -- the mover has just entered the opponent's den and won.  The
harness therefore recorded the loser as the winner, and it did so for every
game that ended the way games in this game end: by a den entry.

The data generator had the identical bug once, and it was caught because a
teacher whose values are anti-correlated with the outcome is visibly wrong.
The match harness had no such tell.  "202 wins, 13 losses, 185 draws" is a
perfectly plausible score for a strong engine; it just happened to be the
score of a weak one.

The fix separates the two things the old code conflated:

* `GameResult::red_won` -- the truth, taken from the board.  It is 1 when the
  side that moved into the opponent's den was Red.
* `GameResult::result` -- the reported value, from Red's point of view, taken
  from the side to move after the move.

They are computed from different sources on purpose, so that they can be
compared.  `engine/tools/game_check.cpp` plays games through the very same
`play_game()` that the harness calls and checks the two against each other:

```
games                              30
decisive                           30   (den 30, no-moves 0)
harness agrees with the board      30
harness disagrees with the board  0
```

Before the fix: 30 of 30 disagreed.  It is a CI step now.

The game loop also moved out of the harness into `engine/src/match_game.hpp`,
because the bug was a *drift* bug: the generator and the harness each had
their own copy of the same rule, and the copies disagreed.  One copy cannot
drift.

What this changes is everything that was measured with the harness: the
acceptance numbers, the Elo differences, the "the network at this scale wins
every decisive game" note in `engine/src/nnue.hpp`, and the value scale
tuning that went with it.  The measurements that did not go through the
harness -- the teacher's values, the network's correlation with the game
outcome, the parity tests -- are unaffected, because they compare the
network and the teacher against each other rather than against a match.

The general lesson: a number that comes out of a program needs a check that
does not come from the same program.  The harness had none, and its output
was confidently wrong for a long time.

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

The teacher's values are what the network learns from, so their quality is the
ceiling on the network's quality.  Every position in a shard carries the value
the teacher gave it and the result of the game it came from, and the
correlation between the two is the teacher's quality.
`engine/tools/teacher_probe.cpp` measures it by running the teacher over a
fixed sample of those positions, so that two configurations are always
compared on the same positions.

Measured over 4,000 positions of a fresh shard, with the same 800
simulations per move that the data is generated with:

| teacher's leaf | scale | corr(value, result) | on decisive games | spread |
|---|---|---|---|---|
| handcrafted | — | +0.189 | +0.238 | 0.192 |
| network, mean over the opponent's replies | 1.0 | +0.150 | +0.223 | 0.563 |
| network, mean over the opponent's replies | 0.5 | +0.173 | +0.247 | 0.466 |
| network, mean over the opponent's replies | 0.25 | +0.210 | +0.276 | 0.334 |
| **network, mean over the opponent's replies** | **0.125** | **+0.236** | **+0.290** | 0.202 |
| network, mean over the opponent's replies | 0.0625 | +0.241 | +0.277 | 0.121 |
| network + a quarter of the handcrafted evaluation | 0.125 | +0.227 | +0.279 | 0.198 |

The earlier version of this table was measured with a race in the measuring
tool (see below) and put the network at +0.209 against the handcrafted
teacher's +0.147.  The ordering survived the fix; the gap did not: with the
race gone, the network's leaf at the right scale is 25% more informative than
the handcrafted one, and at the engine's own scale it is worse than it.

The scale is a lever of its own, and it is the one that is easiest to get
wrong.  The engine multiplies the network's value by ten before it is used
as a search score (`Evaluator::value_scale`), because the network only uses a
small part of its range and a compressed evaluation makes the search give up
on positions it thinks it is slightly behind in.  That reasoning is about
the *search*; the teacher's leaf is a different quantity, and at a scale of
ten three quarters of the teacher's values come out at the clamp of [-1,1].
The spread of the values in the table is the tell: 0.192 for the handcrafted
teacher, 0.563 for the network at scale one.  A teacher whose values are
mostly at the ends of the range is not measuring the position, and the
correlation with the result falls to 0.150 -- below the handcrafted teacher's.

Mixing the handcrafted evaluation back into the leaf makes it worse
(+0.227 against +0.236).  The handcrafted evaluation is not a useful
correction to a network trained on the same game; it is noise added to a
signal.

Three things follow from that table.

The search is not what makes the teacher's values good; the leaf is.  With the
handcrafted evaluation, one ply of search over the opponent's replies moves
the correlation by three thousandths: the teacher is barely better than the
static evaluation it sits on top of.  With the network as the leaf, the same
search is worth six hundredths, and the teacher's values are 39% more
informative than the handcrafted ones.

The "best reply" aggregation is worse than the mean, and with a network it is
much worse.  The network's value uses only a small part of the [-1,1] range,
so the maximum over the opponent's thirty replies is decided by the luckiest
of thirty samples and then saturates at the clamp.  Twenty per cent of the
values end up outside [-0.5, 0.5] and their mean is dragged to -0.37: the
aggregation has stopped measuring the position and started measuring the tail
of the distribution.  The mean keeps the distribution's scale and stays
informative.

The scale of the leaf matters for the same reason.  The network's value is
clamped before it is aggregated, and clamping hard enough destroys the
ordering it was supposed to preserve: the correlation falls from +0.209 at
scale 1 to +0.198 at scale 10, where six per cent of the values are clipped.

This is the single largest lever the project has, and it is why the Colab
notebook puts a network into the teacher half an hour into the session rather
than after it.

## The evaluator was shared between threads

`selfplay` and `teacher_probe` created one `Evaluator` and handed it to
every thread.  The evaluator holds the network's feature accumulators, and
those are changed while a position is evaluated: the leaf evaluation
refreshes them for the parent, moves a piece, reads the value, and moves the
piece back, once for every legal move.

With two threads evaluating at the same time, each one reads whatever the
other has half-written.  The teacher's values came out as a mixture of
several positions' worth of arithmetic, and the mixture was different every
run.  The handcrafted evaluation has no such state -- it is a pure function
of the position -- so the handcrafted teacher was never affected, which is
why the bug hid for so long: the handcrafted path worked, and the neural
path did not, and the difference was read as "the neural leaf is not worth
it".

Each thread has its own evaluator now.  The numbers in the table above are
from after the fix.

There is a second thing about this worth writing down.  The check that found
it was not a test that failed; it was a *disagreement* between two
implementations of the same thing.  The GPU teacher's values were compared
with the CPU teacher's, position by position, and they did not match --
because the CPU teacher's values were the racy ones.  A parity test between
two implementations finds bugs in both of them.

## The teacher on the GPU

The teacher is the bottleneck of the project: the network is small and trains
in minutes, but the data it learns from takes days to generate on a CPU,
because every node the search expands needs the network evaluated once for
every legal move.

The teacher can run on the GPU instead, and `engine/gpu/` holds it in a form
that compiles both as CUDA and as plain C++:

* the move generator is written with bitboards.  Every legal move of every
  piece of a type comes out of one shift and a few masks, which is what makes
  it cheap enough to run a thousand games' worth at once;
* the network's forward pass is the engine's own arithmetic, arranged so that
  a warp evaluates one position at a time: the feature accumulators live in
  the lanes' registers and the hidden layers read the activations out of
  shared memory.  A warp evaluating a position on its own would leave
  thirty-one lanes idle;
* the search is a state machine.  A simulation can stop in the middle, when a
  node is expanded and the value of the position has to be looked up, and
  carry on once the value arrives.  That is what lets a thousand games ask
  for their leaf evaluations at the same time and have them answered in one
  batch;
* the game loop writes the same 68-byte records the CPU generator writes, so
  nothing downstream knows or cares where the data came from.

Three details are worth writing down because getting them wrong is easy and
the symptom is subtle.

**The captured piece has to be put back.**  A search that walks down the
tree and then walks back up has to undo the moves it made, and undoing a
capture means putting the captured piece back on the square.  The first
version of the port forgot, and the symptom was not a crash: the board after
every search had pieces missing, so the game that followed was played on a
board that was not the board.  The games were still legal-looking, and the
teacher's values were still plausible, and everything was wrong.

**A tree that does not know that a den entry wins is a different game.**  The
CPU teacher marks a child that enters the opponent's den as lost, and its
one-ply leaf evaluation returns 1.0 the moment it sees such a move.  A tree
that does not know this will happily search past a winning move.

**The order of the children decides ties, and therefore games.**  The CPU
teacher picks the first child with the best PUCT score and the first child
with the most visits, and both of those are ties that the creation order
decides.  The children are therefore created in the order the priors rank
them and kept in a list in that order.

With the port checked value by value against the CPU teacher -- the root's
noise turned off, which makes both of them deterministic -- the two agree on
every position: 300 out of 300 leaf values identical, root values identical
to six decimal places.

How many games to run at once is measured rather than chosen.  The games
finish at different times, and the batch has to wait for the slowest lane in
the warp, so past a point more games in flight stop helping.  The auto-tuner
asks the compiled code how much video memory a game needs, works out how many
fit in the card, and then runs each candidate number and counts the positions
per second.

## What the network actually learns from

The training target for a position is a blend of the teacher's value and the
result of the game:

    target = 0.75 * teacher_value + 0.25 * game_result

The trained network's correlation with the game outcome is +0.356, and the
blend's own correlation with the game outcome is 0.75 x 0.150 + 0.25 x 1.000
= 0.362.  The two agree to within the noise, which says the network predicts
its target about as well as the target can be predicted: it has learned the
blend.

That is worth spelling out, because it says where the strength comes from.
A third of the target's information about the outcome comes from the
teacher's search and two thirds from the game's result — and the result is
the only ground truth there is.  Raising the teacher from 0.150 to 0.209
lifts the target's correlation with the outcome from 0.362 to 0.407, so the
work does compound; but no amount of teacher quality gets past the fact that
the teacher is estimating a quantity the game's result already answers
exactly.

## A league needs two networks in one game

The reinforcement loop keeps a population of networks and plays them against
each other, so the generator had to learn to play a game in which the two
sides use *different* networks.

That is a small change in principle: the leaf evaluation asks the network
for the value of a position from the point of view of the side to move, so
the network it asks is the one the side to move plays with.  In the CUDA
kernel that means the request carries the network to evaluate it with, and
the kernel holds an array of networks:

```cpp
const float value = nnue_eval_batch(*nets[r.net_id], r.board, int(r.stm), ...);
```

The accumulators of the forward pass live in registers, not in the network,
so two networks can be evaluated at the same time without a race, and the
only thing the kernel needs is the right pointer.

Two things did need care:

* **The shared memory is sized for one shape.**  Every network in a league
  has to have the same shape, because the slice the forward pass uses is
  laid out for one network's width.  The host refuses to start a league
  whose networks differ, rather than quietly reading past the end of the
  slice.
* **The game loop and the kernel must not drift apart.**  The request is
  filled in by `Game::fill_request()`, in the portable core, and the CPU
  reference build uses the same function.  The same mistake as the one that
  inverted the match harness -- two copies of the same rule that disagree --
  would have been much harder to find here, because a league that rates the
  candidates backwards still produces a plausible-looking table.

So the league has its own check, `scripts/verify_league.py`, and it does not
use a sign convention at all: it plays a round, reads the records back, and
works out who won from the *board* -- the last record of a game is the
position the winner moved from, and the winner is the side that can step
into the opponent's den.  The league's own answer and the board's answer
have to agree on every game.

### The league and the engine agree

The league measures the networks as teacher leaves, which is not quite the
same thing as the alpha-beta engine that plays them.  Measured on the two
networks this project has, the two agree:

| | old network's score |
|---|---|
| the alpha-beta engine, 200 games at 0.3 s/move | 0.603 (Elo +72) |
| the league, 40 games of the MCTS teacher | 0.650 |

The league is a good proxy, and it is the one that can be run by the thousand
on a GPU.  The number that decides what ships is still the match against the
handcrafted engine.

## The population loop

`scripts/rl_population.py` runs the whole thing as a loop over generations,
and `colab/rl_population.ipynb` is the Colab version of it.

One generation:

1. **League.**  Every pair of candidates plays, in both colours.  The
   candidates are rated by the games.
2. **Data.**  The champion plays itself.  The games are the reward: every
   position in a game carries the game's result, and the result is the only
   ground truth there is.
3. **Training.**  Every candidate is trained on that data.  The target is
   the game's result with a configurable weight, the teacher's value with
   the rest.
4. **Mutation.**  The last member of the population is a fresh mutant of the
   champion -- the champion's weights with a little noise on them -- so
   something new is always in the population.
5. **Selection.**  The league decides who the champion is.  The champion is
   kept untouched, so a generation can never be worse than the one before it.

Two things about this are worth stating plainly.

**The fitness is the result of the games.**  Not a loss, not a correlation.
The rule that shipped the wrong network was a rule about how well a
candidate looked like the data, and it was wrong; the games are the thing
that is actually being optimised.

**The loop cannot go past the teacher on its own.**  A network distilled from
a teacher converges to that teacher, and the teacher is a ceiling.  The
strongest network in this project is the one that was trained when the
teacher was weak: it had to learn the game's result on its own, and it ended
up above the teacher it was distilled from.  So the loop's job is to keep the
reward heavier than the teacher's value, and to select by playing strength.
Going *past* the fixed point needs the policy -- the visit counts of the
search, learned and used to guide the next search -- and that is a bigger
change to the network format and to the engine than this loop makes.

### A hang worth remembering

The training loop checks its time budget inside the batch loop.  A dataset
smaller than one batch produces no batches, so the check is never reached
and the run loops forever.  The budget is now checked at the top of the
epoch as well, and a dataset smaller than one batch still gets one batch,
because a run that does nothing looks exactly like a run that is not
learning.

## A network distilled from a strong teacher is the teacher

A second network was trained later, on data generated by the GPU teacher,
and it came out *weaker* than the first one: +11.6 Elo against the
handcrafted engine where the first one was +231.9 on the same 60 games at
0.3 s/move.  The cause is visible in two numbers.

Measured on the same positions:

| | correlation with the game outcome |
|---|---|
| the GPU teacher's own values | +0.342 |
| the network distilled from them | +0.341 |
| the handcrafted evaluation | +0.206 |
| the first network | +0.323 |

The distilled network reproduces its teacher's outcome-correlation to within
a thousandth.  It is a faithful copy of the teacher, and a faithful copy of
the teacher cannot be better than the teacher.  The first network, trained
when the teacher was much weaker (+0.150), sat far *above* its teacher,
because the teacher's values did not contain the answer and the game's
result had to be learned from the data.

So the strength of a distilled network is not the strength of its teacher.
It is the distance between the teacher and the ground truth, and how much of
that distance the network managed to cover.  A better teacher narrows the
gap the network has to cover, and a network trained on a good teacher with a
strong result term converges to the teacher.  Improving the teacher is
therefore not automatically an improvement to the network.

The practical consequence is that the *result* term has to carry weight the
teacher cannot supply, and the candidate has to be chosen on the criterion
that measures it -- see below.

## The rule that shipped the wrong network

`train/train_budget.py` used to export the candidate with the lowest
validation loss.  That rule is not a weak proxy, it is systematically
inverted.

The validation loss is the mean squared error against the target, and the
target depends on the configuration's `result_weight` `w`:

    target = (1 - w) * teacher_value + w * game_result

The game result is a step function at the end of the game.  No network can
predict it from a single position, so the more of it is in the target, the
higher the MSE floor of a *perfect* network.  The target variance on the
training data:

| `w` | 0.00 | 0.25 | 0.50 | 0.75 |
|---|---|---|---|---|
| target variance | 0.034 | 0.048 | 0.102 | 0.198 |

"Lowest validation loss" therefore means "smallest `w`", always: the
candidate that imitates the teacher most closely.  The teacher is a
ceiling, so that candidate is the weakest one, and the pipeline was
discarding exactly the network that had been trained to look at who won.

Two things follow, and both are now implemented.

*The selection is made on a comparable metric.*  The correlation between the
network's value and the game outcome is measured on the same validation
positions for every candidate, does not depend on the target blend, and
tracks playing strength.  `train/train_budget.py --select-by corr` is the
default.  A two-minute run of two configurations on a small shard shows the
inversion exactly:

| configuration | validation loss | correlation with the outcome |
|---|---|---|
| `w=0.25` | 0.0156 | +0.531 |
| `w=0.75` | 0.1101 | +0.610 |

The old rule picks the first row.  The second row is the better player.

*Both checkpoints are kept.*  A network can have a falling validation loss
and a falling correlation at the same time, so `train/train.py` now writes
`best_corr.pt` next to `best.pt`, and the notebook plays the correlation
checkpoint.

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
