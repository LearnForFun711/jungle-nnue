#!/usr/bin/env python3
"""A population-based self-play reinforcement loop.

    python3 scripts/rl_population.py --champion net/net.nnue --out runs/rl \
        --generations 10 --population 4

Why this exists
---------------

The pipeline it replaces trained one network on one dataset and shipped it.
Two things were wrong with that, and both were found by measuring rather than
by reasoning:

1. *The candidate was chosen by the lowest validation loss*, and that rule is
   systematically inverted: the validation loss is the mean squared error
   against a target that mixes the teacher's value with the game's result,
   and the game's result cannot be predicted from a single position, so the
   more of it there is in the target, the higher the loss floor of a perfect
   network.  "Lowest loss" therefore always meant "the candidate that copies
   the teacher most closely", and a copy of the teacher is never better than
   the teacher.

2. *The network was only ever taught to imitate the teacher*, so it could
   not be better than the teacher.  The network that is actually the strongest
   one in this project is the one that was trained when the teacher was weak:
   it had to learn the game's result on its own, and it ended up well above
   the teacher it was distilled from.

This loop fixes both.  It keeps a *population* of networks, plays them
against each other, and keeps the one that wins.  The fitness is the result
of the games -- not a loss, not a correlation -- so a candidate cannot be
selected for looking like the data.

The loop, one generation
------------------------

    1. League.  Every pair of candidates plays a number of games on the GPU,
       in both colours.  The games are played by the MCTS teacher with each
       side using *its own* network as the leaf evaluation, which is what the
       generator already does -- so the league needs two networks in one
       game, and the GPU engine has been taught to do that.

    2. Data.  The champion plays itself on the GPU.  The games are the
       reward: every position in a game carries the game's result, and the
       result is the only ground truth there is.

    3. Training.  Each candidate is trained on that data.  The target is the
       game's result with a configurable weight, the teacher's value with
       the rest: the teacher is a dense signal and a ceiling, the result is
       the truth and a noisy one.

    4. Mutation.  The candidates for the next generation are perturbations
       of the champion -- a little noise on the weights, and a different
       result weight and learning rate for each of them.  The champion
       itself is always kept, so a generation can never be worse than the
       one before it.

    5. Selection.  The league results decide who is the champion.

Everything that costs GPU time (the games) happens on the GPU, and the
training happens on the GPU.  The only thing that needs a CPU is the final
check against the handcrafted engine, and that is done once, at the end.

The league measures the networks as *teacher leaves*, which is not quite the
same thing as the alpha-beta engine that finally plays them.  It is a good
proxy -- it is the same evaluation, used by the same search -- but the
number that decides what ships is still the match against the handcrafted
engine, and the notebook runs that at the end.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# The league
# ---------------------------------------------------------------------------


def elo_from_score(score):
    """The Elo difference a score of 0.5 plus is worth."""
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


class League:
    """Plays the candidates against each other and rates them.

    The games are played by the MCTS teacher: each side searches with its own
    network as the leaf evaluation, so a better network shows up as a better
    player and not just as a better number.
    """

    def __init__(self, ext, cpu_tool, sims=400, leaf_scale=0.125,
                 value_scale=10.0, seed=1234, max_ply=200, games=None):
        self.ext = ext                  # the CUDA extension, or None
        self.cpu_tool = cpu_tool        # the portable CPU generator
        self.sims = sims
        self.leaf_scale = leaf_scale
        self.value_scale = value_scale
        self.seed = seed
        self.max_ply = max_ply
        self.games = games              # games per ordered pair, or None

    def _schedule(self, n_candidates, games):
        """Which pair plays in which game.

        Every ordered pair plays the same number of games, and each pair
        plays half of them with one network as Red and half with the other,
        so a network that is only good as Black cannot get a free ride."""
        red, black = [], []
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i == j:
                    continue
                for k in range(games):
                    if k % 2 == 0:
                        red.append(i); black.append(j)
                    else:
                        red.append(j); black.append(i)
        return red, black

    def play(self, nets):
        """Play a full round.  Returns (results_per_game, pairs).

        `results_per_game` is +1 when Red won, 0 for a draw and -1 when
        Black won; `pairs` is the (red, black) pair of each game."""
        n = len(nets)
        if n < 2:
            return [], []

        games = self.games
        red, black = self._schedule(n, games)
        if not red:
            return [], []

        if self.ext is not None:
            stats = self.ext.league(
                list(nets), red, black, len(red), self.sims, self.seed,
                self.leaf_scale, self.value_scale, 1.4, 1.6, 0.30, 0.25,
                self.max_ply, 0)
            results = list(stats["results"])
        else:
            results = self._play_on_cpu(nets, red, black)

        pairs = list(zip(red, black))
        return results, pairs

    def _play_on_cpu(self, nets, red, black):
        """The same league, on the CPU.

        It is slower, but it is the reference implementation and it is what
        runs when there is no GPU.  The games are played one at a time
        through the portable core, so this is the same code the CUDA build
        has to agree with."""
        results = []
        tmp = os.path.join(ROOT, "runs", "league-tmp")
        os.makedirs(tmp, exist_ok=True)
        for idx, (r, b) in enumerate(zip(red, black)):
            cmd = [self.cpu_tool, "--nets", ",".join(nets),
                   "--schedule", f"{r}:{b}", "--games", "1",
                   "--sims", str(self.sims), "--seed", str(self.seed + idx),
                   "--leaf-scale", str(self.leaf_scale),
                   "--value-scale", str(self.value_scale),
                   "--max-ply", str(self.max_ply),
                   "--out", os.path.join(tmp, "g")]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            res = self._parse_cpu_result(proc.stdout)
            results.append(res)
        return results

    @staticmethod
    def _parse_cpu_result(stdout):
        """The CPU tool prints the result; pull it out of the line.

        The line looks like
            pair 0:1  games 4  red wins 2  black wins 1  draws 1  score(red) 0.625
        """
        import re
        for line in stdout.splitlines():
            if not line.startswith("pair "):
                continue
            m = re.search(r"red wins (\d+)\s+black wins (\d+)\s+draws (\d+)", line)
            if not m:
                continue
            red_wins = int(m.group(1))
            if red_wins:
                return 1
            if int(m.group(2)):
                return -1
            return 0
        raise RuntimeError("the CPU league tool printed nothing useful:\n" + stdout)

    @staticmethod
    def rate(results, pairs, n_candidates):
        """Rate the candidates by their games.

        Two numbers come out: the score against the whole field, and an Elo
        rating computed from the same games the way a tournament computes
        it.  The score against the field is the one to watch -- it is what a
        round robin means -- and the Elo is there because it is the unit
        everybody quotes."""
        wins = [0.0] * n_candidates
        played = [0] * n_candidates
        rating = [1500.0] * n_candidates

        for res, (r, b) in zip(results, pairs):
            # From the point of view of the network that played Red.
            score_red = 1.0 if res > 0 else (0.5 if res == 0 else 0.0)
            score_black = 1.0 - score_red
            wins[r] += score_red
            wins[b] += score_black
            played[r] += 1
            played[b] += 1

            # The usual Elo update, one game at a time.
            expected_red = 1.0 / (1.0 + 10.0 ** ((rating[b] - rating[r]) / 400.0))
            rating[r] += 24.0 * (score_red - expected_red)
            rating[b] += 24.0 * (score_black - (1.0 - expected_red))

        out = []
        for i in range(n_candidates):
            score = wins[i] / played[i] if played[i] else 0.5
            out.append({"index": i, "score": score, "games": played[i],
                        "elo": rating[i], "elo_vs_field": elo_from_score(score)})
        return out


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def mutate_checkpoint(src, dst, sigma, seed):
    """A mutant: the champion's weights with a little noise on them.

    The noise is proportional to the size of each tensor, so a layer with
    large weights is perturbed as much as a layer with small ones.  The
    tensors that are a single number -- the per-layer gains -- are perturbed
    in absolute terms instead, because their relative size is meaningless.

    The mutant then trains on the data like everybody else, so it is not a
    random network -- it is the champion, moved a little, and then trained
    on the same games with a slightly different recipe."""
    import torch
    torch.manual_seed(seed)
    ck = torch.load(src, map_location="cpu")
    state = ck["model"]
    with torch.no_grad():
        for name, p in state.items():
            if not p.is_floating_point():
                continue
            if p.numel() > 1:
                scale = float(p.std())
                if not (scale > 0.0) or math.isnan(scale):
                    # A tensor that is all zeros: perturb it a little, in
                    # absolute terms.
                    scale = 0.1
            else:
                scale = 0.1
            p.add_(torch.randn_like(p) * (sigma * scale))
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(ck, dst)
    return dst


# ---------------------------------------------------------------------------
# The generation loop
# ---------------------------------------------------------------------------


class Loop:
    def __init__(self, args, ext, cpu_tool):
        self.args = args
        self.ext = ext
        self.cpu_tool = cpu_tool
        self.league = League(ext, cpu_tool, sims=args.league_sims,
                             leaf_scale=args.leaf_scale, value_scale=args.value_scale,
                             seed=args.seed, max_ply=args.max_ply,
                             games=args.league_games)
        self.history = []

    # -- data ----------------------------------------------------------

    def generate_data(self, champion_net, gen):
        """The champion plays itself, and the games become the data.

        The games are the reward: the result of the game is the only ground
        truth in the dataset, and the teacher's value is what the champion
        currently believes."""
        outdir = os.path.join(self.args.out, f"gen{gen:03d}")
        os.makedirs(outdir, exist_ok=True)
        shard = os.path.join(outdir, "data.t0.bin")
        if self.ext is not None:
            stats = self.ext.run(champion_net, shard, self.args.data_games,
                                  self.args.data_sims, self.args.seed + gen,
                                  self.args.leaf_scale, 10.0, 1.4, 1.6,
                                  0.30, 0.25, 200, 0)
            print(f"    data: {stats['games']} games, {stats['positions']} positions, "
                  f"{stats['seconds']:.1f}s on the GPU", flush=True)
        else:
            cmd = [self.cpu_tool, "--net", champion_net,
                   "--games", str(self.args.data_games), "--sims", str(self.args.data_sims),
                   "--seed", str(self.args.seed + gen),
                   "--leaf-scale", str(self.args.leaf_scale),
                   "--out", shard[:-4]]
            subprocess.run(cmd, cwd=ROOT, check=True)
            print(f"    data: {self.args.data_games} games (CPU)", flush=True)
        return os.path.join(outdir, "*.bin")

    # -- training ------------------------------------------------------

    def train_candidate(self, name, init_ckpt, data_glob, result_weight, lr, seed):
        """Train one candidate and export it."""
        outdir = os.path.join(self.args.out, "nets", name)
        os.makedirs(outdir, exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "train", "train.py"),
               "--data", data_glob, "--out", outdir,
               "--budget-minutes", str(self.args.train_minutes),
               "--batch", str(self.args.batch), "--lr", str(lr),
               "--ft-hidden", str(self.args.ft_hidden), "--hidden", str(self.args.hidden),
               "--result-weight", str(result_weight),
               "--target", self.args.target, "--sample-weight", self.args.sample_weight,
               "--seed", str(seed), "--threads", str(self.args.threads)]
        if self.args.huber > 0:
            cmd += ["--huber", str(self.args.huber)]
        if init_ckpt:
            cmd += ["--resume", init_ckpt]
        print(f"    training {name} (result weight {result_weight}, lr {lr})", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

        # The checkpoint that is the best *player* is the one with the best
        # correlation with the outcome, not the one with the lowest loss --
        # the lowest loss is not comparable between candidates that use
        # different targets.
        best = os.path.join(outdir, "best_corr.pt")
        if not os.path.exists(best):
            best = os.path.join(outdir, "best.pt")
        net = os.path.join(outdir, "net.nnue")
        subprocess.run([sys.executable, os.path.join(ROOT, "train", "export_nnue.py"),
                        "--checkpoint", best, "--out", net], cwd=ROOT, check=True)
        return net, best

    # -- one generation ------------------------------------------------

    def generation(self, gen, population):
        """One generation: league, data, training, mutation.

        `population` is a list of (name, net, checkpoint).  It comes back as
        the next generation's population, so nothing that was learned is
        thrown away.
        """
        args = self.args
        print(f"\n=== generation {gen} ===", flush=True)

        # 1. the league: every pair plays, in both colours, and the results
        #    rate the candidates.  The fitness is the result of the games.
        nets = [p[1] for p in population]
        results, pairs = self.league.play(nets)
        table = League.rate(results, pairs, len(nets))
        ranked = sorted(table, key=lambda r: -r["score"])
        print(f"    league over {len(results)} games:", flush=True)
        for row in ranked:
            name = population[row["index"]][0]
            print(f"      {name:<22} score {row['score']:.3f}  "
                  f"({row['games']} games)  Elo {row['elo']:.0f}", flush=True)

        # 2. the champion is the one that won.  It is kept untouched, so a
        #    generation can never be worse than the one before it.
        champion = population[ranked[0]["index"]]
        champion_net, champion_ckpt = champion[1], champion[2]

        # 3. the data: the champion plays itself.  The games are the reward.
        data_glob = self.generate_data(champion_net, gen)

        # 4. training.  Every member is trained on the new data, each
        #    resuming from its own checkpoint, and each with a slightly
        #    different recipe.  The last member is a fresh mutant of the
        #    champion: something new is always in the population.
        trained = []
        for i, (name, net, ck) in enumerate(population):
            if i == len(population) - 1:
                name = f"explorer{gen}"
                ck = os.path.join(args.out, "nets", f"explorer-gen{gen:03d}.pt")
                net = os.path.join(args.out, "nets", f"explorer-gen{gen:03d}.nnue")
                mutate_checkpoint(champion_ckpt, ck, args.mutation_sigma,
                                  args.seed + gen * 100 + i)
                subprocess.run([sys.executable, os.path.join(ROOT, "train", "export_nnue.py"),
                                "--checkpoint", ck, "--out", net], cwd=ROOT, check=True)
            rw = min(0.95, args.result_weight + args.result_weight_step * i)
            lr = args.lr * (args.lr_step ** i)
            new_net, new_ck = self.train_candidate(f"g{gen:03d}-{i}", ck, data_glob,
                                                   rw, lr, args.seed + gen * 1000 + i)
            trained.append((f"g{gen:03d}-{i}", new_net, new_ck))

        # 5. the next population: the champion, untouched, and the trained
        #    candidates.  The size stays the same.
        next_population = [champion] + trained[:max(0, len(population) - 1)]

        self.history.append({
            "generation": gen,
            "league": [{"name": population[r["index"]][0], "score": r["score"],
                        "elo": r["elo"], "games": r["games"]} for r in ranked],
            "champion": champion[0],
            "champion_score": ranked[0]["score"],
            "champion_elo": ranked[0]["elo"],
        })
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump({"generations": self.history}, f, indent=2)

        return next_population


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="net/net.nnue")
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--league-games", type=int, default=8,
                    help="games per ordered pair in a league round")
    ap.add_argument("--league-sims", type=int, default=400)
    ap.add_argument("--data-games", type=int, default=512)
    ap.add_argument("--data-sims", type=int, default=400)
    ap.add_argument("--train-minutes", type=float, default=10.0)
    ap.add_argument("--ft-hidden", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--result-weight", type=float, default=0.75)
    ap.add_argument("--result-weight-step", type=float, default=0.05)
    ap.add_argument("--target", default="blend", choices=("blend", "outcome"))
    ap.add_argument("--sample-weight", default="both",
                    choices=("none", "decisive", "late", "both"))
    ap.add_argument("--huber", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr-step", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--mutation-sigma", type=float, default=0.05)
    ap.add_argument("--leaf-scale", type=float, default=0.125)
    ap.add_argument("--value-scale", type=float, default=10.0)
    ap.add_argument("--max-ply", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--cpu-only", action="store_true",
                    help="use the portable CPU generator instead of the GPU")
    args = ap.parse_args()

    os.chdir(ROOT)
    os.makedirs(args.out, exist_ok=True)

    ext = None
    cpu_tool = None
    if not args.cpu_only:
        sys.path.insert(0, os.path.join(ROOT, "engine", "gpu"))
        try:
            import gpu_selfplay  # noqa: E402
            ext = gpu_selfplay.build_extension(verbose=False)
        except Exception as exc:
            print(f"the GPU extension is not available ({exc}); "
                  f"falling back to the CPU generator", file=sys.stderr)
    if ext is None:
        cpu_tool = os.path.join(ROOT, "engine", "buildgpu", "gpu_selfplay_cpu")
        if not os.path.exists(cpu_tool):
            subprocess.run(["make", "-C", os.path.join(ROOT, "engine"),
                            "BUILD=buildgpu", "buildgpu/gpu_selfplay_cpu"], check=True)

    champion_net = args.champion
    champion_ckpt = os.path.join(args.out, "champion.pt")
    if not os.path.exists(champion_ckpt):
        # The champion starts as a checkpoint we can mutate.
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "net_to_checkpoint.py"),
                        "--net", champion_net, "--out", champion_ckpt], cwd=ROOT, check=True)

    # Every candidate in a population has the same shape: the league plays
    # them against each other, and the GPU's forward pass has one shared
    # memory slice for all of them.  So the shape comes from the champion.
    import torch
    ck = torch.load(champion_ckpt, map_location="cpu")
    shape = (int(ck.get("ft_hidden", 96)), int(ck.get("hidden", 48)))
    if (args.ft_hidden, args.hidden) != shape:
        print(f"the population uses the champion's shape {shape[0]}/{shape[1]}")
        args.ft_hidden, args.hidden = shape

    loop = Loop(args, ext, cpu_tool)
    # The starting population: the champion, and mutants of it.  A mutant is
    # the champion with a little noise on its weights -- not a random
    # network, but something close enough to the champion that a few minutes
    # of training puts it back in the same basin.
    population = [("champion", champion_net, champion_ckpt)]
    for m in range(args.population - 1):
        ck = os.path.join(args.out, "nets", f"mutant-{m}.pt")
        net = os.path.join(args.out, "nets", f"mutant-{m}.nnue")
        if not os.path.exists(net):
            mutate_checkpoint(champion_ckpt, ck, args.mutation_sigma * (0.5 ** m),
                              args.seed + m)
            subprocess.run([sys.executable, os.path.join(ROOT, "train", "export_nnue.py"),
                            "--checkpoint", ck, "--out", net], cwd=ROOT, check=True)
        population.append((f"mutant{m}", net, ck))

    for gen in range(args.generations):
        population = loop.generation(gen, population)

    print(f"\n==> the loop is finished.  History in {args.out}/history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
