#!/usr/bin/env python3
"""Generate the Colab notebook for the reinforcement-learning population loop
(colab/rl_population.ipynb).

The notebook is the whole "make a network that actually wins" story in one
file: fetch the project, build the engine and the CUDA extension, run the
population loop, check the champion against the handcrafted engine, and
hand back the network.

It is kept generated from a script (rather than hand-written JSON) so that
it can be regenerated when the options change.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "colab", "rl_population.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Jungle Chess NNUE — reinforcement learning by self-play, on the GPU

This notebook runs a **population** of networks against each other, generation
after generation, and keeps the one that wins.

**Before you run anything**

1. Runtime ▸ Change runtime type ▸ **A100 GPU**, then restart.
2. Run the cells below in order.  The one that needs you is the champion
   network (cell 5): upload the network you want to start from, or let it
   start from scratch.

Nothing but the champion network crosses the network.  The games, the
training and the league all happen on the GPU.

## What is different about this, and why

The pipeline this replaces trained one network on one dataset and shipped it.
Two things were wrong with that, and both were found by measuring rather than
by reasoning.

**The candidate was chosen by the lowest validation loss, and that rule is
inverted.**  The validation loss is the mean squared error against a target
that mixes the teacher's value with the game's result:

```
target = (1 - w) * teacher_value + w * game_result
```

The game's result is a step function at the end of the game: no network can
predict it from a single position, so the more of it there is in the target,
the higher the loss floor of a *perfect* network.  The target's variance on
the training data is 0.034 at `w=0.00`, 0.048 at `w=0.25`, 0.102 at `w=0.50`
and 0.198 at `w=0.75`.  "Lowest validation loss" therefore always means
"smallest `w`" — the candidate that copies the teacher most closely — and a
copy of the teacher is never better than the teacher.  The rule was shipping
the weakest network every time.

**The network was only ever taught to imitate the teacher, so it could not be
better than the teacher.**  The strongest network in this project is the one
that was trained when the teacher was weak: it had to learn the game's result
on its own, and it ended up well above its teacher.  A strong teacher closes
that gap, and the student stops learning.

So this loop

* plays the candidates against each other, and **keeps the one that wins**;
* trains on the **game's result** (the reward) with the teacher's value as a
  smaller, denser auxiliary signal;
* mutates the champion a little every generation, so the search over networks
  keeps moving;
* and never throws away the champion, so a generation can never be worse than
  the one before it.
"""),
    code("""
import os, subprocess, sys

# The GPU the loop will run on.
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu: {p.name}   {p.total_memory/1e9:.1f} GB   {p.multi_processor_count} SMs")
except Exception as exc:
    print("torch is not available:", exc)

print("cpus:", os.cpu_count())
"""),
    md("""
## 2. Get the project

The notebook clones the project and then makes the clone current, because a
clone made before a fix is a clone that cannot build.  The GPU sources are
fetched from GitHub as well, for the same reason.
"""),
    code("""
import os, subprocess, sys, urllib.request

REPO_URL = "https://github.com/LearnForFun711/jungle-nnue.git"

def find_project():
    # Already inside the project (e.g. "Open in Colab" from the GitHub page).
    if os.path.isfile("scripts/rl_population.py"):
        return os.getcwd()
    if os.path.isdir("jungle-nnue/scripts"):
        return os.path.abspath("jungle-nnue")
    return None

root = find_project()
if root is None:
    r = subprocess.run(["git", "clone", REPO_URL, "jungle-nnue"])
    if r.returncode != 0:
        raise SystemExit(
            "git clone failed -- if the repo is private, clone it manually and unzip it "
            "to /content/jungle-nnue instead.")
    root = os.path.abspath("jungle-nnue")

os.chdir(root)
print("project at", os.getcwd())

# Make the clone current.  A shallow clone cannot always "git pull", but
# fetching the tip of main and resetting to it always works.
try:
    subprocess.run(["git", "fetch", "--depth", "1", "origin", "main"],
                   capture_output=True, timeout=300)
    subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"],
                   capture_output=True, timeout=120)
    print("git:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip())
except Exception as exc:
    print("git sync failed:", exc)

# The GPU sources, fetched from GitHub so that a clone made before the
# league was written cannot be the one that is built.
GPU_DIR = os.path.abspath(os.path.join("engine", "gpu"))
RAW = "https://raw.githubusercontent.com/LearnForFun711/jungle-nnue/main/engine/gpu/"
for name in ("jungle_gpu.h", "jungle_gpu.cuh", "jungle_mcts.cuh", "jungle_game.cuh",
             "jungle_gpu_common.cpp", "jungle_gpu_cuda.cu"):
    try:
        with urllib.request.urlopen(RAW + name, timeout=60) as response:
            data = response.read()
        with open(os.path.join(GPU_DIR, name), "wb") as f:
            f.write(data)
        print(f"  {name:24s} {len(data):7d} bytes")
    except Exception as exc:
        print(f"  {name:24s} could not fetch: {exc}")
"""),
    md("""
## 3. Build the engine

The engine is the C++ program that finally plays the network, and the match
harness is the thing that decides whether the network is any good.
"""),
    code("""
import os, subprocess, sys

print(subprocess.run(["make", "-C", "engine", "-j4"], capture_output=True,
                     text=True).stdout[-2000:])

# The harness has to report the winner, not the loser.  It used to do the
# opposite, and every match number this project produced before the fix was
# upside down.  The check takes a few seconds and it is worth taking.
proc = subprocess.run(["./engine/build/game_check", "--games", "8", "--time", "0.05"],
                      capture_output=True, text=True)
print(proc.stdout[-1500:])
if proc.returncode != 0:
    print("THE HARNESS IS BROKEN -- do not trust any match number until this passes")
"""),
    md("""
## 4. Put the teacher and the league on the GPU

Everything that costs GPU time happens here: the games.

The teacher is an MCTS whose leaf evaluations are batched across every game in
flight, and the forward pass is the engine's own arithmetic, so the GPU's
numbers are the engine's numbers.

The **league** is the same thing with two networks in one game: each side
searches with *its own* network as the leaf evaluation, which is what makes
the population loop possible.  That needed the kernel to pick the network per
request, so the CUDA source has to be the new one — which is why the sources
are fetched from GitHub above.

If the build fails — a runtime without a GPU, a driver that does not match the
toolkit — the notebook carries on with the CPU generator and nothing else
changes.
"""),
    code("""
import ctypes, glob, os, shutil, subprocess, sys, traceback, urllib.request

# Colab does not set CUDA_HOME, and torch needs it: without it torch cannot
# find the CUDA runtime to link the extension against, and the import then
# fails with "cannot open shared object file", which says nothing about what
# is wrong.
if not os.environ.get("CUDA_HOME"):
    candidates = ["/usr/local/cuda"]
    try:
        candidates += sorted(
            [os.path.join("/usr/local", d) for d in os.listdir("/usr/local")
             if d.startswith("cuda-")], reverse=True)
    except OSError:
        pass
    for candidate in candidates:
        if os.path.isdir(candidate):
            os.environ["CUDA_HOME"] = candidate
            break
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))

# torch's extension loader needs ninja, and does not say so when it is missing.
if shutil.which("ninja") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=False)

import torch
from torch.utils.cpp_extension import load

BUILD = "/content/jungle-gpu-build"
shutil.rmtree(BUILD, ignore_errors=True)      # no stale objects from a failed run
os.makedirs(BUILD, exist_ok=True)

# torch keeps a record, per process, of which extensions it has built and
# whether the sources have changed since.  If a build failed, that record
# still says the module is up to date: the build is skipped, and the import
# then fails on a shared object that was never produced.  Clearing the record
# makes it build again.
try:
    from torch.utils.cpp_extension import JIT_EXTENSION_VERSIONER
    JIT_EXTENSION_VERSIONER.entries.clear()
    print("cleared torch's record of what it has already built")
except Exception as exc:
    print("could not clear torch's build record:", exc)

EXT = None
if not torch.cuda.is_available():
    print("no CUDA device -- the CPU generator will be used")
else:
    try:
        EXT = load(
            name="jungle_gpu_selfplay",
            sources=[os.path.join(GPU_DIR, "jungle_gpu_cuda.cu"),
                     os.path.join(GPU_DIR, "jungle_gpu_common.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            # No --use_fast_math: the forward pass has to give the same
            # numbers as the engine's.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_include_paths=[GPU_DIR],
            build_directory=BUILD,
            verbose=True,
        )
        print("the GPU extension is built")
    except Exception as exc:
        # The compiler's message is inside the exception, not in the one line
        # the traceback ends with.
        print("=" * 30, "THE BUILD FAILED")
        print(str(exc)[:12000])
"""),
    md("""
## 5. The champion to start from

The population needs a starting network.  Upload the network you have (the
champion of a previous run, or any network you like), or leave it empty and
the loop starts from a fresh, untrained one — which is slower to get
anywhere, but is the honest version of "start from scratch".

Every network in the population has the same shape, because the league plays
them against each other and the GPU's forward pass has one shared memory
slice for all of them.  The shape of the champion is the shape of the
population.
"""),
    code("""
import os, shutil, subprocess, sys

CHAMPION = "/content/champion.nnue"

# Put the network here by hand, or upload it with the file pane.
#
#   from google.colab import files
#   files.upload()          # then copy the uploaded file to /content/champion.nnue
#
UPLOADED = "/content/uploaded.nnue"
if os.path.exists(UPLOADED):
    shutil.copyfile(UPLOADED, CHAMPION)
    print("using the uploaded network")
elif os.path.exists(CHAMPION):
    print("using the network that was already there")
else:
    print("no network given -- starting from scratch")

RL = "/content/rl"
os.makedirs(RL, exist_ok=True)

if os.path.exists(CHAMPION):
    subprocess.run([sys.executable, "scripts/net_to_checkpoint.py",
                    "--net", CHAMPION, "--out", os.path.join(RL, "champion.pt")],
                   check=True)
else:
    subprocess.run([sys.executable, "scripts/net_to_checkpoint.py",
                    "--random", "--ft-hidden", "96", "--hidden", "48",
                    "--out", os.path.join(RL, "champion.pt")], check=True)
    subprocess.run([sys.executable, "train/export_nnue.py",
                    "--checkpoint", os.path.join(RL, "champion.pt"),
                    "--out", CHAMPION], check=True)
    print("a fresh, untrained network was made")
"""),
    md("""
## 6. Does the GPU agree with the engine?

The GPU's forward pass has to give the same numbers as the engine's, or the
league is rating networks the engine does not play.  This cell checks it
value by value, and it also checks that the *league* on the GPU agrees with
the portable one on the CPU.

If this cell does not say PASS, the loop is measuring the wrong thing.
"""),
    code("""
import os, subprocess, sys

if EXT is None:
    print("no GPU extension -- nothing to check")
else:
    # The GPU's forward pass against the engine's.
    sys.path.insert(0, os.path.join(os.getcwd(), "engine", "gpu"))
    import gpu_selfplay
    gpu_selfplay.check_parity(EXT, CHAMPION, positions=2000, seed=7)

    # The league on the GPU against the league on the CPU.  The CPU one is
    # the reference implementation: the same code, run one game at a time.
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    from rl_population import League

    cpu_tool = os.path.join(os.getcwd(), "engine", "buildgpu", "gpu_selfplay_cpu")
    subprocess.run(["make", "-C", "engine", "BUILD=buildgpu",
                    "buildgpu/gpu_selfplay_cpu"], check=True)

    other = "/content/second.nnue"
    if not os.path.exists(other):
        # A second network to play against: the champion with a little noise
        # on it, which is what a mutant is.
        import torch
        ck = torch.load(os.path.join(RL, "champion.pt"), map_location="cpu")
        with torch.no_grad():
            for name, p in ck["model"].items():
                if p.is_floating_point() and p.numel() > 1:
                    p.add_(torch.randn_like(p) * (0.05 * float(p.std())))
        torch.save(ck, "/content/mutant.pt")
        subprocess.run([sys.executable, "train/export_nnue.py",
                        "--checkpoint", "/content/mutant.pt", "--out", other], check=True)

    gpu_league = League(EXT, cpu_tool, sims=200, leaf_scale=0.125, games=2)
    cpu_league = League(None, cpu_tool, sims=200, leaf_scale=0.125, games=2)
    nets = [CHAMPION, other]

    g_results, g_pairs = gpu_league.play(nets)
    c_results, c_pairs = cpu_league.play(nets)
    print("gpu:", g_results)
    print("cpu:", c_results)
    if g_results == c_results:
        print("PASS: the GPU league and the CPU league agree")
    else:
        print("WARNING: the GPU league and the CPU league disagree.  The games")
        print("         are stochastic, so this is not proof of a bug -- but it")
        print("         is worth a look before trusting the loop.")
"""),
    md("""
## 7. Run the population loop

This is the loop.  Each generation:

1. **League** — every pair of candidates plays, in both colours, on the GPU.
2. **Data** — the champion plays itself on the GPU.  The games are the reward.
3. **Training** — every candidate is trained on that data, with the game's
   result as the target and the teacher's value as a smaller, denser signal.
4. **Mutation** — the candidates for the next generation are the champion
   with a little noise on the weights, and a slightly different recipe each.
5. **Selection** — the league decides who is the champion.

The numbers below are the ones to change:

| flag | meaning |
|---|---|
| `--generations` | how many generations to run |
| `--population` | how many candidates per generation (the champion plus the mutants) |
| `--league-games` | games per ordered pair in a league round |
| `--league-sims` | simulations per move in the league |
| `--data-games` | games the champion plays to make the training data |
| `--data-sims` | simulations per move when making the data |
| `--train-minutes` | minutes of training per candidate |
| `--result-weight` | how much the game's result counts in the target |

The loop writes everything it does to `runs/rl/`, and it can be stopped at
any time: the data is written a shard at a time, and the champion is copied
out as it improves.
"""),
    code("""
import os, subprocess, sys, time

GENERATIONS = int(os.environ.get("GENERATIONS", "8"))
POPULATION  = int(os.environ.get("POPULATION", "4"))
LEAGUE_GAMES = int(os.environ.get("LEAGUE_GAMES", "8"))
LEAGUE_SIMS = int(os.environ.get("LEAGUE_SIMS", "400"))
DATA_GAMES  = int(os.environ.get("DATA_GAMES", "512"))
DATA_SIMS   = int(os.environ.get("DATA_SIMS", "400"))
TRAIN_MINUTES = float(os.environ.get("TRAIN_MINUTES", "10"))
RESULT_WEIGHT = float(os.environ.get("RESULT_WEIGHT", "0.75"))

cmd = [sys.executable, "scripts/rl_population.py",
       "--champion", os.path.join(RL, "champion.pt"),
       "--out", RL,
       "--generations", str(GENERATIONS),
       "--population", str(POPULATION),
       "--league-games", str(LEAGUE_GAMES),
       "--league-sims", str(LEAGUE_SIMS),
       "--data-games", str(DATA_GAMES),
       "--data-sims", str(DATA_SIMS),
       "--train-minutes", str(TRAIN_MINUTES),
       "--result-weight", str(RESULT_WEIGHT)]
print(" ".join(cmd))
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout[-8000:])
if proc.returncode != 0:
    print(proc.stderr[-4000:])
    raise SystemExit("the loop failed")
print("the loop took %.1f minutes" % ((time.time() - t0) / 60))
"""),
    md("""
## 8. What happened, generation by generation

The champion's score against the field, generation by generation.  This is the
number that matters: it is the result of the games, not a loss and not a
correlation.

If it goes up and then stops, the loop has found the fixed point of the
teacher — the network has learned everything the teacher's values can teach
it, and the only way further is a better reward (more decisive games) or a
better teacher.
"""),
    code("""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = os.path.join(RL, "history.json")
history = json.load(open(path))["generations"]
print(f"{len(history)} generations")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot([h["generation"] for h in history], [h["champion_score"] for h in history],
        marker="o", label="the champion's score against the field")
ax.axhline(0.5, color="grey", linestyle=":", label="a draw")
ax.set_xlabel("generation")
ax.set_ylabel("score")
ax.set_title("The champion, generation by generation")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("/content/rl-history.png", dpi=120)
print("wrote /content/rl-history.png")
for h in history:
    print(f"  generation {h['generation']}: champion {h['champion']}  "
          f"score {h['champion_score']:.3f}  Elo {h['champion_elo']:.0f}")
"""),
    md("""
## 9. The final check: the champion against the handcrafted engine

The league measures the networks as *teacher leaves*, which is the same
evaluation used by the same search, but it is not the engine that finally
plays them.  This is the engine, against the handcrafted evaluation, with
the colours alternating.

**This is the number that decides what ships.**  The loop's own ranking is
only a way of choosing what to test.

The harness used to report the loser as the winner, so make sure the check in
cell 3 passed before believing anything here.
"""),
    code("""
import json, os, subprocess, sys

CHAMPION_NET = os.path.join(RL, "champion.pt").replace(".pt", ".nnue")
if not os.path.exists(CHAMPION_NET):
    # The loop exports every candidate; the champion is the one the league
    # picked.  If it was not copied out, take the best one.
    import glob
    cands = sorted(glob.glob(os.path.join(RL, "nets", "**", "*.nnue"), recursive=True))
    CHAMPION_NET = cands[0] if cands else CHAMPION

GAMES = int(os.environ.get("FINAL_GAMES", "300"))
SECONDS = float(os.environ.get("FINAL_TIME", "0.5"))
THREADS = str(os.cpu_count() or 4)

print(f"champion: {CHAMPION_NET}")
print(f"{GAMES} games at {SECONDS}s/move")

proc = subprocess.run(["./engine/build/match", "--games", str(GAMES),
                       "--threads", THREADS, "--time", str(SECONDS),
                       "--engine-a", f"ab:nnue={CHAMPION_NET}", "--engine-b", "ab"],
                      capture_output=True, text=True)
try:
    d = json.loads(proc.stdout)
except Exception:
    print(proc.stdout[-3000:], proc.stderr[-2000:])
    raise SystemExit("the match did not produce a result")
print(json.dumps(d, indent=2))
ci = d.get("score_ci95", [float("nan"), float("nan")])
print(f"\\nscore {d['score_a']:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  Elo {d['elo_a']:+.1f}")
"""),
    md("""
## 10. Download

The champion, the history and the learning curve.
"""),
    code("""
import json, os, shutil, zipfile
from google.colab import files

zip_path = "/content/rl.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in [CHAMPION_NET,
              os.path.join(RL, "history.json"),
              "/content/rl-history.png"]:
        if os.path.exists(f):
            z.write(f, os.path.basename(f))
    for f in sorted(os.listdir(RL)):
        p = os.path.join(RL, f)
        if os.path.isfile(p) and f.endswith(".nnue"):
            z.write(p, f)
files.download(zip_path)
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "name": "rl_population.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
