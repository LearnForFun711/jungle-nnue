#!/usr/bin/env python3
"""Generate the Colab notebook (colab/train_colab.ipynb).

The notebook is the whole "train on the A100" story in one file: it fetches
the project, uploads and unpacks the dataset, trains the network, exports it
to the engine's binary format, verifies that the PyTorch and C++ forward
passes agree and hands back a download.

Keeping it generated from a script (rather than hand-written JSON) means the
notebook can be regenerated when the training options change.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "colab", "train_colab.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# Jungle Chess NNUE — train on Colab (A100)

This notebook trains the neural network on the MCTS self-play dataset and
exports it to the binary format the C++ engine reads.

**Before you run anything**

1. Runtime ▸ Change runtime type ▸ **A100 GPU**, then restart.
2. Run the cells below in order.  Cell 3 asks you for the dataset file.

The dataset is the only thing that crosses the network, and it is uploaded
once.  Training then happens entirely on the GPU.
"""),
    md("## 1. Environment"),
    code("""
# Colab already ships a CUDA build of PyTorch that matches the runtime's
# driver.  Only install one if the runtime has no GPU at all -- installing a
# different CUDA wheel can break the runtime.
import subprocess, sys

if subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
    print("no GPU visible -- installing a CUDA build of torch")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device -- did you pick an A100 runtime?")
print("device:", torch.cuda.get_device_name(0))
print("memory: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
"""),
    md("## 2. Get the project"),
    code("""
import os, subprocess, sys

REPO_URL = "https://github.com/LearnForFun711/jungle-nnue.git"

def find_project():
    # Already inside the project (e.g. "Open in Colab" from the GitHub page).
    if os.path.isfile("train/train.py"):
        return os.getcwd()
    if os.path.isdir("jungle-nnue/train"):
        return os.path.abspath("jungle-nnue")
    return None

root = find_project()
if root is None:
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "jungle-nnue"])
    if r.returncode != 0:
        raise SystemExit(
            "git clone failed -- if the repo is private, clone it manually and unzip it "
            "to /content/jungle-nnue instead.")
    root = os.path.abspath("jungle-nnue")

os.chdir(root)
print("project at", os.getcwd())
print("contents:", sorted(os.listdir("."))[:12])
"""),
    code("""
# Build the C++ engine.  It is used for the parity check at the end and for
# generating extra self-play data on the VM.
!make -C engine NATIVE=1 -j4
"""),
    md("""
## 3. Upload the dataset

Click the folder icon in the Colab file pane (or use the button in the cell
below) and choose **`jungle-dataset.tar.gz`** from your computer.

It is the only file you need to move.
"""),
    code("""
import os, glob, tarfile

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

shards = sorted(glob.glob(os.path.join(DATA_DIR, "data.t*.bin")))
archives = sorted(glob.glob(os.path.join(DATA_DIR, "*.tar.gz"))) + \\
           sorted(glob.glob(os.path.join(DATA_DIR, "*.tgz"))) + \\
           sorted(glob.glob("/content/*.tar.gz")) + \\
           sorted(glob.glob("/content/*.tgz")) + \\
           sorted(glob.glob("*.tar.gz")) + \\
           sorted(glob.glob("*.tgz"))

if not shards and archives:
    for a in archives:
        print("unpacking", a)
        with tarfile.open(a) as tar:
            try:
                tar.extractall(DATA_DIR, filter="data")
            except TypeError:
                tar.extractall(DATA_DIR)
    shards = sorted(glob.glob(os.path.join(DATA_DIR, "data.t*.bin")))

print("shards found:", len(shards))
total = sum(os.path.getsize(s) for s in shards)
print("total: %.1f MB  (~%.0fk positions)" % (total / 1e6, total / 68 / 1000))
for s in shards[:5]:
    print("  ", s, "%.1f MB" % (os.path.getsize(s) / 1e6))
if not shards:
    raise SystemExit(
        "No dataset found.\\n"
        "Upload jungle-dataset.tar.gz with the file pane (folder icon) into the "
        "'data' folder, or into the runtime root, then run this cell again.")
"""),
    md("""
### Or upload it right here

If you prefer not to use the file pane, the cell below opens a file chooser.
"""),
    code("""
# Uncomment and run this if you would rather upload with a file chooser than
# use the file pane.  Pick jungle-dataset.tar.gz.
#
# from google.colab import files
# uploaded = files.upload()
# for name in uploaded:
#     with tarfile.open(name) as tar:
#         tar.extractall(DATA_DIR)
# shards = sorted(glob.glob(os.path.join(DATA_DIR, "data.t*.bin")))
# print("shards:", len(shards))
"""),
    md("""
## 4. Start the data generator in the background

The GPU is not the bottleneck -- generating the teacher's data is.  Colab gives
a couple of CPU cores, so this is much slower than a laptop, but it runs
*while* the GPU trains, and the trainer picks the new shards up as they
appear.

This cell starts it in the background and returns immediately.  Stop it with
the `pkill` line in the last cell.
"""),
    code("""
import os, subprocess, sys, time

SELFPLAY_MINUTES = int(os.environ.get("SELFPLAY_MINUTES", "480"))
os.makedirs("data/more", exist_ok=True)

# --nnue is deliberately not used here: at this point there is no trained
# network yet, so the teacher still uses the handcrafted leaf.  Once the
# first network exists, the cell at the end of this notebook restarts the
# generator with the network as the teacher's leaf -- that is the expert
# iteration loop, and it is what makes the teacher's data improve from one
# round to the next.
proc = subprocess.Popen(
    ["bash", "scripts/selfplay_loop.sh", str(SELFPLAY_MINUTES),
     "data/more/data", str(os.cpu_count() or 2)],
    stdout=open("selfplay.log", "w"), stderr=subprocess.STDOUT)
print("self-play started in the background (pid %d), log: selfplay.log" % proc.pid)
time.sleep(5)
print(open("selfplay.log").read()[-800:])
"""),
    md("""
## 5. Train a quick first network, and put it in the teacher

The teacher's values are what the network learns from, and they are only as
good as the evaluation the teacher searches with.  With the handcrafted
evaluation the teacher's value correlates with the result of the game by
about 0.14 -- barely better than nothing -- so the data it produces is
mostly a noisy version of "who won".  That is the real bottleneck of this
project, and no amount of GPU time fixes it.

A network is a much better leaf evaluation than the handcrafted one, so the
fix is to train a small network first, plug it into the teacher, and let the
generator spend the rest of the session producing data from it.  That is the
expert-iteration loop, and starting it half an hour into the session instead
of after it is worth a lot of data quality.

This cell trains a small network for `PRETRAIN_MINUTES` minutes (a couple of
dozen on an A100 is plenty -- it only has to be good enough to be a better
leaf), exports it, and restarts the generator with it as the teacher's leaf.
"""),
    code("""
import os, subprocess, sys, time

PRETRAIN_MINUTES = int(os.environ.get("PRETRAIN_MINUTES", "20"))
FIRST_NET = "/content/net-first.nnue"
os.makedirs("data/round1", exist_ok=True)

print("training the first network for %d minutes ..." % PRETRAIN_MINUTES)
!{sys.executable} train/train.py --data "data/**/*.bin" \\
    --out runs/first --budget-minutes {PRETRAIN_MINUTES} \\
    --batch 8192 --lr 2e-3 --ft-hidden 96 --hidden 48 --result-weight 0.25
!{sys.executable} train/export_nnue.py --checkpoint runs/first/best.pt \\
    --out {FIRST_NET}
print("first network written to", FIRST_NET)

# Restart the generator with the network as the teacher's leaf.
subprocess.run(["pkill", "-f", "selfplay_loop.sh"])
time.sleep(2)
os.makedirs("data/round1", exist_ok=True)
proc = subprocess.Popen(
    ["bash", "scripts/expert_iteration.sh", FIRST_NET, str(SELFPLAY_MINUTES),
     "data/round1/data"],
    stdout=open("round1.log", "w"), stderr=subprocess.STDOUT)
print("generator restarted with the network as the teacher's leaf (pid %d),"
      " log: round1.log" % proc.pid)
time.sleep(5)
print(open("round1.log").read()[-800:])
"""),
    md("""
## 6. Train -- eight hours of it

The whole training budget is one number.  `train/train_budget.py` splits it
across several network shapes; each run gets a slice, trains on whatever data
is on disk at that moment (so the shards the background generator is writing
get picked up), and exports its network.

| flag | meaning |
|---|---|
| `--minutes` | the total wall-clock budget |
| `--configs` | comma separated `ft_hidden:hidden:result_weight` |
| `--batch` | positions per step (8192 is a good default on an A100) |
| `--lr` | peak learning rate |

The learning-rate schedule is re-aimed at the budget after every epoch, so
the rate decays across the whole eight hours instead of stopping part-way
through a schedule sized for fewer epochs.

Note that the network with the lowest validation loss is not automatically
the strongest one: a wider network evaluates more accurately but also more
slowly, and at a fixed clock the slower engine searches less deeply.  Every
candidate is exported so they can be compared by playing strength.
"""),
    code("""
import os, sys, time

BUDGET_MINUTES = int(os.environ.get("BUDGET_MINUTES", "480"))   # eight hours
CONFIGS        = os.environ.get("CONFIGS", "96:48:0.25,96:48:0.50,96:48:0.75,128:64:0.50,256:128:0.50")
BATCH          = int(os.environ.get("BATCH", "8192"))
LR             = float(os.environ.get("LR", "2e-3"))

print(f"budget={BUDGET_MINUTES} min  configs={CONFIGS}  batch={BATCH}  lr={LR}")
t0 = time.time()
!{sys.executable} train/train_budget.py --data "data/**/*.bin" \\
    --out runs/budget --minutes {BUDGET_MINUTES} --configs "{CONFIGS}" \\
    --batch {BATCH} --lr {LR} --best-out /content/net.nnue
print("training took %.1f minutes" % ((time.time() - t0) / 60))
"""),
    md("""
## 7. The learning curve

Two curves per configuration, and the second one is the one that matters.

The **validation loss** is how well the network predicts the training
target.  The **correlation with the game outcome** is how well it predicts
*who won the game* -- the thing a value function is actually for.  A
network can have a falling validation loss and a falling correlation at the
same time, and only one of them is playing strength.

The curves are for the held-out positions, so they are not the numbers the
network was fitted to.  The horizontal axis is the epoch and the label says
how long the run had been going, because the runs are cut by the clock
rather than by the epoch count.
"""),
    code("""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = sorted(glob.glob("runs/budget/*/history.json"))
print("runs with a curve:", len(runs))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for path in runs:
    name = os.path.basename(os.path.dirname(path))
    try:
        h = json.load(open(path))
    except Exception as exc:
        print("  ", name, "unreadable:", exc)
        continue
    curve = h.get("curve") or []
    if not curve:
        print("  ", name, "has no curve")
        continue
    epochs = [c["epoch"] for c in curve]
    loss = [c["val_loss"] for c in curve]
    corr = [c.get("corr", 0.0) for c in curve]
    minutes = curve[-1].get("minutes", 0.0)
    label = f"{name}  ({minutes:.0f} min)"
    axes[0].plot(epochs, loss, marker=".", label=label)
    axes[1].plot(epochs, corr, marker=".", label=label)
    best = max(corr) if corr else 0.0
    print(f"  {name}: best val_loss {min(loss):.6f}  best corr(outcome) {best:+.4f}")

axes[0].set_xlabel("epoch")
axes[0].set_ylabel("validation loss")
axes[0].set_title("Validation loss (lower is better)")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("correlation with the game outcome")
axes[1].set_title("Playing strength (higher is better)")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/content/learning-curve.png", dpi=120)
print("wrote /content/learning-curve.png")
try:
    from IPython.display import Image, display
    display(Image(filename="/content/learning-curve.png"))
except Exception:
    pass
"""),
    md("""
## 8. Pick the strongest candidate -- by playing strength

A lower validation loss does not mean a stronger engine.  A wider network
evaluates more accurately but also more slowly, and at a fixed clock the
slower engine searches less deeply, so the network with the best loss can
easily be the weaker player.

So the candidates play.  Each one plays the handcrafted engine, which is
the baseline the whole project has to beat, and the scores are printed.
The winner is copied to `/content/net-strongest.nnue` and replaces
`/content/net.nnue`.

The clock is short on purpose: a long match would take longer than the
training did.  What is being measured here is which candidate to take
forward, not the final acceptance number.
"""),
    code("""
import glob, json, os, shutil, subprocess

MATCH_GAMES = int(os.environ.get("MATCH_GAMES", "30"))
MATCH_TIME = float(os.environ.get("MATCH_TIME", "0.25"))
THREADS = str(os.cpu_count() or 2)

results = []
for net in sorted(glob.glob("/content/candidates/*.nnue")):
    name = os.path.basename(net)
    print(f"\\n=== {name} vs the handcrafted engine "
          f"({MATCH_GAMES} games at {MATCH_TIME}s/move) ===", flush=True)
    proc = subprocess.run(
        ["./engine/build/match", "--games", str(MATCH_GAMES),
         "--threads", THREADS, "--time", str(MATCH_TIME),
         "--engine-a", f"ab:nnue={net}", "--engine-b", "ab"],
        capture_output=True, text=True)
    try:
        d = json.loads(proc.stdout)
    except Exception:
        print(proc.stdout[-500:], proc.stderr[-500:])
        continue
    score = d["score_a"]
    print(f"    {d['a_wins']} W / {d['b_wins']} L / {d['draws']} D   "
          f"score {score:.3f}   Elo {d['elo_a']:+.1f}   "
          f"decisive {d.get('decisive_score_a', float('nan')):.3f}")
    results.append((score, name, net))

if results:
    results.sort(reverse=True)
    print("\\n=== the candidates, strongest first ===")
    for score, name, net in results:
        print(f"    {score:.3f}  {name}")
    best = results[0]
    shutil.copyfile(best[2], "/content/net-strongest.nnue")
    shutil.copyfile(best[2], "/content/net.nnue")
    print(f"\\nstrongest: {best[1]}  ->  /content/net-strongest.nnue")
    with open("/content/strength.json", "w") as f:
        json.dump([{"name": n, "score": sc} for sc, n, _ in results], f, indent=2)
else:
    print("no candidates were measured")
"""),
    md("""
## 9. Export

The exported file holds the int8 weights, the per-layer scales and biases,
and the architecture metadata.  The C++ engine reads it directly.
"""),
    code("""
import glob, os, shutil

# Every candidate, plus the best one by validation loss.
os.makedirs("/content/candidates", exist_ok=True)
for net in sorted(glob.glob("runs/budget/*/net.nnue")):
    shutil.copyfile(net, os.path.join("/content/candidates", os.path.basename(os.path.dirname(net)) + ".nnue"))
    print("candidate:", net)

for net in sorted(glob.glob("runs/budget/*/net.nnue")):
    ck = net.replace("/net.nnue", "/best.pt")
    if os.path.exists(ck):
        out = "/content/candidates/" + os.path.basename(os.path.dirname(net)) + ".nnue"
        subprocess_ok = os.system(f"python3 train/export_nnue.py --checkpoint {ck} --out {out}") == 0
        print("re-exported", out, "ok" if subprocess_ok else "FAILED")

print()
print("best net:", "/content/net.nnue",
      "%.2f MB" % (os.path.getsize("/content/net.nnue") / 1e6) if os.path.exists("/content/net.nnue") else "MISSING")
"""),
    md("""
## 10. Verify

This proves that the PyTorch forward pass and the C++ inference produce
identical values, so the engine plays exactly the network that was trained.
It should print `PASS`.
"""),
    code("""
import glob, os
for ck in sorted(glob.glob("runs/budget/*/best.pt")):
    print("==", ck)
    os.system(f"python3 scripts/verify_nnue.py --checkpoint {ck} --net build/verify.nnue --positions 200")
    os.system(f"./engine/build/nnue_check /content/net.nnue 5 60")
"""),
    md("""
## 11. Download the network

`net.nnue` is about 300 KB.  Put it in the project's `net.nnue` and you can
run the match locally:

```bash
cd engine
./build/match --games 400 --time 1.0 --engine-a "ab:nnue=../net.nnue" --engine-b ab
```
"""),
    code("""
import shutil, zipfile, os, glob
from google.colab import files

# The best network ...
files.download("/content/net.nnue")

# The learning curve ...
if os.path.exists("/content/learning-curve.png"):
    files.download("/content/learning-curve.png")

# ... every candidate, in one archive, so the strongest one can be chosen by
# playing strength rather than by validation loss.
zip_path = "/content/candidates.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("/content/candidates/*.nnue"):
        z.write(f, os.path.basename(f))
    for f in ("/content/net-strongest.nnue", "/content/learning-curve.png",
              "/content/strength.json"):
        if os.path.exists(f):
            z.write(f, os.path.basename(f))
files.download(zip_path)
"""),
    md("""
## 12. (Optional) Take the data with you, and start the next round

The shards the background generator produced are worth keeping: they are
teacher data for the *next* iteration.  The cell below packs them and stops
the generator.
"""),
    code("""
import glob, os, subprocess, tarfile
subprocess.run(["pkill", "-f", "selfplay_loop.sh"])

shards = sorted(glob.glob("data/more/data.t*.bin"))
print("new shards:", len(shards),
      "%.1f MB" % (sum(os.path.getsize(s) for s in shards) / 1e6))
if shards:
    with tarfile.open("/content/jungle-dataset-round2.tar.gz", "w:gz") as tar:
        for s in shards:
            tar.add(s, arcname=os.path.basename(s))
    print("packed /content/jungle-dataset-round2.tar.gz")
    from google.colab import files
    files.download("/content/jungle-dataset-round2.tar.gz")
"""),
    md("""
## 13. (Optional) Start the expert-iteration round

This is the self-play reinforcement loop.  The network you have just trained
becomes the MCTS teacher's leaf evaluation, so the teacher's values are now
the network's values plus a search on top of them, the data generated from
them is better than the data the network was trained on, and the next
network trained on that data is better than this one.

The ground truth that keeps the loop honest is the *result of the game*, not
the teacher's value: the teacher's value is a noisy estimate of a position's
value, and if the teacher's leaf is the network then the teacher's value
starts chasing its own tail.  The result of the game does not.  That is why
the training target blends the two (`--result-weight`).

Run this cell to generate the next round's data.  It runs in the background
for `ROUND2_MINUTES` minutes; the shards appear under `data/round2/`.
"""),
    code("""
import os, subprocess, time

ROUND2_MINUTES = int(os.environ.get("ROUND2_MINUTES", "480"))
NET = "/content/net.nnue"

if not os.path.exists(NET):
    raise SystemExit("no trained network at %s -- run the training cell first" % NET)

os.makedirs("data/round2", exist_ok=True)
proc = subprocess.Popen(
    ["bash", "scripts/expert_iteration.sh", NET, str(ROUND2_MINUTES),
     "data/round2/data"],
    stdout=open("round2.log", "w"), stderr=subprocess.STDOUT)
print("expert iteration started (pid %d), log: round2.log" % proc.pid)
time.sleep(5)
print(open("round2.log").read()[-800:])
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "name": "train_colab.ipynb"},
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
