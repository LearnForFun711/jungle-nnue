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
        "No dataset found.\n"
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
## 5. Train -- eight hours of it

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
CONFIGS        = os.environ.get("CONFIGS", "256:128:0.25,384:192:0.25")
BATCH          = int(os.environ.get("BATCH", "8192"))
LR             = float(os.environ.get("LR", "2e-3"))

print(f"budget={BUDGET_MINUTES} min  configs={CONFIGS}  batch={BATCH}  lr={LR}")
t0 = time.time()
!{sys.executable} train/train_budget.py --data "data/data.t*.bin" \\
    --out runs/budget --minutes {BUDGET_MINUTES} --configs "{CONFIGS}" \\
    --batch {BATCH} --lr {LR} --best-out /content/net.nnue
print("training took %.1f minutes" % ((time.time() - t0) / 60))
"""),
    md("""
## 6. Export

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
## 7. Verify

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
## 8. Download the network

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

# ... every candidate, in one archive, so the strongest one can be chosen by
# playing strength rather than by validation loss.
zip_path = "/content/candidates.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("/content/candidates/*.nnue"):
        z.write(f, os.path.basename(f))
files.download(zip_path)
"""),
    md("""
## 9. (Optional) Take the data with you, and start the next round

The shards the background generator produced are worth keeping: they are
teacher data for the *next* iteration.  The cell below packs them and stops
the generator.

To start the expert-iteration round -- the network becomes the teacher's leaf
evaluation, so the teacher's data improves, so the next network is better --
run this after the download:

```bash
cd engine
./build/selfplay --games 2000 --threads $(nproc) --sims 800 \\
                 --nnue ../net.nnue --leaf-depth 1 --leaf-agg 1 \\
                 --out /content/round2/data
```
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
