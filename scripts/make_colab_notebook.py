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
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
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
## 4. Train

`--data` is a glob.  The trainer keeps the whole dataset on the GPU when it
fits, which makes the feature construction essentially free.

| flag | meaning |
|---|---|
| `--epochs` | passes over the data (40 is cheap: the whole run takes under a minute on an A100) |
| `--batch` | positions per step (8192 is a good default on an A100) |
| `--lr` | peak learning rate |
| `--ft-hidden` / `--hidden` | network width |
| `--result-weight` | how much of the target comes from the game result vs. the MCTS value |
| `--max-minutes` | wall-clock budget; training stops cleanly when it is hit |

The network is only ~300k parameters, so an A100 runs through a million
positions in well under a second.  Training is not the bottleneck — generating
the data was.
"""),
    code("""
import os, sys, time

PATTERN = "data/data.t*.bin"
EPOCHS   = int(os.environ.get("EPOCHS", "40"))
BATCH    = int(os.environ.get("BATCH", "8192"))
LR       = float(os.environ.get("LR", "2e-3"))
FT       = int(os.environ.get("FT_HIDDEN", "256"))
HIDDEN   = int(os.environ.get("HIDDEN", "128"))
MAXMIN   = float(os.environ.get("MAX_MINUTES", "0"))
RESULTW  = float(os.environ.get("RESULT_WEIGHT", "0.25"))

print(f"epochs={EPOCHS} batch={BATCH} lr={LR} ft={FT} hidden={HIDDEN} result_weight={RESULTW}")
t0 = time.time()
!{sys.executable} train/train.py --data "{PATTERN}" --out runs/colab \\
    --epochs {EPOCHS} --batch {BATCH} --lr {LR} --ft-hidden {FT} --hidden {HIDDEN} \\
    --result-weight {RESULTW} --max-minutes {MAXMIN}
print("training took %.1f minutes" % ((time.time() - t0) / 60))
"""),
    md("""
## 5. Export

The exported file holds the int8 weights, the per-layer scales and the
architecture metadata.  The C++ engine reads it directly.
"""),
    code("""
!python3 train/export_nnue.py --checkpoint runs/colab/best.pt --out /content/net.nnue
import os
print("net.nnue: %.2f MB" % (os.path.getsize("/content/net.nnue") / 1e6))
"""),
    md("""
## 6. Verify

This proves that the PyTorch forward pass and the C++ inference produce
identical values, so the engine plays exactly the network that was trained.
It should print `PASS`.
"""),
    code("""
!python3 scripts/verify_nnue.py --checkpoint runs/colab/best.pt --net build/verify.nnue --positions 300
"""),
    md("""
## 7. Download the network

`net.nnue` is about 300 KB.  Put it in the project's `net.nnue` and you can
run the match locally:

```bash
cd engine
./build/match --games 400 --time 1.0 --engine-a "ab:nnue=../net.nnue" --engine-b ab
```
"""),
    code("""
from google.colab import files
files.download("/content/net.nnue")
"""),
    md("""
## 8. (Optional) More self-play data on the VM

Colab only gives a couple of CPU cores, so this is much slower than a laptop
— but it runs without any further upload, and it can be combined with the
dataset you already have.

```bash
cd jungle-nnue/engine
./build/selfplay --games 400 --threads $(nproc) --sims 800 \\
                 --leaf-depth 1 --leaf-agg 1 --c-puct 1.4 \\
                 --out /content/more/data
```

Download `/content/more/*.bin` and add them to the next run's dataset.
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
