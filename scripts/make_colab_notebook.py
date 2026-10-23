#!/usr/bin/env python3
"""Generate the Colab notebook (colab/train_colab.ipynb).

The notebook is the whole "train on the A100" story in one file: it fetches
the project, builds the engine, finds the dataset, trains the network, exports
it to the engine's binary format and hands back a download.

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

**What you need**

1. A Colab Pro+ runtime with an **A100 GPU** (Runtime ▸ Change runtime type).
2. The dataset. Either
   * mount Google Drive and put the shards somewhere, or
   * upload `data/data.t*.bin` to the runtime's file pane.

The dataset is the only thing that crosses the network, and it is uploaded
**once**.  Training happens entirely on the GPU.
"""),
    md("## 1. Environment"),
    code("""
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("memory: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
"""),
    md("## 2. Get the project"),
    code("""
import os, subprocess, sys

REPO_URL = "https://github.com/LearnForFun711/jungle-nnue.git"   # change if needed
if os.path.isdir("jungle-nnue/.git"):
    print("already present")
else:
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "jungle-nnue"])
    if r.returncode != 0:
        raise SystemExit(
            "git clone failed -- either the repo is private or the URL is wrong.\\n"
            "You can also upload the project as a zip and unzip it to /content/jungle-nnue.")

os.chdir("jungle-nnue")
print("project at", os.getcwd())
"""),
    code("""
# Build the C++ engine (used for extra self-play and for the final match).
!make -C engine NATIVE=1 -j4
"""),
    md("""
## 3. Locate the dataset

The trainer reads the shards produced by `engine/tools/selfplay.cpp`.
Every file matching `data/data.t*.bin` is read.
"""),
    code("""
import glob, os

def find_data():
    candidates = []
    # 1. uploaded into the runtime
    candidates += sorted(glob.glob("/content/data/data.t*.bin"))
    candidates += sorted(glob.glob("/content/*.bin"))
    # 2. Google Drive mounted
    if os.path.isdir("/content/drive/MyDrive"):
        candidates += sorted(glob.glob("/content/drive/MyDrive/**/*.t*.bin", recursive=True))
    return candidates

files = find_data()
print("found", len(files), "shards")
for f in files[:10]:
    print("  ", f, os.path.getsize(f) / 1e6, "MB")
"""),
    md("""
## 4. Train

`--data` is a glob.  The trainer keeps the whole dataset on the GPU when it
fits, which makes the feature construction essentially free.

Knobs worth knowing:

| flag | meaning |
|---|---|
| `--epochs` | passes over the data |
| `--batch` | positions per step (8192 is a good default on an A100) |
| `--lr` | peak learning rate |
| `--ft-hidden` / `--hidden` | network width |
| `--result-weight` | how much of the target comes from the game result vs. the MCTS value |
| `--max-minutes` | hard wall-clock budget; training stops cleanly when hit |
"""),
    code("""
import glob, os, subprocess, sys, time

shards = find_data()
if not shards:
    raise SystemExit("No dataset found -- upload data/data.t*.bin or mount Drive.")

pattern = shards[0].replace("data.t0.bin", "data.t*.bin")
if not any("data.t" in s for s in shards):
    pattern = shards[0]

EPOCHS   = int(os.environ.get("EPOCHS", "12"))
BATCH    = int(os.environ.get("BATCH", "8192"))
LR       = float(os.environ.get("LR", "2e-3"))
FT       = int(os.environ.get("FT_HIDDEN", "512"))
HIDDEN   = int(os.environ.get("HIDDEN", "256"))
MAXMIN   = float(os.environ.get("MAX_MINUTES", "0"))

cmd = [sys.executable, "train/train.py",
       "--data", pattern,
       "--out", "runs/colab",
       "--epochs", str(EPOCHS),
       "--batch", str(BATCH),
       "--lr", str(LR),
       "--ft-hidden", str(FT),
       "--hidden", str(HIDDEN),
       "--max-minutes", str(MAXMIN)]
print(" ".join(cmd))
t0 = time.time()
!{sys.executable} train/train.py --data "{pattern}" --out runs/colab --epochs {EPOCHS} --batch {BATCH} --lr {LR} --ft-hidden {FT} --hidden {HIDDEN} --max-minutes {MAXMIN}
print("training took %.1f minutes" % ((time.time()-t0)/60))
"""),
    md("""
## 5. Export

The exported file contains the int8 weights, the per-layer scales and the
architecture metadata.  The C++ engine reads it directly.
"""),
    code("""
!python3 train/export_nnue.py --checkpoint runs/colab/best.pt --out /content/net.nnue
import os
print("net.nnue: %.2f MB" % (os.path.getsize("/content/net.nnue")/1e6))
"""),
    code("""
# Download the network (about 1 MB).
from google.colab import files
files.download("/content/net.nnue")
"""),
    md("""
## 6. Verify (optional but recommended)

This proves that the PyTorch forward pass and the C++ inference produce
identical values, so the engine plays exactly the network that was trained.
"""),
    code("""
!python3 scripts/verify_nnue.py --checkpoint runs/colab/best.pt --net build/verify.nnue --positions 200
"""),
    md("""
## 7. (Optional) Generate more self-play data on the VM

Colab only gives a couple of CPU cores, so this is much slower than your
machine -- but it runs without any further upload.

```bash
cd jungle-nnue/engine
./build/selfplay --games 200 --threads $(nproc) --sims 800 --leaf-depth 1 \\
                 --out /content/more/data
```

Download `/content/more/*.bin` and add it to the next run's dataset.
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
