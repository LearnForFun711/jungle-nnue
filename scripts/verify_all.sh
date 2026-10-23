#!/usr/bin/env bash
# Run every check the CI runs, locally.
#
#   scripts/verify_all.sh
#
# It builds the engine, checks the rules against the golden data, and -- if
# numpy and torch are available -- trains a throwaway network from a tiny
# self-play dataset and proves that the PyTorch and C++ inference agree.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
WORK="${TMPDIR:-/tmp}/jungle-nnue-verify"

echo "==> building the engine"
make -C engine -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

echo
echo "==> golden rule test"
python3 scripts/verify_golden.py

echo
echo "==> random-position rule cross-check against the Python reference"
python3 scripts/verify_rules.py --games 20 --samples 400

if ! python3 -c "import numpy, torch" >/dev/null 2>&1; then
    echo
    echo "==> numpy/torch not available -- skipping the training checks"
    echo "    (pip install numpy torch to run them)"
    exit 0
fi

echo
echo "==> end-to-end: selfplay -> train -> export -> parity -> incremental -> match"
rm -rf "$WORK"
mkdir -p "$WORK"

./engine/build/selfplay --games 6 --threads 2 --sims 60 \
                        --leaf-depth 1 --leaf-agg 1 \
                        --out "$WORK/data"
python3 train/train.py --data "$WORK/data.t*.bin" \
                       --out "$WORK/run" --epochs 1 --batch 512 \
                       --lr 1e-3 --ft-hidden 64 --hidden 32
python3 train/export_nnue.py --checkpoint "$WORK/run/best.pt" \
                             --out "$WORK/net.nnue"
python3 scripts/verify_nnue.py --checkpoint "$WORK/run/best.pt" \
                               --net "$WORK/net.nnue" --positions 60
./engine/build/nnue_check "$WORK/net.nnue" 5 60
./engine/build/match --games 4 --threads 2 --time 0.02 \
                      --engine-a "ab:nnue=$WORK/net.nnue" --engine-b ab

echo
echo "==> all checks passed"
