#!/usr/bin/env bash
# Train a small sweep of network shapes and export each one.
#
#   bash scripts/sweep.sh [outdir]
#
# Each configuration takes a couple of minutes on an A100, so a whole sweep
# costs a fraction of the GPU budget.  The point is *not* to pick the model
# with the lowest validation loss -- it is to pick the one that plays
# strongest, because those two are not the same thing: a wider network
# evaluates more accurately but also more slowly, and at a fixed clock the
# slower engine searches less deeply.
#
# The exported networks land next to each other in the output directory; run
# them against each other with engine/build/match to pick a winner.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-runs/sweep}"
DATA="${DATA:-data/data.t*.bin}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-8192}"
LR="${LR:-2e-3}"
RESULTW="${RESULTW:-0.25}"

mkdir -p "$OUT"

# name                 ft   hid
CONFIGS=(
  "h128-64              128   64"
  "h256-128             256  128"
  "h384-192             384  192"
  "h512-256             512  256"
)

for cfg in "${CONFIGS[@]}"; do
    read -r name ft hid <<<"$cfg"
    dir="$OUT/$name"
    if [ -f "$dir/net.nnue" ]; then
        echo "==> $name already done, skipping"
        continue
    fi
    echo "==> $name (ft_hidden=$ft hidden=$hid)"
    python3 train/train.py --data "$DATA" --out "$dir" \
        --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
        --ft-hidden "$ft" --hidden "$hid" --result-weight "$RESULTW"
    python3 train/export_nnue.py --checkpoint "$dir/best.pt" --out "$dir/net.nnue"
done

echo
echo "==> done.  Compare them with:"
echo "    for d in $OUT/*/; do echo \"\$d\"; done"
