#!/usr/bin/env bash
# Generate self-play data on the GPU, continuously.
#
#   bash scripts/gpu_selfplay_loop.sh <net.nnue> <minutes> [outdir] [games]
#
# The teacher runs entirely on the GPU: the legal moves come out of a
# bitboard move generator, the search is an MCTS whose leaf evaluations are
# batched across every game in flight, and the network's forward pass is the
# engine's own arithmetic, so the data is the same kind of data the CPU
# generator writes.
#
# The number of games in flight is not written down here.  It is measured:
# engine/gpu/gpu_selfplay.py works out how many games the card can keep busy
# and how much video memory they need, and the number it recommends is
# written to runs/gpu/autotune.json.  This script reads it from there.
#
# The generator writes one shard per round, so the loop can be stopped at
# any time and whatever is on disk is already usable -- and the trainer
# picks the shards up as they appear.
set -euo pipefail

cd "$(dirname "$0")/.."

NET="${1:?usage: gpu_selfplay_loop.sh <net.nnue> <minutes> [outdir] [games]}"
MINUTES="${2:?usage: gpu_selfplay_loop.sh <net.nnue> <minutes> [outdir] [games]}"
OUT="${3:-data/gpu/data}"
SIMS="${SIMS:-800}"
LEAF_SCALE="${LEAF_SCALE:-0.125}"

if [ ! -f "$NET" ]; then
    echo "no such network: $NET" >&2
    exit 1
fi

# How many games in flight.  The auto-tuner measured it; if the tuner has
# not been run, fall back on a number that fits comfortably.
GAMES="${4:-$(python3 - <<'PY'
import json, os
try:
    plan = json.load(open("runs/gpu/autotune.json"))
    print(int(plan["chosen_games"]))
except Exception:
    print(1024)
PY
)}"

mkdir -p "$(dirname "$OUT")"
echo "==> GPU self-play for ${MINUTES} min, ${GAMES} games in flight, ${SIMS} sims/move"

DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
ROUND=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    python3 engine/gpu/gpu_selfplay.py --net "$NET" --out "${OUT}.t${ROUND}.bin" \
        --games "$GAMES" --sims "$SIMS" --seed "$((ROUND + 1))" --leaf-scale "$LEAF_SCALE" \
        2>&1 | tail -3
    ROUND=$((ROUND + 1))
done

echo "==> done.  Shards in $(dirname "$OUT"):"
ls -la "$(dirname "$OUT")" | head -20
