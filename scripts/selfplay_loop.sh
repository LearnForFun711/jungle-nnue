#!/usr/bin/env bash
# Generate self-play data continuously for a while.
#
#   bash scripts/selfplay_loop.sh [minutes] [outdir] [threads]
#
# The generator writes one shard per thread and appends to it, so the loop
# can be stopped at any time and whatever is on disk is already usable.
#
# This is what makes the GPU budget worth spending: the trainer picks the
# data up as it appears, so the Colab VM's CPUs can generate data for the
# MCTS teacher while the GPU is busy with the network.
set -euo pipefail

cd "$(dirname "$0")/.."
MINUTES="${1:-480}"
OUT="${2:-data/more/data}"
THREADS="${3:-$(nproc 2>/dev/null || echo 2)}"
SIMS="${SIMS:-800}"
LEAF="${LEAF:-1}"
AGG="${AGG:-1}"

mkdir -p "$(dirname "$OUT")"
echo "==> self-play for ${MINUTES} min, ${THREADS} thread(s), ${SIMS} sims/move"

DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # Run the generator in rounds; each round is a batch of games, so the
    # shards stay small enough to be flushed to disk regularly.
    ./engine/build/selfplay --games 200 --threads "$THREADS" --sims "$SIMS" \
                            --leaf-depth "$LEAF" --leaf-agg "$AGG" --c-puct 1.4 \
                            --out "$OUT"
done

echo "==> done.  Shards:"
ls -la "$(dirname "$OUT")" | head -20
