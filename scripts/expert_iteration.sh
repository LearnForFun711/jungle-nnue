#!/usr/bin/env bash
# One round of expert iteration.
#
#   bash scripts/expert_iteration.sh <net.nnue> <minutes> [outdir]
#
# The network is plugged into the MCTS teacher as its leaf evaluation, so
# the teacher's values get better than the handcrafted evaluation's, the
# data generated from them is better, and the next network trained on it is
# better than the one that produced it.  That is the loop; it only needs the
# teacher to accept a network as its leaf, which it already does.
#
# The ground truth that anchors the loop is the *result* of the game, not the
# teacher's value: the teacher's value is a noisy estimate of the value of a
# position, and if the teacher's leaf is the network then the teacher's value
# starts chasing its own tail.  The result of the game does not.
#
#   1. train a network on the current data                (Colab, GPU)
#   2. generate new data with that network as the leaf    (this script)
#   3. train the next network on old + new data           (Colab, GPU)
#   4. repeat
set -euo pipefail

cd "$(dirname "$0")/.."

NET="${1:?usage: expert_iteration.sh <net.nnue> <minutes> [outdir]}"
MINUTES="${2:?usage: expert_iteration.sh <net.nnue> <minutes> [outdir]}"
OUT="${3:-data/round/data}"
THREADS="${THREADS:-$(nproc 2>/dev/null || echo 4)}"
SIMS="${SIMS:-800}"
LEAF_SCALE="${LEAF_SCALE:-1.0}"

if [ ! -f "$NET" ]; then
    echo "no such network: $NET" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "==> expert iteration round: net=$NET  budget=${MINUTES}min  threads=$THREADS"

DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    ./engine/build/selfplay --games 200 --threads "$THREADS" --sims "$SIMS" \
                            --nnue "$NET" --leaf-scale "$LEAF_SCALE" \
                            --leaf-depth 1 --leaf-agg 1 --c-puct 1.4 \
                            --out "$OUT"
done

echo "==> round complete.  Shards in $(dirname "$OUT"):"
ls -la "$(dirname "$OUT")" | head -20
