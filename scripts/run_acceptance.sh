#!/usr/bin/env bash
# The acceptance test: the neural engine against the handcrafted engine.
#
#   bash scripts/run_acceptance.sh <net.nnue> [games] [seconds]
#
# Both engines play the same search; the only difference is the evaluation.
# The harness alternates the colours from game to game, so every opening is
# played from both sides, and reports the score with a 95% confidence
# interval, the corresponding Elo difference, and -- separately -- the score
# over the decisive games, because most games end in a draw and the decisive
# games are where the difference shows.
set -euo pipefail

cd "$(dirname "$0")/.."

NET="${1:?usage: run_acceptance.sh <net.nnue> [games] [seconds]}"
GAMES="${2:-400}"
SECONDS_PER_MOVE="${3:-1.0}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ ! -f "$NET" ]; then
    echo "no such network: $NET" >&2
    exit 1
fi

echo "==> neural engine : $NET"
echo "==> games $GAMES at ${SECONDS_PER_MOVE}s/move, $THREADS thread(s)"

OUT="runs/acceptance-$(date +%Y%m%d-%H%M%S).json"
mkdir -p runs

./engine/build/match --games "$GAMES" \
                      --threads "$THREADS" \
                      --time "$SECONDS_PER_MOVE" \
                      --engine-a "ab:nnue=$NET" \
                      --engine-b ab \
                      | tee "$OUT"

echo
echo "==> written to $OUT"
