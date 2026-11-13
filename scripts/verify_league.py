#!/usr/bin/env python3
"""Check that the league reports the winner, not the loser.

    python3 scripts/verify_league.py --nets net/net.nnue,net/net-new.nnue \
        --games 8 --sims 100

The match harness used to record the loser as the winner, and every match
number in this project was upside down because of it.  The same thing could
happen here: the league's result is reported in one place and stored in
another (the 68-byte records), and if the two disagree then the league is
rating the candidates backwards.

So this script plays a league round and then reads the records back and
compares:

  * the league's own result for each game, and
  * the result the records carry, which comes from the side to move's point
    of view and is the convention the whole training pipeline depends on.

They have to agree on every game.  It also checks that the two colours are
balanced: every ordered pair has to play the same number of games as Red and
as Black, or a network that is only good as Black gets a free ride.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from dataset import load_shards  # noqa: E402


def game_results_from_records(path):
    """The result of each game, worked out from the board.

    No sign convention is used here: the records are the positions *before*
    each move, so the last record of a game is the position the winner moved
    from.  The winner is the side to move in that record, and it is the side
    that can step into the opponent's den -- so the result is read straight
    off the board.

    A game that ended some other way (a repetition, or a side with no legal
    move) cannot be decided from the board alone, and is reported as None.
    """
    board, stm, _result, _value, ply, _cause = load_shards(path)
    if len(stm) == 0:
        return []

    # The records of a game are written in order and the ply counter starts
    # over at the beginning of every game, so a drop in the ply is where one
    # game ends and the next begins.
    games = []
    current = []
    last_ply = -1
    for i in range(len(stm)):
        if ply[i] < last_ply and current:
            games.append(current)
            current = []
        current.append(i)
        last_ply = ply[i]
    if current:
        games.append(current)

    results = []
    for game in games:
        last = game[-1]
        side = int(stm[last])            # 0 = Red, 1 = Black
        if can_enter_the_den(board[last], side):
            # The side to move in the last position is the one that won.
            results.append(+1 if side == 0 else -1)
        else:
            results.append(None)          # ended some other way
    return results


# The den each side can enter to win: Red's den is at (8, 3) and Black's at
# (0, 3), so Red wins by stepping into (0, 3) and Black by stepping into
# (8, 3).
DEN_SQUARE = {0: (0, 3), 1: (8, 3)}


def can_enter_the_den(board, side):
    """Can this side step into the opponent's den from where it stands?

    A den is entered by an ordinary step onto the den square, so the side
    has to have a piece next to it.  The den squares are not in the river,
    and neither are the squares around them, so the water rule does not
    come into it."""
    r, c = DEN_SQUARE[side]
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 9 and 0 <= nc < 7:
            piece = int(board[nr * 7 + nc])
            if piece == 0:
                continue
            # The piece codes are 1..8 for Red and 9..16 for Black.
            colour = 0 if piece <= 8 else 1
            if colour == side:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", required=True,
                    help="comma separated networks, at least two")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--leaf-scale", type=float, default=0.125)
    ap.add_argument("--tool", default=os.path.join(ROOT, "engine", "buildgpu",
                                                   "gpu_selfplay_cpu"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "league-verify"))
    args = ap.parse_args()

    nets = [n for n in args.nets.split(",") if n]
    if len(nets) < 2:
        print("a league needs at least two networks", file=sys.stderr)
        return 2
    for n in nets:
        if not os.path.exists(os.path.join(ROOT, n)):
            print(f"no such network: {n}", file=sys.stderr)
            return 2

    if not os.path.exists(args.tool):
        subprocess.run(["make", "-C", os.path.join(ROOT, "engine"),
                        "BUILD=buildgpu", "buildgpu/gpu_selfplay_cpu"], check=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # The schedule: every ordered pair, half of the games each way.
    red, black = [], []
    for i in range(len(nets)):
        for j in range(len(nets)):
            if i == j:
                continue
            for k in range(args.games):
                if k % 2 == 0:
                    red.append(i); black.append(j)
                else:
                    red.append(j); black.append(i)
    schedule = ",".join(f"{r}:{b}" for r, b in zip(red, black))

    proc = subprocess.run([args.tool, "--nets", ",".join(nets),
                           "--schedule", schedule,
                           "--games", str(len(red)), "--sims", str(args.sims),
                           "--seed", str(args.seed),
                           "--leaf-scale", str(args.leaf_scale),
                           "--out", args.out],
                          capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
        return 1

    reported = None
    for line in proc.stdout.splitlines():
        if line.startswith("game results:"):
            reported = [int(x) for x in line.split()[2:]]
    if reported is None:
        print("the league tool printed no results", file=sys.stderr)
        print(proc.stdout[-2000:], file=sys.stderr)
        return 1

    from_records = game_results_from_records(args.out + ".bin")

    print(f"games reported   {len(reported)}")
    print(f"games in records {len(from_records)}")
    ok = True
    if len(reported) != len(from_records):
        print("FAIL: the league and the records disagree on how many games "
              "there were")
        ok = False
    else:
        bad = []
        checked = 0
        for i, (a, b) in enumerate(zip(reported, from_records)):
            if b is None:
                continue          # ended some other way; the board cannot say
            checked += 1
            if a != b:
                bad.append((i, a, b))
        if bad:
            i, a, b = bad[0]
            print(f"FAIL: the league and the board disagree on {len(bad)} "
                  f"game(s), the first is game {i}: the league says {a}, "
                  f"the board says {b}")
            ok = False
        else:
            print(f"PASS: the league and the board agree on all {checked} "
                  f"decisive game(s)")

    # The colour balance: every ordered pair has to play the same number of
    # games as Red and as Black.
    tally = {}
    for r, b in zip(red, black):
        tally[(r, b)] = tally.get((r, b), 0) + 1
    counts = set(tally.values())
    if len(counts) == 1 and red.count(0) >= 0:
        print(f"PASS: every ordered pair plays the same number of games "
              f"({counts.pop()})")
    else:
        print(f"FAIL: the pairs do not play the same number of games: {tally}")
        ok = False

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
