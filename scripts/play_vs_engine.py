#!/usr/bin/env python3
"""Play a complete game against the neural engine through the web API.

    python3 scripts/play_vs_engine.py [--url http://127.0.0.1:5001] [--time 1.0]

This is the end-to-end check: it drives the Flask application exactly the way
the browser does -- set the configuration, make the human move, ask the
engine for its reply -- and reports the result.

The human side plays the first legal move it finds, which is not good play,
but it exercises the whole path: the repetition rule, the move encoding, the
engine's reply and the win conditions.  The legal moves come from the front
end's own rules implementation, which is imported from the original project.

The server has the last word on every move: its state is copied back after
each exchange, because a move can be rejected by the rules (a repetition)
and the local copy would then be wrong.
"""
import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5001"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(f"{path} -> HTTP {exc.code}: {body}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode())


def load_rules(backend_dir):
    """Import the front end's own rules engine."""
    sys.path.insert(0, backend_dir)
    spec = importlib.util.spec_from_file_location("jungle_rules",
                                                  os.path.join(backend_dir, "engine.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jungle_rules"] = mod
    spec.loader.exec_module(mod)
    return mod.JungleEngine


def sync(local, state):
    """Copy the server's position into the local copy."""
    local.board = [list(row) for row in state["board"]]
    local.turn = state["turn"]
    local.winner = state.get("winner")


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=BASE)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--backend", default="")
    args = ap.parse_args()

    BASE = args.url

    backend = args.backend
    if not backend:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cand in ("/Users/taimanchan/Desktop/鬥獸棋/backend",
                     os.path.join(os.path.dirname(here), "backend")):
            if os.path.isfile(os.path.join(cand, "engine.py")):
                backend = cand
                break
    if not backend:
        print("could not find the front end's backend/engine.py")
        return 2
    JungleEngine = load_rules(backend)
    print("rules from:", backend)

    post("/api/reset", {})
    post("/api/set_config", {"red": "human", "black": "model7",
                              "red_time": args.time, "black_time": args.time})

    local = JungleEngine()
    state = get("/api/state")
    sync(local, state)

    ply = 0
    for ply in range(args.max_plies):
        if state.get("winner"):
            break
        legal = local.get_valid_moves()
        if not legal:
            print("no legal moves for the human side")
            break
        move = legal[0]
        state = post("/api/move", move)
        sync(local, state)
        if state.get("winner"):
            break
        state = post("/api/ai_move", {})
        sync(local, state)
        engine_move = state.get("last_move")
        print(f"ply {ply:3d}  human {move['from']}->{move['to']}   "
              f"engine {engine_move}", flush=True)

    print(f"\nresult: winner={state.get('winner')}  plies={ply + 1}  "
          f"turn={state.get('turn')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
