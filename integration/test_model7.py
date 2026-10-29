#!/usr/bin/env python3
"""Check that the neural engine can be dropped into the web front end.

The test does not modify the front end: it imports the model from
``integration/model7.py``, points it at the library and the network in this
repository, and plays a few moves against the Python engine's own rules.

    python3 integration/test_model7.py [path-to-the-original-project]

The original project is the one that holds ``backend/engine.py`` -- the
reference implementation of the rules.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_original_project():
    """Look for the project that holds backend/engine.py."""
    candidates = [
        os.environ.get("JUNGLE_ORIGINAL_PROJECT"),
        os.path.join(os.path.dirname(ROOT), "backend"),
        os.path.join(os.path.dirname(os.path.dirname(ROOT)), "backend"),
        "/Users/taimanchan/Desktop/鬥獸棋/backend",
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "engine.py")):
            return os.path.dirname(c)
    return None


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    original = (sys.argv[1] if len(sys.argv) > 1 else find_original_project())
    if not original:
        print("could not find the original project (the one with backend/engine.py)")
        return 2
    print("original project:", original)
    sys.path.insert(0, os.path.join(original, "backend"))

    from engine import JungleEngine

    model7 = load_module(os.path.join(HERE, "model7.py"), "model7")
    ai = model7.Model7AI()
    if not ai.lib or not ai.net_path:
        print("FAIL: the neural engine did not load")
        return 1

    game = JungleEngine()
    print("initial position:")
    for row in game.board:
        print("   ", " ".join(f"{p:>2}" for p in row))

    for ply in range(6):
        move = ai.get_move(game, max_time=0.5)
        if move is None:
            print(f"ply {ply}: no move")
            break
        ok = game.apply_move(move)
        print(f"ply {ply}: {move} accepted={ok}  turn={game.turn}")
        if not ok:
            print("FAIL: the engine produced a move the rules reject")
            return 1
        if game.winner:
            print("winner:", game.winner)
            break

    # The move must also be one the rules consider legal.
    legal = {(tuple(m['from']), tuple(m['to'])) for m in game.get_valid_moves()}
    print(f"legal moves available: {len(legal)}")
    print("PASS: the neural engine plays legal moves in the front end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
