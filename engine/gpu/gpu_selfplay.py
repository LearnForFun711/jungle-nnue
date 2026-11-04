#!/usr/bin/env python3
"""The GPU self-play generator, and the program that decides how to run it.

Two things live here.

`build_extension()` compiles the CUDA source with torch's own compiler, so
the notebook does not have to know how.  It returns None when the compile
fails -- on a machine without a GPU, or when the toolkit is missing -- and
the caller falls back on the CPU teacher.

`autotune()` is the part that answers the question "how many games should
be in flight?".  The answer is not a matter of taste:

  * the memory model says how many games fit in the video memory, and the
    answer is different for a 16 GB card and a 40 GB one;
  * the throughput curve says how many games the chip can actually keep
    busy, which is usually a good deal fewer than fit, because the games
    finish at different times and the network evaluations have to wait for
    the slowest lane in the warp.

So the program measures both and prints the number it would use, together
with the table it measured.  Nothing in the notebook hard-codes a batch
size.

    python3 engine/gpu/gpu_selfplay.py --net net/net.nnue --tune --budget-gb 40
    python3 engine/gpu/gpu_selfplay.py --net net/net.nnue --games 4096 --out data/gpu
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GPU_DIR = HERE
CUDA_SOURCE = os.path.join(GPU_DIR, "jungle_gpu_cuda.cu")
COMMON_SOURCE = os.path.join(GPU_DIR, "jungle_gpu_common.cpp")

# The teacher settings.  These are the ones that measured best: the neural
# leaf at a quarter of the engine's scale is more informative than the
# handcrafted evaluation, and it is not saturated.
DEFAULT_SIMS = 800
DEFAULT_LEAF_SCALE = 0.125
DEFAULT_VALUE_SCALE = 10.0
DEFAULT_C_PUCT = 1.4
DEFAULT_WIDENING = 1.6
DEFAULT_DIRICHLET_ALPHA = 0.30
DEFAULT_DIRICHLET_EPS = 0.25
DEFAULT_MAX_PLY = 200


def prepare_cuda_env():
    """Find the CUDA toolkit, and put ninja in place.

    Colab does not set `CUDA_HOME`, and torch needs it: without it torch
    cannot find the CUDA runtime to link the extension against, the shared
    object is built without the path to `libcudart` in it, and the import
    then fails with "cannot open shared object file" -- which says nothing
    about what is actually wrong.  Setting `CUDA_HOME` makes torch put the
    runtime's directory into the library search path of the extension, and
    the import works.
    """
    if not os.environ.get("CUDA_HOME"):
        candidates = ["/usr/local/cuda"]
        # Colab images sometimes have the toolkit under a versioned name.
        try:
            for name in sorted(os.listdir("/usr/local"), reverse=True):
                if name.startswith("cuda-"):
                    candidates.append(os.path.join("/usr/local", name))
        except OSError:
            pass
        for candidate in candidates:
            if os.path.isdir(candidate):
                os.environ["CUDA_HOME"] = candidate
                print(f"  CUDA_HOME set to {candidate}")
                break

    # torch's extension loader needs ninja, and does not say so when it is
    # missing.
    if shutil.which("ninja") is None:
        print("  installing ninja (torch needs it to build the extension)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"],
                       check=False)


def build_extension(verbose=True, build_dir=None):
    """Compile the CUDA extension.  Returns the module, or None."""
    try:
        import torch
        from torch.utils.cpp_extension import load
    except Exception as exc:
        print(f"  torch is not available: {exc}")
        return None

    if not os.path.exists(CUDA_SOURCE):
        print(f"  {CUDA_SOURCE} is missing")
        return None

    prepare_cuda_env()

    build_dir = build_dir or os.path.join(ROOT, "engine", "buildgpu", "torch")
    os.makedirs(build_dir, exist_ok=True)
    name = "jungle_gpu_selfplay"
    try:
        if verbose:
            print("  compiling the CUDA extension (this takes a minute)...")
        module = load(
            name=name,
            sources=[CUDA_SOURCE, COMMON_SOURCE],
            extra_cflags=["-O3", "-std=c++17"],
            # No --use_fast_math: the forward pass has to give the same
            # numbers as the engine's, and fast math would let the compiler
            # change the rounding of the divisions.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_include_paths=[GPU_DIR],
            build_directory=build_dir,
            verbose=verbose,
        )
        return module
    except Exception as exc:
        print(f"  the CUDA extension did not build: {exc}")
        # The one-line message above hides the reason, which is usually in
        # the build log.  Print it: a compile error and a missing runtime
        # look the same from here.
        log = os.path.join(build_dir, ".ninja_log")
        if os.path.exists(log):
            try:
                with open(log, errors="replace") as f:
                    tail = f.read()[-3000:]
                if tail.strip():
                    print("---- build log (tail) ----")
                    print(tail)
            except OSError:
                pass
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def device_memory_bytes():
    """How much video memory the card has."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory)
    except Exception:
        pass
    return 0


def memory_model(module, threads=256):
    """The memory the generator needs, from the compiled code itself."""
    model = dict(module.memory_model(threads))
    model["bytes_per_game_total"] = int(model["bytes_per_game_total"])
    model["bytes_per_game"] = int(model["bytes_per_game"])
    model["queue_bytes_per_warp"] = int(model["queue_bytes_per_warp"])
    return model


def games_that_fit(model, budget_bytes, margin=0.8):
    """How many games fit in the budget, and how much room is left over.

    The margin is for the parts of the process the model does not know
    about: the CUDA context, the library's own buffers, and the room the
    allocator keeps so that it does not have to ask the driver for more
    memory in the middle of a run.
    """
    per_game = model["bytes_per_game_total"]
    usable = int(budget_bytes * margin)
    return usable // per_game, usable


def run_once(module, net, out_path, games, sims=DEFAULT_SIMS, seed=12345,
             leaf_scale=DEFAULT_LEAF_SCALE, threads=0):
    """One round of generation.  Returns the statistics the kernel reports."""
    return dict(module.run(
        net_path=net,
        out_path=out_path,
        games=int(games),
        sims=int(sims),
        seed=int(seed),
        leaf_scale=float(leaf_scale),
        value_scale=DEFAULT_VALUE_SCALE,
        c_puct=DEFAULT_C_PUCT,
        widening=DEFAULT_WIDENING,
        dirichlet_alpha=DEFAULT_DIRICHLET_ALPHA,
        dirichlet_eps=DEFAULT_DIRICHLET_EPS,
        max_ply=DEFAULT_MAX_PLY,
        threads=int(threads),
    ))


def autotune(module, net, budget_gb=40.0, sims=DEFAULT_SIMS, out_dir=None,
             candidates=None, seed=12345, leaf_scale=DEFAULT_LEAF_SCALE,
             tune_sims=200, max_candidates=4096):
    """Measure the throughput curve and work out the best number of games.

    The curve is measured, not guessed: each candidate number of games is
    run for a moment and the positions per second are counted.  The number
    the program recommends is the one that reaches the best rate -- and,
    when several do, the smallest of them, because a smaller number leaves
    more room for the other things the card has to do.

    Two things are kept apart on purpose.

    The *memory model* says how many games fit.  On a 40 GB card that is
    hundreds of thousands, and measuring a run of that size would take
    hours, so the curve is measured at a few thousand games and the memory
    limit is reported next to it.

    The *curve* says how many games the chip can keep busy.  That is a good
    deal fewer than fit, because the games finish at different times and a
    batch has to wait for the slowest lane in the warp.  The sweep uses
    fewer simulations a move than the real generator, so that each point
    costs a few seconds rather than a few minutes.
    """
    out_dir = out_dir or os.path.join(ROOT, "runs", "gpu")
    os.makedirs(out_dir, exist_ok=True)

    budget = int(budget_gb * (1024 ** 3))
    total_memory = device_memory_bytes()
    if total_memory:
        budget = min(budget, total_memory)

    threads = 0
    model = memory_model(module, threads)
    fits, usable = games_that_fit(model, budget)

    print("== the memory model ==")
    print(f"  card memory        : {total_memory/2**30:.1f} GB" if total_memory
          else "  card memory        : unknown")
    print(f"  budget             : {budget/2**30:.1f} GB")
    print(f"  per game           : {model['bytes_per_game_total']/1024:.1f} KiB "
          f"(tree {model['bytes_per_game']/1024:.1f} KiB, "
          f"records {model['records_bytes_per_game']/1024:.1f} KiB, "
          f"queue {model['queue_bytes_per_warp']/1024:.1f} KiB per warp)")
    print(f"  games that fit     : {fits}")

    if candidates is None:
        # Powers of two up to the point where the curve has certainly
        # turned over.  Measuring beyond that would cost minutes per point
        # and tell us nothing the curve has not already said.
        candidates = []
        n = 128
        while n <= max_candidates and n <= fits:
            candidates.append(n)
            n *= 2
        if fits < 128:
            candidates = [max(32, fits)]
        candidates = sorted(set(candidates))

    print()
    print(f"== the measured curve ({tune_sims} simulations a move, so each point is quick) ==")
    print(f"  {'games':>8}  {'seconds':>8}  {'positions':>11}  {'positions/s':>12}  {'games/s':>9}")
    rows = []
    for games in candidates:
        if games > fits:
            continue
        path = os.path.join(out_dir, f"tune_{games}.bin")
        stats = run_once(module, net, path, games, sims=tune_sims, seed=seed,
                         leaf_scale=leaf_scale)
        positions = int(stats["positions"])
        seconds = float(stats["seconds"])
        rate = positions / seconds if seconds > 0 else 0.0
        print(f"  {games:8d}  {seconds:8.2f}  {positions:11d}  {rate:12.0f}  "
              f"{float(stats['games_per_second']):9.2f}")
        rows.append({"games": int(games), "seconds": seconds, "positions": positions,
                     "positions_per_second": rate,
                     "games_per_second": float(stats["games_per_second"])})
        try:
            os.remove(path)
        except OSError:
            pass

    if not rows:
        print("  nothing could be measured")
        return None

    best = max(rows, key=lambda r: r["positions_per_second"])
    # Among the configurations that are within one percent of the best, take
    # the smallest: it does the same work and leaves the card room to spare.
    near = [r for r in rows if r["positions_per_second"] >= 0.99 * best["positions_per_second"]]
    chosen = min(near, key=lambda r: r["games"])

    print()
    print("== the answer ==")
    print(f"  best measured rate  : {best['positions_per_second']:.0f} positions/s "
          f"at {best['games']} games")
    print(f"  recommended         : {chosen['games']} games in flight "
          f"({chosen['positions_per_second']:.0f} positions/s)")
    print(f"  video memory used   : "
          f"{chosen['games'] * model['bytes_per_game_total'] / 2**30:.2f} GB of "
          f"{budget/2**30:.1f} GB")
    print(f"  the memory would allow {fits} games; the chip runs out of work to do "
          f"long before that.")
    print(f"  the generator will use {sims} simulations a move "
          f"(the sweep above used {tune_sims})")
    result = {"chosen_games": chosen["games"], "sims": sims,
              "positions_per_second": chosen["positions_per_second"],
              "budget_bytes": budget, "fits": int(fits),
              "bytes_per_game_total": model["bytes_per_game_total"],
              "curve": rows}
    with open(os.path.join(out_dir, "autotune.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def build_reference_tool():
    """Compile the tool that gives the CPU's value for real positions."""
    exe = os.path.join(ROOT, "engine", "buildgpu", "nnue_ref")
    src = os.path.join(ROOT, "engine", "tools", "nnue_ref.cpp")
    if os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(src):
        return exe
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    cmd = ["c++", "-O3", "-std=c++17", "-o", exe, src,
           "-I", os.path.join(ROOT, "engine", "src")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        return None
    return exe


def check_parity(module, net, positions=20000, seed=7):
    """Check the GPU's forward pass against the CPU's.

    The CPU's forward pass is the engine's own, and the engine's has already
    been checked against the engine's int8 path.  If the GPU agrees with the
    CPU here, it agrees with the engine.  The positions come from random
    games played with the engine's rules, because a random board would not
    exercise the features the way a game does.
    """
    import numpy as np
    import torch

    exe = build_reference_tool()
    if exe is None:
        print("  the reference tool could not be built; skipping the check")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "positions.bin")
        proc = subprocess.run([exe, "--net", net, "--positions", str(positions),
                               "--seed", str(seed), "--out", path],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
            return None
        raw = np.fromfile(path, dtype=np.uint8)

    n = len(raw) // 68
    raw = raw[: n * 68].reshape(n, 68)
    boards = np.ascontiguousarray(raw[:, :63])
    stms = np.ascontiguousarray(raw[:, 63].astype(np.int32))
    cpu = raw[:, 64:68].copy().view(np.float32).reshape(n)

    gpu = module.nnue_batch(torch.from_numpy(boards), torch.from_numpy(stms), net).numpy()
    worst = float(np.max(np.abs(gpu - cpu))) if n else 0.0
    print(f"  GPU vs CPU forward pass: {n} positions, worst difference {worst:.3g}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default=os.path.join(ROOT, "net", "net.nnue"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "gpu", "data"))
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--leaf-scale", type=float, default=DEFAULT_LEAF_SCALE)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tune", action="store_true", help="measure the best number of games")
    ap.add_argument("--budget-gb", type=float, default=40.0)
    ap.add_argument("--candidates", type=int, nargs="*", default=None)
    ap.add_argument("--tune-sims", type=int, default=200,
                    help="simulations a move during the sweep (the real "
                         "generator uses --sims)")
    ap.add_argument("--parity", action="store_true", help="check the GPU against the CPU")
    args = ap.parse_args()

    module = build_extension()
    if module is None:
        print("no GPU extension; use the CPU teacher instead")
        return 1

    if args.parity:
        check_parity(module, args.net)

    if args.tune:
        autotune(module, args.net, budget_gb=args.budget_gb, sims=args.sims,
                 candidates=args.candidates, seed=args.seed,
                 leaf_scale=args.leaf_scale, tune_sims=args.tune_sims)

    if args.games > 0 and not args.tune:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        stats = run_once(module, args.net, args.out, args.games,
                         sims=args.sims, seed=args.seed, leaf_scale=args.leaf_scale,
                         threads=args.threads)
        print(json.dumps(stats, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
