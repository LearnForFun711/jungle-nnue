# Cell 5a -- put the teacher on the GPU.
#
# Copy this cell into Colab and run it in place of the notebook's own 5a.
#
# Three things have gone wrong here, and all three are handled below.
#
# 1. The clone was stale.  The notebook clones the project, and a clone
#    made before a fix is a clone that cannot build.  A shallow clone
#    cannot always be pulled, so the cell fetches the tip of main and
#    resets to it, which always works.
#
# 2. nvcc has a rule the plain C++ compiler does not: a function called
#    from device code has to be a device function itself.  The move
#    generator called piece_type() and feature_index(), and neither was
#    marked, so nvcc rejected the file.  This is the "GPU function calling
#    a CPU function" error, and it is a compile error, not a runtime one.
#
# 3. torch keeps a record, per process, of which extensions it has built
#    and whether the sources have changed since.  If a build failed, that
#    record still says the module is up to date: the next build is
#    skipped, and the import then fails on a shared object that was never
#    produced -- with an error that mentions nothing about the build.  A
#    failed build poisons every later attempt until the record is cleared.
#
# The cell also fetches the six GPU sources straight from GitHub, so that
# a clone made before the fix cannot be the one that is built, and prints
# the compiler's own message if the build fails -- that message is inside
# the exception, not in the one line the traceback ends with.

import ctypes
import glob
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request

GPU_DIR = os.path.abspath(os.path.join("engine", "gpu"))
RAW = "https://raw.githubusercontent.com/LearnForFun711/jungle-nnue/main/engine/gpu/"
SOURCES = ("jungle_gpu.h", "jungle_gpu.cuh", "jungle_mcts.cuh", "jungle_game.cuh",
           "jungle_gpu_common.cpp", "jungle_gpu_cuda.cu")

# --- 1. make the clone current ------------------------------------------------
try:
    subprocess.run(["git", "fetch", "--depth", "1", "origin", "main"],
                   capture_output=True, timeout=300)
    subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"],
                   capture_output=True, timeout=120)
    print("git:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip())
except Exception as exc:
    print("git sync failed:", exc)

# --- 2. get the fixed sources -------------------------------------------------
print("--- fetching the GPU sources ---")
for name in SOURCES:
    try:
        with urllib.request.urlopen(RAW + name, timeout=60) as response:
            data = response.read()
        with open(os.path.join(GPU_DIR, name), "wb") as f:
            f.write(data)
        print(f"  {name:24s} {len(data):7d} bytes")
    except Exception as exc:
        print(f"  {name:24s} COULD NOT FETCH: {exc}")

# --- 3. the CUDA toolkit and ninja ---------------------------------------------
if not os.environ.get("CUDA_HOME"):
    candidates = ["/usr/local/cuda"]
    try:
        candidates += sorted(
            [os.path.join("/usr/local", d) for d in os.listdir("/usr/local")
             if d.startswith("cuda-")], reverse=True)
    except OSError:
        pass
    for candidate in candidates:
        if os.path.isdir(candidate):
            os.environ["CUDA_HOME"] = candidate
            break
print("CUDA_HOME =", os.environ.get("CUDA_HOME"))

if shutil.which("ninja") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=False)

# --- 4. build ------------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load

BUILD = "/content/jungle-gpu-build"
shutil.rmtree(BUILD, ignore_errors=True)      # no stale objects from a failed run
os.makedirs(BUILD, exist_ok=True)

# Clear torch's record of what it thinks it has already built, so that a
# failed build does not make every later attempt skip the build.
try:
    from torch.utils.cpp_extension import JIT_EXTENSION_VERSIONER
    JIT_EXTENSION_VERSIONER.entries.clear()
    print("cleared torch's record of what it has already built")
except Exception as exc:
    print("could not clear torch's build record:", exc)

EXT = None
if not torch.cuda.is_available():
    print("no CUDA device -- the CPU teacher will be used")
else:
    try:
        EXT = load(
            name="jungle_gpu_selfplay",
            sources=[os.path.join(GPU_DIR, "jungle_gpu_cuda.cu"),
                     os.path.join(GPU_DIR, "jungle_gpu_common.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            # No --use_fast_math: the forward pass has to give the same
            # numbers as the engine's.
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_include_paths=[GPU_DIR],
            build_directory=BUILD,
            verbose=True,
        )
    except Exception as exc:
        # The compiler's message is inside the exception, not in the one
        # line the traceback ends with.  Print all of it.
        print("=" * 30, "THE BUILD FAILED")
        print(str(exc)[:12000])
        cause = exc.__cause__
        while cause is not None:
            print("=" * 30, "CAUSE")
            print(str(cause)[:12000])
            cause = cause.__cause__
        traceback.print_exc()

    if EXT is None:
        # The shared object may have been built but would not load, because
        # the CUDA runtime is not on the loader's search path.  Loading the
        # runtime by its full path first puts it into the process for good.
        cuda_lib = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
        for lib in ("libcudart.so.12", "libcudart.so.11.0", "libcudart.so"):
            path = os.path.join(cuda_lib, lib)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    print("preloaded", path)
                except OSError as exc:
                    print("could not preload", path, exc)

        for so in glob.glob(os.path.join(BUILD, "*.so")):
            import importlib.util
            module_name = os.path.basename(so).split(".")[0]
            try:
                spec = importlib.util.spec_from_file_location(module_name, so)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                EXT = module
                print("the extension was loaded directly from", so)
                break
            except Exception as exc:
                print("could not load", so, exc)

    if EXT is None:
        for path in (glob.glob(os.path.join(BUILD, "*.log"))
                     + [os.path.join(BUILD, ".ninja_log")]):
            if os.path.exists(path):
                print("=" * 30, path)
                with open(path, errors="replace") as f:
                    print(f.read()[-4000:])

print("GPU teacher:", "ready" if EXT is not None else "not available")
