# Cell 5a -- put the teacher on the GPU.
#
# Copy this cell into Colab and run it in place of the notebook's own 5a.
# It is self-contained: it does not need the repository to have been
# updated, so it works with a clone that predates the fix.
#
# What went wrong before: Colab does not set CUDA_HOME, and torch needs it.
# Without it torch cannot find the CUDA runtime to link the extension
# against, the shared object is built without the runtime's path in it, and
# the import fails with "cannot open shared object file" -- which says
# nothing about what is actually wrong.  The cell below sets CUDA_HOME,
# makes sure ninja is there, builds the extension, and if the import still
# fails it loads the runtime by its full path and tries again.
#
# There was a second reason, found once the first was out of the way: nvcc
# has a rule the plain C++ compiler does not, and a function called from
# device code has to be a device function itself.  The move generator
# called piece_type() and feature_index(), and neither was marked, so the
# .cu file did not compile at all.  That is fixed in the repository, and
# the pull at the top of this cell brings it in.
#
# If it still does not work, the last thing the cell prints is the build log.
# Paste that back and it will say what is wrong.

import ctypes
import glob
import os
import shutil
import subprocess
import sys
import traceback

# --- 0. get the fix ----------------------------------------------------------
# The functions the kernels call have to be marked for the device, or nvcc
# refuses to compile the file.  If this clone is older than that fix, pull
# it; if the pull does not work, carry on and the build log will say why.
try:
    subprocess.run(["git", "pull", "--ff-only"], capture_output=True, timeout=120)
    print("git pull:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip())
except Exception:
    pass

# --- 1. find the CUDA toolkit ------------------------------------------------
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

# --- 2. ninja ---------------------------------------------------------------
if shutil.which("ninja") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=False)

# --- 3. build ---------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load

GPU_DIR = os.path.abspath(os.path.join("engine", "gpu"))
BUILD = "/content/jungle-gpu-build"
os.makedirs(BUILD, exist_ok=True)

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
    except Exception:
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

        so = os.path.join(BUILD, "jungle_gpu_selfplay.so")
        if os.path.exists(so):
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("jungle_gpu_selfplay", so)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                EXT = module
                print("the extension was loaded directly")
            except Exception:
                traceback.print_exc()

    if EXT is None:
        # The one-line error hides the reason; the build log has it.
        for path in (glob.glob(os.path.join(BUILD, "*.log"))
                     + [os.path.join(BUILD, ".ninja_log")]):
            if os.path.exists(path):
                print("=" * 30, path)
                with open(path, errors="replace") as f:
                    print(f.read()[-4000:])

print("GPU teacher:", "ready" if EXT is not None else "not available")
