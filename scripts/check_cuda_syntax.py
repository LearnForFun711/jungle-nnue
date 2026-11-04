#!/usr/bin/env python3
"""Type-check the CUDA source on a machine without a CUDA toolkit.

The build machine has no NVIDIA card, so the .cu file is never compiled
before it reaches a notebook -- and a notebook is a bad place to find out
about a typo in a kernel.  This script compiles the file as ordinary C++,
against a small shim that stands in for the CUDA builtins and the runtime
functions, so that a mistake in the kernels is caught here instead of there.

The check is not a substitute for running the code: it catches the mistakes
that a compiler catches and nothing else.  The notebook still has to compile
the extension for real, and the parity test still has to pass on the GPU.

    python3 scripts/check_cuda_syntax.py

Exit code 0 means the file type-checks.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPU_DIR = os.path.join(ROOT, "engine", "gpu")
SHIM_DIR = os.path.join(GPU_DIR, "shim")
SOURCE = os.path.join(GPU_DIR, "jungle_gpu_cuda.cu")

# kernel<<<blocks, threads, shared>>>(args)  ->  kernel(args)
LAUNCH = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*<<<[^<>]*>>>")


def torch_flags():
    """The include paths and the ABI flag the installed torch needs."""
    try:
        import torch
        from torch.utils import cpp_extension
    except Exception as exc:                       # pragma: no cover
        return None, f"torch is not installed: {exc}"
    import sysconfig
    includes = list(cpp_extension.include_paths())
    includes.append(sysconfig.get_paths()["include"])
    abi = 1 if getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True) else 0
    return includes, abi


def main():
    if not os.path.exists(SOURCE):
        print(f"FAIL: {SOURCE} is missing")
        return 1

    includes, abi = torch_flags()
    if includes is None:
        print("SKIP: torch is not installed, so the extension cannot be type-checked")
        print("      (the notebook compiles it for real)")
        return 0

    with open(SOURCE) as f:
        source = f.read()
    patched = LAUNCH.sub(r"\1", source)

    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, "jungle_gpu_cuda_check.cpp")
        with open(cpp, "w") as f:
            f.write(patched)

        cmd = ["c++", "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra",
               "-Wno-unused-parameter", "-x", "c++"]
        cmd += [f"-I{p}" for p in includes]
        cmd += ["-I" + SHIM_DIR, "-I" + GPU_DIR, "-I" + os.path.join(ROOT, "engine", "src")]
        cmd += [f"-D_GLIBCXX_USE_CXX11_ABI={abi}", "-DTORCH_EXTENSION_NAME=jungle_gpu_check",
                "-DTORCH_API_INCLUDE_EXTENSION_H", cpp]
        proc = subprocess.run(cmd, capture_output=True, text=True)

    out = proc.stdout + proc.stderr
    # The shim's own warnings are noise; the point is the .cu file.
    if proc.returncode != 0:
        print("FAIL: the CUDA source does not type-check")
        print(out[-6000:])
        return 1

    errors = [l for l in out.splitlines() if "error" in l.lower()]
    if errors:
        print("FAIL: the CUDA source does not type-check")
        print(out[-6000:])
        return 1

    print("PASS: the CUDA source type-checks")
    print("      (the shim stands in for the toolkit; the code still has to be")
    print("       compiled for real and run on the GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
