#!/usr/bin/env python3
"""Check that every function the device code calls is a device function.

nvcc has a rule the plain C++ compiler does not: a function called from
device code has to be marked `__device__` itself.  A host function called
from a kernel is a compile error, and the error is a page of template
output that does not say which of your functions was the problem.

The build machine has no CUDA toolkit, so the kernels are never compiled
before they reach a notebook.  This script stands in for the compiler's
check: it reads the GPU teacher's headers, works out which functions are
device functions, and then looks at every device function's body for calls
to functions that are not.

It found the first batch of them: the move generator calls `piece_type`
and `feature_index`, the teacher calls `move_prior`, and none of those
were marked, so the .cu file did not compile at all.

    python3 scripts/check_device_calls.py

Exit code 0 means every call the device code makes is a device call.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPU_DIR = os.path.join(ROOT, "engine", "gpu")

FILES = ["jungle_gpu.h", "jungle_gpu.cuh", "jungle_mcts.cuh", "jungle_game.cuh"]

# A function definition: a name, a parameter list with no braces or
# semicolons in it, and a body.  A *call* is followed by a comma or a
# semicolon, so it does not match.
DEFINITION = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{")
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Words that look like a function but are not one.
KEYWORDS = {"if", "else", "while", "for", "switch", "do", "return", "sizeof",
            "catch", "case", "and", "or", "not", "new", "delete", "throw",
            "assert", "static_cast", "reinterpret_cast", "const_cast"}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def body_of(text, brace):
    depth = 1
    i = brace + 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[brace + 1:i - 1], i


def scan():
    """Returns (device_functions, all_functions).

    device_functions maps a name to the list of (file, body) it was defined
    in; all_functions maps a name to the file it lives in.
    """
    device = {}
    everything = {}
    for name in FILES:
        path = os.path.join(GPU_DIR, name)
        if not os.path.exists(path):
            continue
        text = strip_comments(open(path).read())
        for match in DEFINITION.finditer(text):
            fname = match.group("name")
            if fname in KEYWORDS:
                continue
            # A call in a control statement -- `if (f()) {` -- looks like a
            # definition to the pattern above.  It is not one.
            before = text[max(0, match.start() - 12):match.start()]
            if re.search(r"(if|while|for|switch|elif)\s*$", before):
                continue
            brace = text.index("{", match.end() - 1)
            body, _ = body_of(text, brace)
            # Is the function marked as a device function?  The marker is in
            # the signature, which is the text between the previous
            # statement and the name.
            start = text.rfind(";", 0, match.start())
            start = max(start, text.rfind("}", 0, match.start()))
            start = max(start, text.rfind("{", 0, match.start()))
            prefix = text[start:match.start()]
            is_device = "JGPU_DEVICE" in prefix
            everything.setdefault(fname, name)
            if is_device:
                device.setdefault(fname, []).append((name, body))
    return device, everything


def main():
    device, everything = scan()

    # The functions that are not device functions, and so may not be called
    # from one.
    host_only = set()
    for fname, where in everything.items():
        if fname not in device:
            host_only.add(fname)

    problems = []
    for fname, entries in device.items():
        for where, body in entries:
            for call in CALL.findall(body):
                if call in KEYWORDS:
                    continue
                if call in host_only:
                    problems.append((where, fname, call))

    # The host's own entry points are allowed to call host functions.
    allowed = {"main", "run_gpu", "nnue_batch", "memory_model", "build_extension"}
    problems = [p for p in problems if p[1] not in allowed]

    if problems:
        print("FAIL: the device code calls functions that are not device functions")
        seen = set()
        for where, caller, callee in problems:
            key = (caller, callee)
            if key in seen:
                continue
            seen.add(key)
            print(f"  {where}: {caller}() calls {callee}(), which is not marked JGPU_DEVICE")
        print()
        print("  Every function the kernels reach has to be marked JGPU_DEVICE,")
        print("  or nvcc will refuse to compile the .cu file.")
        return 1

    ndev = sum(len(v) for v in device.values())
    print(f"PASS: every call the device code makes is a device call")
    print(f"      ({len(device)} device functions, {len(everything)} functions in all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
