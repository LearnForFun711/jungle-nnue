#!/usr/bin/env python3
"""Check that every code cell in the Colab notebook parses.

    python3 scripts/verify_notebook.py [colab/train_colab.ipynb]

The notebook is the interface the whole training programme runs through, and a
cell with a syntax error only shows up when the user runs it -- in the middle
of an eight hour session, which is the worst possible moment.  This check runs
in CI instead.

Jupyter cells are not always valid Python: `!pip install` and `%matplotlib`
are shell and magic commands.  IPython's own transformer turns them into
valid Python, so it is used when it is available; otherwise the cell is
parsed as plain Python and the magics are stripped.
"""
import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT = os.path.join(ROOT, "colab", "train_colab.ipynb")

# `!cmd`, `%magic` and `%%cell magic` are not Python.
MAGIC_LINE = re.compile(r"^\s*(\!|%{1,2})")


def strip_magics(source):
    """Turn a notebook cell into something ``ast.parse`` accepts."""
    out = []
    for line in source.splitlines():
        if MAGIC_LINE.match(line):
            # Keep the indentation of a magic line so that a magic inside an
            # indented block does not break the parse of the lines after it.
            indent = len(line) - len(line.lstrip())
            out.append(" " * indent + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def transformer():
    """IPython's transformer, if IPython is installed."""
    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except Exception:
        return None
    return TransformerManager()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        raise SystemExit(f"no such notebook: {path}")

    with open(path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    errors = []
    checked = 0
    tm = transformer()

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        checked += 1
        if tm is not None:
            try:
                tm.transform_cell(source)
                continue
            except Exception as exc:
                errors.append((i, f"IPython transform: {exc}"))
                continue
        try:
            ast.parse(strip_magics(source))
        except SyntaxError as exc:
            errors.append((i, f"SyntaxError: {exc}"))

    print(f"checked {checked} code cells in {os.path.basename(path)}")
    if errors:
        for i, msg in errors:
            print(f"  cell {i}: {msg}")
        print(f"\nFAIL: {len(errors)} cell(s) do not parse")
        return 1
    print("PASS: every code cell parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
