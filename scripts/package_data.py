#!/usr/bin/env python3
"""Package the self-play shards into a single archive for upload to Colab.

The dataset is the only thing that has to cross the network, so it is packed
into one file once.  The shards are 68-byte records with a lot of structure,
so they compress well; the script reports the ratio so it is obvious whether
the upload is worth doing in one go.

    python3 scripts/package_data.py --out data/jungle-dataset.tar.gz
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.t*.bin")
    ap.add_argument("--out", default="data/jungle-dataset.tar.gz")
    ap.add_argument("--keep", action="store_true", help="keep the shards as well")
    args = ap.parse_args()

    shards = sorted(glob.glob(args.data))
    if not shards:
        print(f"no shards matched {args.data!r}", file=sys.stderr)
        return 2

    raw = sum(os.path.getsize(s) for s in shards)
    positions = raw // 68
    print(f"{len(shards)} shards, {raw/1e6:.1f} MB raw, ~{positions/1000:.0f}k positions")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)

    with tarfile.open(args.out, "w:gz", compresslevel=6) as tar:
        for s in shards:
            tar.add(s, arcname=os.path.basename(s))

    packed = os.path.getsize(args.out)
    print(f"{args.out}: {packed/1e6:.1f} MB  (compression {raw/max(packed,1):.2f}x)")
    print(f"upload this file to Colab:  {os.path.abspath(args.out)}")

    if not args.keep:
        for s in shards:
            os.remove(s)
        print("shards removed (use --keep to keep them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
