"""Bypass MG's HEAD-check (broken on this cluster) and call gdown directly.

MG's download_url_from_gdrive() guards on robomimic.url_is_alive(), which
issues an HTTP HEAD against drive.google.com/file/d/<ID>/view; the cluster
gets a non-2xx for that probe even when the actual file is reachable. We
keep MG's URL list as the source of truth and just call gdown.download().
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import gdown
import mimicgen  # noqa: F401 — populates DATASET_REGISTRY
from mimicgen import DATASET_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks", nargs="+", default=["square", "threading", "coffee"],
        help="MG source-task identifiers (no _d0 suffix)."
    )
    ap.add_argument(
        "--download_dir", required=True,
        help="Output directory; HDF5s land at <download_dir>/<task>.hdf5"
    )
    args = ap.parse_args()

    target = Path(args.download_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    src_reg = DATASET_REGISTRY["source"]
    print("Available 'source' tasks:", sorted(src_reg.keys()))

    for task in args.tasks:
        if task not in src_reg:
            raise SystemExit(f"task '{task}' not in DATASET_REGISTRY['source']")
        url = src_reg[task]["url"]
        print(f"\n=== {task} ===  {url}", flush=True)
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            try:
                os.chdir(td)
                fpath = gdown.download(url, quiet=False, fuzzy=True)
            finally:
                os.chdir(cwd)
            if not fpath or not os.path.exists(fpath):
                raise SystemExit(f"gdown returned no file for {task}")
            final = target / f"{task}.hdf5"
            shutil.move(fpath, str(final))
            print(f"  -> {final}  ({os.path.getsize(final):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
