"""Quarantine shard datasets that were torn by a mid-write kill.

A collection shard writes its LeRobotDataset incrementally and only at the very end
records held-out results. If SLURM kills it during the write (wall-clock timeout, OOM),
the directory survives with a half-written parquet in it. The finetune counts shard
directories and loads every one, so a torn shard takes down the whole iteration and,
because the chain only fires on exit 0, kills the loop.

Reading a parquet footer is enough to catch a torn file and costs almost nothing, so
this is safe to run on the login node. Bad shards are RENAMED, never deleted.

    python validate_shards.py <output_dir> <iteration>

Exit 0 always: a validation failure must not itself break the chain.
"""

import json
import sys
import time
from pathlib import Path


def check(ds: Path) -> str | None:
    """Return a reason string if the dataset looks torn, else None."""
    info = ds / "meta" / "info.json"
    if not info.exists():
        return "meta/info.json missing"
    try:
        meta = json.loads(info.read_text())
    except Exception as e:  # noqa: BLE001
        return f"meta/info.json unreadable: {e}"
    if not meta.get("total_episodes"):
        return "total_episodes is 0"

    parquets = sorted(ds.rglob("*.parquet"))
    if not parquets:
        return "no parquet files"
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None  # cannot check; assume fine rather than quarantine blindly
    for f in parquets:
        try:
            pq.read_metadata(f)
        except Exception as e:  # noqa: BLE001
            return f"unreadable parquet {f.relative_to(ds)}: {type(e).__name__}"
    return None


def main() -> int:
    out = Path(sys.argv[1])
    it = int(sys.argv[2])
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for ds in sorted(out.glob(f"iter_{it:03d}_shard*/online_episodes")):
        reason = check(ds)
        if reason is None:
            n = json.loads((ds / "meta" / "info.json").read_text()).get("total_episodes")
            print(f"  OK      {ds.parent.name}  ({n} episodes)")
            continue
        dest = ds.with_name(f"online_episodes.torn_{stamp}")
        ds.rename(dest)
        print(f"  QUARANTINED {ds.parent.name}: {reason}")
        print(f"              -> {dest.name} (renamed, not deleted)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"  validation errored ({type(e).__name__}: {e}) — leaving shards untouched")
        sys.exit(0)
