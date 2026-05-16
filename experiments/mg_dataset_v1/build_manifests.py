"""Build policy.json (BC) and critic.json (Q-function) manifests for the v1 dataset.

Walks the dataset tree at
  /storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play/
        <task>/<bucket>/{demo.hdf5,demo_failed.hdf5}

and produces two JSON manifests:

  manifests/policy.json:
    BC training set. Lists every successful demo across every (task, bucket).
    Each entry: {file, demo_key, task, bucket}.

  manifests/critic.json:
    Q-function training set. Lists every demo (success and failed) across
    every (task, bucket). Each entry adds a `success` flag so the loader
    knows the trajectory outcome and `bucket` so it can pick a per-bucket
    reward labelling at training time.

Both manifests stay format-stable across dataset versions, so the loader
side of the pipeline does not have to know how the buckets were generated.

Run after data gen + play rollouts finish:
    python build_manifests.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

DEFAULT_ROOT = Path(
    "/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
)


def list_demos_in_hdf5(path: Path) -> list[str]:
    """Return the sorted list of demo keys inside the file, or [] if missing."""
    if not path.exists():
        return []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return []
        # Sort numerically by trailing index when possible.
        keys = list(f["data"].keys())
        try:
            keys.sort(key=lambda k: int(k.split("_")[-1]))
        except Exception:
            keys.sort()
        return keys


TOP_LEVEL_NON_TASK = {"configs", "manifests", "bc_ckpts"}
# Working dirs inside each task that aren't real buckets:
#   - bc_input: merged q5+q3_termjitter we feed to convert_mimicgen_to_lerobot;
#     listing it would double-count those demos in the manifest.
NON_BUCKET_DIRS = {"bc_input"}


def _hdf5_paths(bucket_dir: Path, fname: str) -> Path | None:
    """Resolve where a bucket's demo.hdf5 / demo_failed.hdf5 actually lives.

    MimicGen writes <bucket_dir>/<bucket_dir>/<fname> (the inner dir is the
    'experiment.name' from the MG config). Our play_rollout.py writes
    <bucket_dir>/<fname> directly. Try both.
    """
    nested = bucket_dir / bucket_dir.name / fname
    if nested.exists():
        return nested
    flat = bucket_dir / fname
    if flat.exists():
        return flat
    return None


def collect_entries(root: Path) -> list[dict]:
    """Walk <task>/<bucket>/.../{demo,demo_failed}.hdf5 → one entry per demo."""
    entries: list[dict] = []
    for task_dir in sorted(p for p in root.iterdir()
                           if p.is_dir() and p.name not in TOP_LEVEL_NON_TASK):
        task = task_dir.name
        for bucket_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            bucket = bucket_dir.name
            if bucket in NON_BUCKET_DIRS:
                continue
            for fname, success in [("demo.hdf5", True), ("demo_failed.hdf5", False)]:
                hpath = _hdf5_paths(bucket_dir, fname)
                if hpath is None:
                    continue
                rel = hpath.relative_to(root)
                for k in list_demos_in_hdf5(hpath):
                    entries.append({
                        "file": str(rel),       # relative path so the manifest is portable
                        "demo_key": k,
                        "task": task,
                        "bucket": bucket,
                        "success": success,
                    })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    root = args.root.resolve()
    out_dir = root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(root)
    if not entries:
        raise SystemExit(f"No demos found under {root}")

    # critic = everything
    critic = {
        "root": str(root),
        "count": len(entries),
        "entries": entries,
    }
    (out_dir / "critic.json").write_text(json.dumps(critic, indent=2))

    # policy = success-only across all buckets
    policy_entries = [e for e in entries if e["success"]]
    policy = {
        "root": str(root),
        "count": len(policy_entries),
        "entries": policy_entries,
    }
    (out_dir / "policy.json").write_text(json.dumps(policy, indent=2))

    # Per-task / per-bucket summary
    print(f"Wrote {out_dir / 'policy.json'}  ({policy['count']} demos)")
    print(f"Wrote {out_dir / 'critic.json'}  ({critic['count']} demos)")
    print()
    print(f"{'task':<12} {'bucket':<16} {'success':>8} {'failed':>8} {'total':>8}")
    summary: dict[tuple[str, str], dict] = {}
    for e in entries:
        key = (e["task"], e["bucket"])
        s = summary.setdefault(key, {"success": 0, "failed": 0})
        s["success" if e["success"] else "failed"] += 1
    for (task, bucket), s in sorted(summary.items()):
        print(f"{task:<12} {bucket:<16} {s['success']:>8} {s['failed']:>8} {s['success']+s['failed']:>8}")


if __name__ == "__main__":
    main()
