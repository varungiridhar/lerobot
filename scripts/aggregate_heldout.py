"""Pool one iteration's per-shard rollout results into a single record.

Each collection shard scores its own ~6 tasks and writes heldout_eval.json. This
combines them into the numbers to actually read:

  heldout_pc_success  success on held-out episodes only. Their seeds are fixed across
                      iterations and are never trained on, so this is the metric to
                      TREND — identical seeds every iteration make it a paired
                      comparison and seed variance cancels.
  overall_pc_success  success on every episode rolled out (held-out + training). Twice
                      the episodes, so the tighter estimate of CURRENT performance.
                      Not leakage: the training episodes are rolled out with the current
                      Q before it is finetuned on them.

Also reports shard coverage. A missing shard removes ~6 tasks from BOTH numbers, which
would silently make that iteration's rate incomparable to the others — so an incomplete
iteration is flagged rather than quietly averaged.

    python scripts/aggregate_heldout.py <output_dir> <iteration> [expected_shards]
"""

import glob
import json
import math
import sys
from pathlib import Path


def _pooled(vals):
    """(rate, n) from [(pct, count), ...], ignoring NaN/zero-count entries."""
    n = tot = 0
    for pct, cnt in vals:
        if cnt and pct is not None and not math.isnan(pct):
            n += cnt
            tot += pct * cnt / 100.0
    return (100.0 * tot / n if n else float("nan")), n


def main() -> int:
    out = Path(sys.argv[1])
    it = int(sys.argv[2])
    expected = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    files = sorted(glob.glob(str(out / f"iter_{it:03d}_shard*" / "heldout_eval.json")))
    if not files:
        print(f"No held-out results for iteration {it} "
              f"(expected if this iteration predates the merged eval stage).")
        return 0

    held, over = [], []
    for f in files:
        d = json.loads(Path(f).read_text())
        held.append((d.get("eval_pc_success"), d.get("n_eval", 0)))
        if "overall_pc_success" in d:
            over.append((d.get("overall_pc_success"), d.get("n_overall", 0)))
        else:
            # Shards written before overall_pc_success existed: derive it, so a
            # mid-flight upgrade does not leave a hole in the series.
            over.append((d.get("eval_pc_success"), d.get("n_eval", 0)))
            over.append((d.get("train_pc_success"), d.get("n_train", 0)))

    h_pc, h_n = _pooled(held)
    o_pc, o_n = _pooled(over)

    complete = (expected == 0) or (len(files) == expected)
    print(f"=== iteration {it} rollout ({len(files)}"
          + (f"/{expected}" if expected else "") + " shards) ===")
    print(f"  OVERALL  {o_pc:5.1f}%  (n={o_n})   <- tighter estimate of current performance")
    print(f"  HELD-OUT {h_pc:5.1f}%  (n={h_n})   <- compare THIS across iterations")
    if not complete:
        print(f"  !! INCOMPLETE: {expected - len(files)} shard(s) missing. Those tasks are "
              f"absent from both rates, so this iteration is NOT directly comparable to "
              f"a complete one — compare on the shared task subset instead.")

    rec = {
        "iteration": it,
        "heldout_pc_success": h_pc, "n_heldout": h_n,
        "overall_pc_success": o_pc, "n_overall": o_n,
        "n_shards": len(files), "n_shards_expected": expected,
        "complete": complete,
    }
    (out / f"iter_{it:03d}_heldout.json").write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
