#!/usr/bin/env python
"""Round-1 self-improvement gate: rollout failures must rank below successes.

Scores Q(s, a_true) along every episode of the autonomous rollout datasets for
a finetuned Q (Q1) and, optionally, a reference Q (Q0 — which never saw any
rollout episode). Per episode: terminal-window mean Q (last TERMINAL_W in-episode
frames) and a strided Q-vs-time curve. Aggregates AUROC of terminal-window Q
separating success from failure episodes.

Verdict (pre-registered for Phase-3 round 1) — PASS iff, for the finetuned Q:
  (r1) terminal-window AUROC (success vs failure, ALL rollout episodes) >= 0.8
  (r2) that AUROC >= the reference Q's on the same episodes (finetune helped)
  (r3) every HELD-OUT rollout failure's terminal-window Q < the median
       HELD-OUT rollout success terminal-window Q (small-n sanity; skipped
       with a warning if a repo held out no failures)

The wrapper (reward bookkeeping + train/holdout membership) is rebuilt from the
FINETUNED checkpoint's train_config.json — its bucket_overrides knows every
repo. Requires PYTHONHASHSEED=0 (same caveat as q_gate2_real).

    python scripts/q_gate_rollout_rank.py \
        --q_ckpt outputs/train/qf_stackcups_h32_q1/checkpoints/010000/pretrained_model \
        --ref_ckpt outputs/train/qf_stackcups_h32_negft/checkpoints/010000/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from q_gate2_real import (  # noqa: E402
    build_heldout_dataset,
    load_policy,
    load_train_cfg,
    sample_frame_offsets,
)

log = logging.getLogger("q_gate_rollout_rank")

DEFAULT_ROOT = "/storage/scratch1/6/vgiridhar6/lerobot_cache"
DEFAULT_REPOS = [
    "VarunGiridhar3/rollout_stackcups_bc_r0_right",
    "VarunGiridhar3/rollout_stackcups_bcq_r0_right",
]
SUCCESS_BUCKET = "q5"
FAILURE_BUCKET = "play"
TERMINAL_W = 16  # frames in the terminal window (~0.53 s @ 30 fps)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-aware AUROC: P(pos > neg) + 0.5 P(pos == neg)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def score_episodes(policy, preprocessor, wrapper, stride: int, device) -> list[dict]:
    from lerobot.policies.q_function.q_vis import _compute_chunk_q_values

    test_eps = {int(e) for e in wrapper.test_episode_ids}
    rows = []
    for ep_id in range(len(wrapper._ep_from)):
        f0 = int(wrapper._ep_from[ep_id].item())
        T = int(wrapper._ep_to[ep_id].item()) - f0
        bucket = wrapper._bucket_by_global_ep[ep_id]
        offs = sorted(set(sample_frame_offsets(T, stride, 0)) | set(range(max(0, T - TERMINAL_W), T)))
        q_by_off: dict[int, float] = {}
        for cs in range(0, len(offs), 4):
            group = offs[cs : cs + 4]
            items = [wrapper[f0 + off] for off in group]
            q = _compute_chunk_q_values(
                policy, preprocessor, items, num_perturb=0, perturb_std=0.0, device=device
            )
            for row_i, off in enumerate(group):
                q_by_off[off] = float(q[row_i, 0])
        term = [q_by_off[o] for o in range(max(0, T - TERMINAL_W), T)]
        rows.append(
            {
                "ep_id": ep_id,
                "bucket": bucket,
                "success": bucket == SUCCESS_BUCKET,
                "held_out": ep_id in test_eps,
                "T": T,
                "q_terminal_mean": float(np.mean(term)),
                "q_episode_mean": float(np.mean(list(q_by_off.values()))),
                "q_curve": [(int(o), q_by_off[o]) for o in offs],
            }
        )
        log.info(
            "ep %3d %-4s %-8s T=%3d  q_term=%.3f  q_mean=%.3f",
            ep_id, "TEST" if ep_id in test_eps else "trn", bucket, T,
            rows[-1]["q_terminal_mean"], rows[-1]["q_episode_mean"],
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    t = lambda rs: np.asarray([r["q_terminal_mean"] for r in rs])
    succ = [r for r in rows if r["success"]]
    fail = [r for r in rows if not r["success"]]
    ho_succ = [r for r in succ if r["held_out"]]
    ho_fail = [r for r in fail if r["held_out"]]
    return {
        "n_success": len(succ),
        "n_failure": len(fail),
        "n_heldout_success": len(ho_succ),
        "n_heldout_failure": len(ho_fail),
        "auroc_all": auroc(t(succ), t(fail)),
        "auroc_heldout": auroc(t(ho_succ), t(ho_fail)),
        "succ_term_median": float(np.median(t(succ))) if succ else float("nan"),
        "fail_term_median": float(np.median(t(fail))) if fail else float("nan"),
        "heldout_succ_term_median": float(np.median(t(ho_succ))) if ho_succ else float("nan"),
        "heldout_fail_term_max": float(t(ho_fail).max()) if ho_fail else float("nan"),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_ckpt", type=Path, required=True, help="finetuned Q pretrained_model dir")
    ap.add_argument("--ref_ckpt", type=Path, default=None, help="reference Q (e.g. Q0) to compare")
    ap.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    ap.add_argument("--dataset_root", default=DEFAULT_ROOT)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        log.warning("PYTHONHASHSEED != 0 — holdout membership may not match training!")

    out_dir = args.out_dir or args.q_ckpt.parent.parent / "gate_rollout_rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_cfg = load_train_cfg(args.q_ckpt)

    ckpts = {"finetuned": args.q_ckpt}
    if args.ref_ckpt:
        ckpts["reference"] = args.ref_ckpt

    results: dict[str, dict] = {name: {} for name in ckpts}
    for name, ckpt in ckpts.items():
        policy, preprocessor = load_policy(ckpt, device)
        log.info("=== %s: %s ===", name, ckpt)
        for repo in args.repos:
            wrapper, _, _ = build_heldout_dataset(train_cfg, repo, f"{args.dataset_root}/{repo}")
            rows = score_episodes(policy, preprocessor, wrapper, args.stride, device)
            results[name][repo] = {"episodes": rows, "summary": summarize(rows)}
        del policy
        torch.cuda.empty_cache()

    # combined summary over both repos
    for name in results:
        all_rows = [r for repo in args.repos for r in results[name][repo]["episodes"]]
        results[name]["combined"] = summarize(all_rows)

    fin = results["finetuned"]["combined"]
    ref = results.get("reference", {}).get("combined")
    r1 = fin["auroc_all"] >= 0.8
    r2 = ref is None or fin["auroc_all"] >= ref["auroc_all"]
    if fin["n_heldout_failure"] == 0:
        log.warning("no held-out rollout failures — (r3) skipped")
        r3 = True
    else:
        r3 = fin["heldout_fail_term_max"] < fin["heldout_succ_term_median"]
    verdict = {
        "r1_auroc_all_ge_0.8": bool(r1),
        "r2_auroc_ge_reference": bool(r2),
        "r3_heldout_fail_below_heldout_succ_median": bool(r3),
        "PASS": bool(r1 and r2 and r3),
    }

    lines = ["# Rollout ranking gate (Phase-3 round 1)", ""]
    for name in results:
        lines.append(f"## {name} ({ckpts[name]})")
        for key in [*args.repos, "combined"]:
            s = results[name][key]["summary"] if key != "combined" else results[name]["combined"]
            lines.append(
                f"- {key}: AUROC all={s['auroc_all']:.3f} heldout={s['auroc_heldout']:.3f} "
                f"(n={s['n_success']}s/{s['n_failure']}f, heldout {s['n_heldout_success']}s/"
                f"{s['n_heldout_failure']}f) | term-Q median succ={s['succ_term_median']:.3f} "
                f"fail={s['fail_term_median']:.3f}"
            )
        lines.append("")
    lines.append(f"## Verdict: {'PASS' if verdict['PASS'] else 'FAIL'}  {json.dumps(verdict)}")
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.md").write_text(summary + "\n")
    (out_dir / "results.json").write_text(json.dumps({"verdict": verdict, "results": results}, indent=2))
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
