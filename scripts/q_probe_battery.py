#!/usr/bin/env python
"""Stage-0 forensic Q-probe battery (rebuttal debugging).

Scores Q(s, a) for the TRUE held-out action chunk against a battery of
alternative chunks on held-out test episodes, for one or more Q-function
checkpoints. Distinguishes three failure hypotheses for the BC+play Q:

  H1 magnitude shortcut — Q keys on play-like action statistics (large rotation
     components, non-binary gripper) rather than state-action match: rotation-only
     probes collapse while in-distribution wrong-episode chunks score HIGH.
  H2 OOD-only extrapolation — both checkpoints collapse on white-noise probes but
     decay gradually on planner-matched smoothed noise, and wrong-episode chunks
     score high on both: Q has no in-distribution ranking signal.
  H3 normalization compression — the BC+play checkpoint shows near-zero Q spread
     over planner-matched candidates while the BC-only checkpoint does not.

Probes (all built in RAW action space; the checkpoint's own preprocessor
normalizes, exactly as in training / q_vis):
  white03          legacy q_vis probe: i.i.d. N(0, 0.3) on all dims, clamp ±3
  smooth_{0.5,1,2}s per-dim std-scaled Gaussian, temporally smoothed (sigma_t=2)
                   like the MPPI planner, gripper held at the true value
  grip_flip        gripper dim sign-flipped, arm dims true
  grip_noise       white noise on the gripper dim only
  rot_bcsig        white noise on rotation dims at 1 sigma of the BC data
  rot_playsig      white noise on rotation dims at 1 sigma of the play data
  wrong_ep         TRUE chunk from a different BC test episode (different task
                   when possible) at the same episode fraction — smooth,
                   realistic, but wrong for this state
  cross_bucket     play chunk at a BC state / BC chunk at a play state
  reversed         the true chunk, time-reversed
  shuffled         the true chunk, timesteps randomly permuted

Metrics per (checkpoint x bucket x probe): mean Q_true, mean Q_probe, gap,
AUROC(true vs probe), top-1 ranking accuracy, plus the per-frame Q std over the
smooth_1s candidates (planner-relevant resolution, comparable to the q_std the
planner logs).

Usage (GPU node, lerobot-q env):
    python scripts/q_probe_battery.py \
        --ckpt bc12k=outputs/train/2026-05-19/23-54-42_qf_libero_ddp2_bsz48_bc_h200/checkpoints/last/pretrained_model \
        --ckpt bcplay12k=outputs/train/2026-05-19/23-54-24_qf_libero_ddp2_bsz48_bc_plus_play_h200/checkpoints/012000/pretrained_model \
        --dataset-config outputs/train/2026-05-19/23-54-24_qf_libero_ddp2_bsz48_bc_plus_play_h200/checkpoints/last/pretrained_model/train_config.json \
        --out-dir ~/scratch/qplanning_rebuttal/probe/run1

The dataset config determines the episode pool + deterministic holdout split.
Because the split seed is hash((cfg.seed, repo_id, bucket)), the BC test
episodes of the BC+play config coincide with the BC-only run's test episodes,
so every checkpoint is evaluated on states it never trained on.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("q_probe_battery")

ROT_DIMS = (3, 4, 5)  # LIBERO 7-dim action: xyz, axis-angle rot, gripper
GRIP_DIM = -1


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(config_path: Path):
    """Rebuild the exact QValueLabelDataset (incl. holdout split) of a training run."""
    import draccus

    # Importing the config registers the 'q_function' choice with draccus.
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.q_function.configuration_q_function import QFunctionConfig  # noqa: F401

    cfg = draccus.parse(TrainPipelineConfig, config_path=str(config_path), args=[])
    dataset = make_dataset(cfg)
    if not getattr(dataset, "has_holdout", False):
        raise RuntimeError("dataset config has no held-out test split (test_split_ratio=0?)")
    return dataset, cfg


def load_policy(ckpt_path: Path, device: torch.device):
    from lerobot.policies.q_function.modeling_q_function import QFunctionPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

    policy = QFunctionPolicy.from_pretrained(str(ckpt_path)).to(device).eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(ckpt_path),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    return policy, preprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Episode selection + preloading
# ──────────────────────────────────────────────────────────────────────────────


def bucket_of_episode(dataset, ep_id: int) -> str:
    f = int(dataset._ep_from[ep_id].item())
    ds_idx = int(dataset._dataset_idx_by_frame[f].item())
    return dataset._dataset_index_to_bucket[ds_idx]


def action_sigmas(dataset) -> dict[str, np.ndarray]:
    """Per-dim action std for each bucket (pooled by simple mean across sub-datasets)."""
    from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

    raw = dataset.dataset
    out: dict[str, list[np.ndarray]] = {}
    subs = raw._datasets if isinstance(raw, MultiLeRobotDataset) else [raw]
    for ds_idx, sub in enumerate(subs):
        bucket = dataset._dataset_index_to_bucket[ds_idx]
        out.setdefault(bucket, []).append(np.asarray(sub.meta.stats["action"]["std"], dtype=np.float64))
    return {b: np.mean(np.stack(v), axis=0) for b, v in out.items()}


def select_episodes(dataset, per_bucket: int, rng: random.Random) -> dict[str, list[int]]:
    by_bucket: dict[str, list[int]] = {}
    for ep_id in dataset.test_episode_ids:
        by_bucket.setdefault(bucket_of_episode(dataset, int(ep_id)), []).append(int(ep_id))
    chosen: dict[str, list[int]] = {}
    for bucket, eps in by_bucket.items():
        eps = sorted(eps)
        if len(eps) <= per_bucket:
            chosen[bucket] = eps
        else:
            idx = np.linspace(0, len(eps) - 1, num=per_bucket, dtype=int)
            chosen[bucket] = [eps[i] for i in idx]
    for bucket, eps in chosen.items():
        log.info("bucket %-4s: probing %d / %d test episodes", bucket, len(eps), len(by_bucket[bucket]))
    return chosen


def preload_episode(dataset, ep_id: int, stride: int, max_frames: int) -> dict:
    """Load strided items of one episode. Returns dict with items, fracs, actions, task."""
    from lerobot.utils.constants import ACTION

    from_idx = int(dataset._ep_from[ep_id].item())
    to_idx = int(dataset._ep_to[ep_id].item())
    T = to_idx - from_idx
    n = min(max(2, T // stride), max_frames)
    frame_ids = np.unique(np.linspace(from_idx, to_idx - 1, num=n, dtype=int))
    items, fracs, actions = [], [], []
    task = ""
    for fi in frame_ids:
        item = dataset[int(fi)]
        ts = item.get("task", "")
        task = ts[0] if isinstance(ts, list) else str(ts)
        items.append(item)
        fracs.append((int(fi) - from_idx) / max(1, T - 1))
        actions.append(item[ACTION].clone())
    return {
        "ep_id": ep_id,
        "task": task,
        "items": items,
        "fracs": np.asarray(fracs),
        "actions": actions,  # list of (L, A) raw tensors
        "T": T,
    }


def chunk_at_frac(ep: dict, frac: float) -> torch.Tensor:
    i = int(np.argmin(np.abs(ep["fracs"] - frac)))
    return ep["actions"][i]


# ──────────────────────────────────────────────────────────────────────────────
# Candidate builders
# ──────────────────────────────────────────────────────────────────────────────


def smoothed_scaled_noise(
    true: torch.Tensor, sigma_vec: np.ndarray, scale: float, n: int, gen: torch.Generator
) -> torch.Tensor:
    """Planner-matched noise: per-dim std-scaled, temporally smoothed, gripper held."""
    from lerobot.policies.act_simple.planning import _smooth_time

    L, A = true.shape
    noise = torch.randn(n, L, A, generator=gen)
    noise = noise * torch.as_tensor(sigma_vec, dtype=noise.dtype) * scale
    noise = _smooth_time(noise, sigma=2.0)
    noise[..., GRIP_DIM] = 0.0
    return (true.unsqueeze(0) + noise).clamp(-1.0, 1.0)


def build_candidates(
    true: torch.Tensor,
    frac: float,
    task: str,
    bucket: str,
    sigmas: dict[str, np.ndarray],
    bc_pool: list[dict],
    play_pool: list[dict],
    own_ep_id: int,
    gen: torch.Generator,
    rng: random.Random,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    """Return (K, L, A) candidate chunks + ordered [(probe_name, n), ...] layout."""
    L, A = true.shape
    sigma_bc = sigmas.get("q5")
    sigma_play = sigmas.get("play")
    groups: list[tuple[str, torch.Tensor]] = []

    # (a) legacy white noise (the current q_vis probe, kept for continuity)
    noise = torch.randn(4, L, A, generator=gen) * 0.3
    groups.append(("white03", (true.unsqueeze(0) + noise).clamp(-3.0, 3.0)))

    # (b) planner-matched smoothed noise at graded scales (1-sigma gets extra
    # samples — its per-frame spread doubles as the planner-resolution metric)
    if sigma_bc is not None:
        groups.append(("smooth_0.5s", smoothed_scaled_noise(true, sigma_bc, 0.5, 4, gen)))
        groups.append(("smooth_1s", smoothed_scaled_noise(true, sigma_bc, 1.0, 8, gen)))
        groups.append(("smooth_2s", smoothed_scaled_noise(true, sigma_bc, 2.0, 4, gen)))

    # (c) gripper-only probes
    flip = true.clone().unsqueeze(0)
    flip[..., GRIP_DIM] = -flip[..., GRIP_DIM]
    groups.append(("grip_flip", flip))
    gn = true.unsqueeze(0).repeat(2, 1, 1)
    gn[..., GRIP_DIM] = (gn[..., GRIP_DIM] + torch.randn(2, L, generator=gen)).clamp(-1.0, 1.0)
    groups.append(("grip_noise", gn))

    # (d) rotation-dims-only white noise at BC-sigma and play-sigma
    if sigma_bc is not None:
        rn = true.unsqueeze(0).repeat(3, 1, 1)
        for d in ROT_DIMS:
            rn[..., d] += torch.randn(3, L, generator=gen) * float(sigma_bc[d])
        groups.append(("rot_bcsig", rn.clamp(-1.0, 1.0)))
    if sigma_play is not None:
        rn = true.unsqueeze(0).repeat(3, 1, 1)
        for d in ROT_DIMS:
            rn[..., d] += torch.randn(3, L, generator=gen) * float(sigma_play[d])
        groups.append(("rot_playsig", rn.clamp(-1.0, 1.0)))

    # (e) in-distribution wrong chunk: true chunk of a DIFFERENT BC test episode
    # at the same episode fraction (prefer a different task string)
    sources = [ep for ep in bc_pool if ep["ep_id"] != own_ep_id]
    diff_task = [ep for ep in sources if ep["task"] != task]
    pick_from = diff_task if diff_task else sources
    if pick_from:
        picks = rng.sample(pick_from, min(4, len(pick_from)))
        groups.append(("wrong_ep", torch.stack([chunk_at_frac(ep, frac) for ep in picks])))

    # (f) cross-bucket: play chunk at a BC state / BC chunk at a play state
    cross_pool = play_pool if bucket == "q5" else bc_pool
    cross_pool = [ep for ep in cross_pool if ep["ep_id"] != own_ep_id]
    if cross_pool:
        picks = rng.sample(cross_pool, min(2, len(cross_pool)))
        groups.append(("cross_bucket", torch.stack([chunk_at_frac(ep, frac) for ep in picks])))

    # (g) temporal-structure probes on the true chunk
    groups.append(("reversed", torch.flip(true, dims=[0]).unsqueeze(0)))
    perm = torch.randperm(L, generator=gen)
    groups.append(("shuffled", true[perm].unsqueeze(0)))

    layout = [(name, t.shape[0]) for name, t in groups]
    return torch.cat([t for _, t in groups], dim=0), layout


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-averaged rank AUROC: P(Q_true > Q_probe)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_v = np.concatenate([pos, neg])
    order = np.argsort(all_v, kind="mergesort")
    ranks = np.empty(len(all_v))
    sv = all_v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# ──────────────────────────────────────────────────────────────────────────────
# Main probing loop
# ──────────────────────────────────────────────────────────────────────────────


def probe_checkpoint(
    label: str,
    ckpt_path: Path,
    dataset,
    episodes: dict[str, list[dict]],
    sigmas: dict[str, np.ndarray],
    device: torch.device,
    out_dir: Path,
    seed: int,
    frames_per_fwd: int,
) -> list[dict]:
    from lerobot.policies.q_function.q_vis import _compute_chunk_q_values

    policy, preprocessor = load_policy(ckpt_path, device)
    log.info("[%s] loaded policy h=%d from %s", label, policy.config.h, ckpt_path)

    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    bc_pool = episodes.get("q5", [])
    play_pool = episodes.get("play", [])

    # records[bucket][probe] = {"true": [...], "probe": [...], per-frame lists}
    records: dict[str, dict[str, dict[str, list]]] = {}
    frame_rows: list[dict] = []

    for bucket, eps in episodes.items():
        for ep in eps:
            S = len(ep["items"])
            for cs in range(0, S, frames_per_fwd):
                ce = min(cs + frames_per_fwd, S)
                cand_list, layouts = [], []
                for k in range(cs, ce):
                    cand, layout = build_candidates(
                        true=ep["actions"][k],
                        frac=float(ep["fracs"][k]),
                        task=ep["task"],
                        bucket=bucket,
                        sigmas=sigmas,
                        bc_pool=bc_pool,
                        play_pool=play_pool,
                        own_ep_id=ep["ep_id"],
                        gen=gen,
                        rng=rng,
                    )
                    cand_list.append(cand)
                    layouts.append(layout)
                # All frames in a batch share probe layout sizes by construction
                # (pools are global), so stacking is safe.
                candidates = torch.stack(cand_list)  # (ct, K, L, A)
                q = _compute_chunk_q_values(
                    policy,
                    preprocessor,
                    [ep["items"][k] for k in range(cs, ce)],
                    num_perturb=0,
                    perturb_std=0.0,
                    device=device,
                    candidates=candidates,
                )  # (ct, 1 + K)
                for row_i, k in enumerate(range(cs, ce)):
                    q_true = float(q[row_i, 0])
                    frame_row = {
                        "bucket": bucket,
                        "ep_id": ep["ep_id"],
                        "frac": float(ep["fracs"][k]),
                        "q_true": q_true,
                    }
                    off = 1
                    for name, n in layouts[row_i]:
                        vals = q[row_i, off : off + n].numpy()
                        off += n
                        rec = (
                            records.setdefault(bucket, {})
                            .setdefault(name, {"true": [], "probe": [], "win": [], "spread": []})
                        )
                        rec["true"].append(q_true)
                        rec["probe"].extend(vals.tolist())
                        rec["win"].append(float(q_true > vals.max()))
                        if name == "smooth_1s":
                            rec["spread"].append(float(np.std(np.concatenate([[q_true], vals]))))
                        frame_row[name] = vals
                    frame_rows.append(frame_row)
            log.info("[%s] bucket=%-4s ep=%d done (%d frames)", label, bucket, ep["ep_id"], S)

    # ── Summaries ──
    summary_rows = []
    for bucket, probes in records.items():
        for name, rec in probes.items():
            pos, neg = np.asarray(rec["true"]), np.asarray(rec["probe"])
            summary_rows.append(
                {
                    "ckpt": label,
                    "bucket": bucket,
                    "probe": name,
                    "n_frames": len(rec["true"]),
                    "q_true_mean": round(float(pos.mean()), 4),
                    "q_probe_mean": round(float(neg.mean()), 4),
                    "gap": round(float(pos.mean() - neg.mean()), 4),
                    "auroc": round(auroc(pos, neg), 4),
                    "top1_acc": round(float(np.mean(rec["win"])), 4),
                    "q_std_smooth1s": round(float(np.mean(rec["spread"])), 4) if rec["spread"] else "",
                }
            )

    ckpt_dir = out_dir / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        ckpt_dir / "frames.npy",
        np.array(
            [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fr.items()}
                for fr in frame_rows
            ],
            dtype=object,
        ),
    )
    with open(ckpt_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("[%s] wrote %s", label, ckpt_dir / "summary.csv")

    del policy
    torch.cuda.empty_cache()
    return summary_rows


def write_markdown(all_rows: list[dict], out_path: Path) -> None:
    cols = ["ckpt", "bucket", "probe", "n_frames", "q_true_mean", "q_probe_mean", "gap", "auroc", "top1_acc", "q_std_smooth1s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in sorted(all_rows, key=lambda r: (r["ckpt"], r["bucket"], r["probe"])):
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH",
                    help="Q checkpoint pretrained_model dir; repeatable")
    ap.add_argument("--dataset-config", type=Path, required=True,
                    help="train_config.json of the run whose dataset/holdout split to probe on")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--episodes-per-bucket", type=int, default=12)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-frames-per-ep", type=int, default=40)
    ap.add_argument("--frames-per-fwd", type=int, default=2,
                    help="dataset frames batched per Q forward (B = this x ~38 candidates)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="build dataset + candidates for one episode, skip policy loading/scoring")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    ckpts: list[tuple[str, Path]] = []
    for spec in args.ckpt:
        label, _, path = spec.partition("=")
        if not path:
            ap.error(f"--ckpt must be LABEL=PATH, got {spec!r}")
        ckpts.append((label, Path(path).expanduser()))

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading dataset from %s", args.dataset_config)
    dataset, _cfg = load_dataset(args.dataset_config.expanduser())
    sigmas = action_sigmas(dataset)
    for b, s in sigmas.items():
        log.info("action sigma[%s] = %s", b, np.round(s, 3).tolist())

    chosen = select_episodes(dataset, args.episodes_per_bucket, rng)

    if args.dry_run:
        bucket = next(iter(chosen))
        ep = preload_episode(dataset, chosen[bucket][0], args.stride, args.max_frames_per_ep)
        cand, layout = build_candidates(
            true=ep["actions"][0], frac=0.0, task=ep["task"], bucket=bucket, sigmas=sigmas,
            bc_pool=[ep], play_pool=[], own_ep_id=-1, gen=torch.Generator().manual_seed(0), rng=rng,
        )
        log.info("dry-run OK: ep %d (%s) %d strided frames; candidates %s; layout %s",
                 ep["ep_id"], bucket, len(ep["items"]), tuple(cand.shape), layout)
        return

    log.info("Preloading strided items for all probe episodes (video decode)...")
    episodes: dict[str, list[dict]] = {}
    for bucket, ep_ids in chosen.items():
        episodes[bucket] = [
            preload_episode(dataset, ep_id, args.stride, args.max_frames_per_ep) for ep_id in ep_ids
        ]
        log.info("bucket %-4s: preloaded %d episodes", bucket, len(episodes[bucket]))

    device = torch.device(args.device)
    all_rows: list[dict] = []
    for label, path in ckpts:
        all_rows.extend(
            probe_checkpoint(
                label, path, dataset, episodes, sigmas, device, out_dir, args.seed, args.frames_per_fwd
            )
        )
    write_markdown(all_rows, out_dir / "summary.md")
    log.info("All done → %s", out_dir)


if __name__ == "__main__":
    main()
