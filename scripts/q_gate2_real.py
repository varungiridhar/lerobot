#!/usr/bin/env python
"""G2 offline gate for a Q-function trained on a REAL kinesthetic dataset.

Scores Q(s, a_true) for held-out test episodes of the real stack-cups dataset
(LeRobot v3, 7-dim joint-space action: right_joint_1..6 + right_gripper; the
gripper is the LAST dim in [0, 1], joints are radians) against a battery of
alternative action chunks, then applies a PASS/FAIL verdict on whether the Q
is fit to steer a planner on real data.

Held-out split
--------------
The script reconstructs the training-time holdout by rebuilding the exact
``QValueLabelDataset`` wrapper the training run used: it reads
``<q_ckpt>/train_config.json`` (test_split_ratio, seed, policy.h,
policy.step_reward, policy.terminal_bonuses, policy.bucket_overrides,
policy.reward_mode) and wraps a fresh ``LeRobotDataset`` with the same
parameters. ``bucket_overrides={repo: "episode_labels"}`` resolves per-episode
buckets from ``meta/episode_labels.json``: success -> "q5", failure -> "play".

CAVEAT (split reproducibility): the wrapper seeds the per-(repo, bucket)
shuffle with ``hash((seed, repo_id, bucket))``. Python string hashing is
salted per process unless PYTHONHASHSEED is set, and neither the training
launcher nor accelerate pins it — so the reconstructed held-out episode IDs
are only guaranteed identical to training's if this process shares the
training process's hash salt. Bucket-level holdout COUNTS always match
(10 success + 1 failure at test_split_ratio=0.1). The script warns when
PYTHONHASHSEED is unset and records the episode IDs it actually used in
``results.json`` so the split is auditable.

Probes (built in RAW action space; the checkpoint's own preprocessor
normalizes, exactly as in training / q_vis). Unlike the sim probe battery,
tube candidates are NOT clamped to +/-1 (wrong for radian joints) — they are
clamped to the dataset's per-dim action min/max instead:

  tube_0.5s / tube_1s / tube_2s
              a_true + per-dim Gaussian noise (sigma = dataset action std
              scaled by 0.5 / 1.0 / 2.0), temporally smoothed like the MPPI
              planner (sigma_t=2.0), gripper noise zeroed. 2 variants each.
  swap        TRUE chunk of a DIFFERENT held-out success episode at the same
              episode fraction. 2 variants from 2 different episodes.
  reversed    the true chunk, time-reversed.
  shuffled    the true chunk, timesteps randomly permuted.
  grip_flip   the true chunk with gripper -> 1 - gripper.
  hold        a_true[0] repeated h times (the "freeze" chunk —
              kinesthetic-relevant: action[t] ~= state[t]).

Metrics per (bucket, probe): mean Q_true, mean Q_probe, gap, rank_acc
(= mean over all probe variants of Q_true > Q_probe), tie-aware AUROC.
Plus Q-vs-time per held-out episode: Spearman rho of Q_true vs frame index,
start/end Q. Success episodes should rise (rho > 0); failure should not.

Verdict — PASS iff
  (a) mean rank_acc over {swap, reversed, shuffled, hold} on success-bucket
      frames >= 0.8,
  (b) median success-episode Spearman rho > 0.3,
  (c) failure-episode terminal Q_true < median success terminal Q_true.

Outputs: <out_dir>/summary.md (also printed to stdout) + <out_dir>/results.json.

Runtime environment (PACE, GPU node):
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate lerobot-q
    export PYTHONPATH=/storage/project/r-agarg35-0/vgiridhar6/lerobot/src
    python scripts/q_gate2_real.py \\
        --q_ckpt outputs/train/qf_stackcups_h32/checkpoints/030000/pretrained_model

sbatch-style invocation (from the repo root):
    sbatch --account=gts-agarg35 -N1 --gres=gpu:L40S:1 -q embers \\
        --mem-per-gpu=64G --cpus-per-gpu=4 -t4:00:00 \\
        -o slurm_out/Report-%j.out -J q_gate2_real \\
        --wrap 'source "$(conda info --base)/etc/profile.d/conda.sh" \\
                && conda activate lerobot-q \\
                && export PYTHONPATH="$PWD/src" HF_HOME=/storage/scratch1/6/vgiridhar6/hf \\
                       TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \\
                && python scripts/q_gate2_real.py \\
                       --q_ckpt outputs/train/qf_stackcups_h32/checkpoints/030000/pretrained_model'
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("q_gate2_real")

GRIP_DIM = -1
SMOOTH_SIGMA_T = 2.0            # temporal smoothing bandwidth, matches the MPPI planner
TUBE_SPECS = (("tube_0.5s", 0.5), ("tube_1s", 1.0), ("tube_2s", 2.0))
N_TUBE = 2                      # variants per tube scale
N_SWAP = 2                      # wrong-episode variants per frame
SUCCESS_BUCKET = "q5"
FAILURE_BUCKET = "play"
VERDICT_PROBES = ("swap", "reversed", "shuffled", "hold")

DEFAULT_REPO_ID = "VarunGiridhar3/KT_stack_cups_20260807_211827_right"
DEFAULT_DATASET_ROOT = (
    "/storage/scratch1/6/vgiridhar6/lerobot_qplanning_real/"
    "VarunGiridhar3__KT_stack_cups_20260807_211827_right"
)


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────


def load_train_cfg(q_ckpt: Path) -> dict:
    cfg_path = q_ckpt / "train_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found — pass the pretrained_model dir as --q_ckpt")
    return json.loads(cfg_path.read_text())


def load_policy(ckpt_path: Path, device: torch.device):
    """Same loading pattern as scripts/q_probe_battery.py."""
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


def build_heldout_dataset(train_cfg: dict, repo_id: str, dataset_root: str):
    """Rebuild the training run's QValueLabelDataset (incl. holdout split).

    Underlying LeRobotDataset delta_timestamps: [0] for each camera (predict_value
    only reads the current frame) and an h-length action window (predict_value
    truncates to h anyway — no need for training's 2h bootstrap window).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    pol = train_cfg["policy"]
    h = int(pol["h"])
    meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    fps = float(meta.fps)
    camera_keys = list(pol.get("camera_keys") or meta.camera_keys)

    delta_timestamps: dict[str, list[float]] = {ck: [0.0] for ck in camera_keys}
    delta_timestamps["action"] = [i / fps for i in range(h)]

    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=float(train_cfg.get("tolerance_s", 1e-4)),
        video_backend="pyav",
    )
    wrapper = QValueLabelDataset(
        dataset,
        h=h,
        step_reward=float(pol["step_reward"]),
        terminal_bonuses=dict(pol["terminal_bonuses"]),
        reward_mode=pol.get("reward_mode", "sparse"),
        quality_scalars=pol.get("quality_scalars"),
        load_preencoded=False,
        bucket_overrides=pol.get("bucket_overrides"),
        holdout_fraction=float(train_cfg["test_split_ratio"]),
        holdout_seed=train_cfg.get("seed"),
    )
    if not wrapper.has_holdout:
        raise RuntimeError(
            "No held-out split reconstructed (test_split_ratio="
            f"{train_cfg.get('test_split_ratio')!r}) — cannot run the G2 gate."
        )
    return wrapper, fps, camera_keys


# ──────────────────────────────────────────────────────────────────────────────
# Episode access
# ──────────────────────────────────────────────────────────────────────────────


def episode_action_matrix(wrapper, ep_id: int) -> torch.Tensor:
    """(T, A) raw actions of one episode, read from parquet — no video decode."""
    hf = wrapper.dataset.hf_dataset
    f = int(wrapper._ep_from[ep_id].item())
    t = int(wrapper._ep_to[ep_id].item())
    rows = [torch.as_tensor(hf[i]["action"], dtype=torch.float32) for i in range(f, t)]
    return torch.stack(rows)


def chunk_from_matrix(mat: torch.Tensor, t: int, h: int) -> torch.Tensor:
    """(h, A) chunk starting at t; end-of-episode padded by repeating the last
    frame (mirrors LeRobotDataset's clamped delta-timestamp padding)."""
    idx = np.minimum(np.arange(t, t + h), len(mat) - 1)
    return mat[idx]


def chunk_at_frac(mat: torch.Tensor, frac: float, h: int) -> torch.Tensor:
    t = int(round(float(frac) * (len(mat) - 1)))
    return chunk_from_matrix(mat, t, h)


def sample_frame_offsets(T: int, stride: int, max_frames: int) -> list[int]:
    """Every stride-th in-episode offset, always including the terminal frame."""
    offs = list(range(0, T, stride))
    if offs[-1] != T - 1:
        offs.append(T - 1)
    if max_frames and len(offs) > max_frames:
        keep = np.unique(np.linspace(0, len(offs) - 1, num=max_frames, dtype=int))
        offs = [offs[i] for i in keep]
    return offs


# ──────────────────────────────────────────────────────────────────────────────
# Candidate builder
# ──────────────────────────────────────────────────────────────────────────────


def build_candidates(
    true: torch.Tensor,
    frac: float,
    own_ep_id: int,
    success_ids: list[int],
    action_mats: dict[int, torch.Tensor],
    sigma_vec: np.ndarray,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    gen: torch.Generator,
    rng: random.Random,
    grip_dims: list[int] | None = None,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    """Return ((K, h, A) raw candidate chunks, ordered [(probe_name, n), ...]).

    K is constant across frames (12), so per-frame candidate tensors stack.
    grip_dims: gripper indices (single-arm [A-1]; bimanual YAM [6, 13]).
    """
    from lerobot.policies.act_simple.planning import _smooth_time

    L, A = true.shape
    gd = torch.as_tensor(grip_dims if grip_dims else [A - 1], dtype=torch.long)
    sig = torch.as_tensor(sigma_vec, dtype=torch.float32)
    groups: list[tuple[str, torch.Tensor]] = []

    # (a) planner-matched tubes. NO +/-1 clamp (radian joints!) — clamp to the
    # dataset's per-dim action envelope instead.
    for name, scale in TUBE_SPECS:
        noise = torch.randn(N_TUBE, L, A, generator=gen) * sig * scale
        noise = _smooth_time(noise, sigma=SMOOTH_SIGMA_T)
        noise[..., gd] = 0.0
        cand = true.unsqueeze(0) + noise
        cand = torch.max(torch.min(cand, a_max), a_min)
        groups.append((name, cand))

    # (b) swap: true chunk of a DIFFERENT held-out success episode at the same
    # episode fraction.
    src_ids = [e for e in success_ids if e != own_ep_id]
    if len(src_ids) >= N_SWAP:
        picks = rng.sample(src_ids, N_SWAP)
        swap = torch.stack([chunk_at_frac(action_mats[p], frac, L) for p in picks])
    elif src_ids:
        swap = torch.stack([chunk_at_frac(action_mats[src_ids[0]], frac, L)] * N_SWAP)
    else:
        # Degenerate single-episode fallback: own chunk half an episode away.
        swap = torch.stack(
            [chunk_at_frac(action_mats[own_ep_id], (frac + 0.5) % 1.0, L)] * N_SWAP
        )
    groups.append(("swap", swap))

    # (c) temporal-structure probes on the true chunk
    groups.append(("reversed", torch.flip(true, dims=[0]).unsqueeze(0)))
    perm = torch.randperm(L, generator=gen)
    groups.append(("shuffled", true[perm].unsqueeze(0)))

    # (d) gripper flip: range is [0, 1] on this dataset, so flip = 1 - g
    gf = true.clone()
    gf[..., gd] = 1.0 - gf[..., gd]
    groups.append(("grip_flip", gf.unsqueeze(0)))

    # (e) hold / freeze chunk — kinesthetic-relevant (action[t] ~= state[t])
    groups.append(("hold", true[0:1].repeat(L, 1).unsqueeze(0)))

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


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks (1-based), numpy only."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sv = x[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(x, y) -> float:
    """Spearman rank correlation; scipy if importable, inline numpy otherwise."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    try:
        from scipy.stats import spearmanr  # optional dependency

        return float(spearmanr(x, y)[0])
    except ImportError:
        pass
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


def _fmt(v, nd: int = 4) -> str:
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def render_summary(
    q_ckpt: Path,
    split_info: dict,
    probe_rows: list[dict],
    ep_rows: list[dict],
    verdict: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"# G2 offline gate — {q_ckpt}")
    lines.append("")
    lines.append(
        f"dataset: {split_info['repo_id']} | held-out episodes: "
        f"{split_info['n_success']} success + {split_info['n_failure']} failure "
        f"(test_split_ratio={split_info['test_split_ratio']}, seed={split_info['seed']})"
    )
    lines.append(f"held-out episode ids: {split_info['test_episode_ids']}")
    if split_info.get("hashseed_warning"):
        lines.append("")
        lines.append(f"WARNING: {split_info['hashseed_warning']}")
    lines.append("")

    lines.append("## Probe metrics by bucket")
    lines.append("")
    cols = ["bucket", "probe", "n_frames", "q_true_mean", "q_probe_mean", "gap", "rank_acc", "auroc"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in sorted(probe_rows, key=lambda r: (r["bucket"], r["probe"])):
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    lines.append("")

    lines.append("## Q vs time per held-out episode")
    lines.append("")
    cols = ["ep_id", "bucket", "T", "n_frames", "rho", "q_start", "q_end"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in sorted(ep_rows, key=lambda r: (r["bucket"], r["ep_id"])):
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    a, b, c = verdict["criteria"]["a"], verdict["criteria"]["b"], verdict["criteria"]["c"]
    lines.append(
        f"- (a) mean rank_acc over {list(VERDICT_PROBES)} on success frames = "
        f"{_fmt(a['value'])} (need >= {a['threshold']}) -> "
        f"{'PASS' if a['ok'] else 'FAIL'}"
    )
    lines.append(
        f"- (b) median success-episode Spearman rho = {_fmt(b['value'])} "
        f"(need > {b['threshold']}) -> {'PASS' if b['ok'] else 'FAIL'}"
    )
    lines.append(
        f"- (c) failure terminal q_true = {_fmt(c['value'])} vs median success "
        f"terminal q_true = {_fmt(c['threshold'])} (need <) -> "
        f"{'PASS' if c['ok'] else 'FAIL'}"
        + ("" if c["evaluable"] else "  [no failure episode in the held-out split]")
    )
    lines.append("")
    lines.append(f"**G2 VERDICT: {'PASS' if verdict['pass'] else 'FAIL'}**")
    lines.append("")
    return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--q_ckpt", type=Path, required=True,
                    help="Q checkpoint pretrained_model dir (contains train_config.json)")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT,
                    help="local LeRobot dataset root (the dataset directory itself)")
    ap.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    ap.add_argument("--stride", type=int, default=8,
                    help="score every Nth frame of each held-out episode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="default: outputs/gates/g2_<ckpt-step>")
    ap.add_argument("--frames-per-fwd", type=int, default=4,
                    help="dataset frames batched per Q forward (B = this x 13)")
    ap.add_argument("--max-frames-per-ep", type=int, default=0,
                    help="optional cap on scored frames per episode (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0, help="probe RNG seed (noise/swap/shuffle)")
    args = ap.parse_args()

    q_ckpt = args.q_ckpt.expanduser().resolve()
    step_name = q_ckpt.parent.name if q_ckpt.name == "pretrained_model" else q_ckpt.name
    out_dir = args.out_dir or Path("outputs/gates") / f"g2_{step_name}"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rng = random.Random(args.seed)

    hashseed_warning = None
    if os.environ.get("PYTHONHASHSEED") is None:
        hashseed_warning = (
            "PYTHONHASHSEED is unset: the holdout shuffle seed is "
            "hash((seed, repo_id, bucket)) with a per-process string-hash salt, "
            "so the reconstructed held-out episode IDs are not guaranteed to "
            "bit-match the training process's split (per-bucket counts do match). "
            "The IDs actually used are recorded below / in results.json."
        )
        log.warning(hashseed_warning)

    # ── Dataset + split ──────────────────────────────────────────────────
    train_cfg = load_train_cfg(q_ckpt)
    h = int(train_cfg["policy"]["h"])
    wrapper, fps, camera_keys = build_heldout_dataset(train_cfg, args.repo_id, args.dataset_root)

    test_eps = [int(e) for e in wrapper.test_episode_ids]
    bucket_of = {e: wrapper._bucket_by_global_ep[e] for e in test_eps}
    success_ids = sorted(e for e in test_eps if bucket_of[e] == SUCCESS_BUCKET)
    failure_ids = sorted(e for e in test_eps if bucket_of[e] == FAILURE_BUCKET)
    log.info("held-out episodes: %d success %s | %d failure %s",
             len(success_ids), success_ids, len(failure_ids), failure_ids)

    stats = wrapper.meta.stats["action"]
    sigma_vec = np.asarray(stats["std"], dtype=np.float64)
    a_min = torch.as_tensor(np.asarray(stats["min"], dtype=np.float32))
    a_max = torch.as_tensor(np.asarray(stats["max"], dtype=np.float32))
    log.info("action sigma = %s", np.round(sigma_vec, 4).tolist())

    # Per-episode raw action matrices for the swap probe (parquet only, no decode).
    action_mats = {e: episode_action_matrix(wrapper, e) for e in test_eps}

    action_names = list(wrapper.meta.features["action"].get("names") or [])
    grip_dims = [i for i, n in enumerate(action_names) if str(n).endswith("gripper.pos")]
    if grip_dims and grip_dims != [len(action_names) - 1]:
        log.info("gripper dims from action names: %s (of %d)", grip_dims, len(action_names))

    # ── Policy ───────────────────────────────────────────────────────────
    device = torch.device(args.device)
    policy, preprocessor = load_policy(q_ckpt, device)
    if int(policy.config.h) != h:
        raise RuntimeError(f"checkpoint h={policy.config.h} != train_config h={h}")
    log.info("loaded Q (h=%d, cams=%s) from %s", h, camera_keys, q_ckpt)

    from lerobot.policies.q_function.q_vis import _compute_chunk_q_values
    from lerobot.utils.constants import ACTION

    # records[bucket][probe] = {"true": per-frame q_true, "probe": per-variant q,
    #                           "win": per-variant 1[q_true > q_probe]}
    records: dict[str, dict[str, dict[str, list]]] = {}
    ep_rows: list[dict] = []
    frame_rows: list[dict] = []

    for ep_id in test_eps:
        bucket = bucket_of[ep_id]
        f0 = int(wrapper._ep_from[ep_id].item())
        T = int(wrapper._ep_to[ep_id].item()) - f0
        offs = sample_frame_offsets(T, args.stride, args.max_frames_per_ep)
        q_true_series: list[float] = []

        for cs in range(0, len(offs), args.frames_per_fwd):
            group = offs[cs : cs + args.frames_per_fwd]
            items = [wrapper[f0 + off] for off in group]      # video decode here
            cand_list, layouts = [], []
            for item, off in zip(items, group):
                frac = off / max(1, T - 1)
                cand, layout = build_candidates(
                    true=item[ACTION],
                    frac=frac,
                    own_ep_id=ep_id,
                    success_ids=success_ids,
                    action_mats=action_mats,
                    sigma_vec=sigma_vec,
                    a_min=a_min,
                    a_max=a_max,
                    gen=gen,
                    rng=rng,
                    grip_dims=grip_dims,
                )
                cand_list.append(cand)
                layouts.append(layout)
            candidates = torch.stack(cand_list)               # (ct, K, h, A)
            q = _compute_chunk_q_values(
                policy,
                preprocessor,
                items,
                num_perturb=0,
                perturb_std=0.0,
                device=device,
                candidates=candidates,
            )                                                  # (ct, 1 + K)
            for row_i, off in enumerate(group):
                q_true = float(q[row_i, 0])
                q_true_series.append(q_true)
                frame_row = {
                    "ep_id": ep_id,
                    "bucket": bucket,
                    "off": int(off),
                    "frac": off / max(1, T - 1),
                    "q_true": q_true,
                }
                col = 1
                for name, n in layouts[row_i]:
                    vals = q[row_i, col : col + n].numpy()
                    col += n
                    rec = records.setdefault(bucket, {}).setdefault(
                        name, {"true": [], "probe": [], "win": []}
                    )
                    rec["true"].append(q_true)
                    rec["probe"].extend(vals.tolist())
                    rec["win"].extend([float(q_true > v) for v in vals])
                    frame_row[name] = vals.tolist()
                frame_rows.append(frame_row)

        rho = spearman_rho(np.asarray(offs, float), np.asarray(q_true_series, float))
        ep_rows.append(
            {
                "ep_id": ep_id,
                "bucket": bucket,
                "T": T,
                "n_frames": len(offs),
                "rho": rho,
                "q_start": q_true_series[0],
                "q_end": q_true_series[-1],
                "offsets": [int(o) for o in offs],
                "q_true": q_true_series,
            }
        )
        log.info("ep %3d (%s, T=%d): %d frames scored, rho=%.3f, q %.3f -> %.3f",
                 ep_id, bucket, T, len(offs), rho, q_true_series[0], q_true_series[-1])

    # ── Per-(bucket, probe) metrics ──────────────────────────────────────
    probe_rows: list[dict] = []
    for bucket, probes in records.items():
        for name, rec in probes.items():
            pos, neg = np.asarray(rec["true"]), np.asarray(rec["probe"])
            probe_rows.append(
                {
                    "bucket": bucket,
                    "probe": name,
                    "n_frames": len(rec["true"]),
                    "q_true_mean": float(pos.mean()),
                    "q_probe_mean": float(neg.mean()),
                    "gap": float(pos.mean() - neg.mean()),
                    "rank_acc": float(np.mean(rec["win"])),
                    "auroc": auroc(pos, neg),
                }
            )

    # ── Verdict ──────────────────────────────────────────────────────────
    succ_probes = records.get(SUCCESS_BUCKET, {})
    accs = [float(np.mean(succ_probes[p]["win"])) for p in VERDICT_PROBES if p in succ_probes]
    a_val = float(np.mean(accs)) if accs else float("nan")
    a_ok = bool(np.isfinite(a_val) and a_val >= 0.8)

    succ_rhos = [r["rho"] for r in ep_rows
                 if r["bucket"] == SUCCESS_BUCKET and np.isfinite(r["rho"])]
    b_val = float(np.median(succ_rhos)) if succ_rhos else float("nan")
    b_ok = bool(np.isfinite(b_val) and b_val > 0.3)

    succ_terms = [r["q_end"] for r in ep_rows if r["bucket"] == SUCCESS_BUCKET]
    fail_terms = [r["q_end"] for r in ep_rows if r["bucket"] == FAILURE_BUCKET]
    med_succ_term = float(np.median(succ_terms)) if succ_terms else float("nan")
    c_evaluable = bool(fail_terms) and np.isfinite(med_succ_term)
    c_val = float(max(fail_terms)) if fail_terms else float("nan")
    c_ok = bool(c_evaluable and c_val < med_succ_term)

    verdict = {
        "criteria": {
            "a": {"desc": f"mean rank_acc over {list(VERDICT_PROBES)} on success frames",
                  "value": a_val, "threshold": 0.8, "ok": a_ok, "per_probe": dict(zip(
                      [p for p in VERDICT_PROBES if p in succ_probes], accs))},
            "b": {"desc": "median success-episode Spearman rho",
                  "value": b_val, "threshold": 0.3, "ok": b_ok},
            "c": {"desc": "max failure terminal q_true < median success terminal q_true",
                  "value": c_val, "threshold": med_succ_term, "ok": c_ok,
                  "evaluable": c_evaluable},
        },
        "pass": bool(a_ok and b_ok and c_ok),
    }

    split_info = {
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "test_split_ratio": train_cfg["test_split_ratio"],
        "seed": train_cfg.get("seed"),
        "h": h,
        "fps": fps,
        "camera_keys": camera_keys,
        "test_episode_ids": test_eps,
        "bucket_by_episode": bucket_of,
        "n_success": len(success_ids),
        "n_failure": len(failure_ids),
        "hashseed_warning": hashseed_warning,
    }

    summary = render_summary(q_ckpt, split_info, probe_rows, ep_rows, verdict)
    (out_dir / "summary.md").write_text(summary)
    results = {
        "q_ckpt": str(q_ckpt),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split": split_info,
        "probe_metrics": probe_rows,
        "episodes": ep_rows,
        "frames": frame_rows,
        "verdict": verdict,
    }
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    print(summary)
    log.info("wrote %s and %s", out_dir / "summary.md", out_dir / "results.json")


if __name__ == "__main__":
    main()
