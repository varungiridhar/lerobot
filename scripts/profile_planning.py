#!/usr/bin/env python
"""Profile Q-planning latency for the CoRL rebuttal (ZGT5 Q2 + AC Key Issue #2).

Times ONE planning step (one action chunk) repeatedly on a single fixed
observation — no episodes, no success rates. The timed entry points are exactly
what ``lerobot-eval`` calls when the action queue is empty:

  * BC-only        : ``FastWAMPolicy.predict_action_chunk(batch)``
  * with planning  : ``FastWAMPlanner.plan(policy, batch)``

Runtime settings replicate ``lerobot_eval.eval_main`` (deterministic
algorithms, TF32 off, cudnn.benchmark off, no autocast) so the numbers describe
the configuration the benchmark results were produced with. The observation is
synthetic (latency is content-independent) but shape-exact: shared keys take
their shapes from the BC checkpoint's ``input_features`` (which match the eval
scripts' env settings), Q-only camera keys from the Q checkpoint's.

Conditions per benchmark (see corl_reviews/STATUS.md for the resolved configs):

  A. bc_only_s{3,5,10,20}         BC policy alone at various denoising steps
  B. bcdiff_mppi_n{8,16,32,64}_s3 the shipping planner (bc_diffusion_mppi);
                                  RoboTwin additionally at 5 diffusion steps
  C. bcdiff_argmax at deployed N  same cost as B modulo aggregation
  D. mppi_n64_it3 (temporal smoothing) = the paper's 192 Q-evaluations
  E. mppi_n64_it1                 marginal cost of one MPPI iteration

Each condition reports median + p95 over >= 50 timed planning steps (>= 10
warm-up steps discarded), batch=1, ``torch.cuda.synchronize`` around every
timed region.

A second, separately-timed pass per planner condition decomposes the step into
  bc_sampling (diffusion draw) | bc_denoise (denoising loop only, subset of
  bc_sampling) | obs_encode (Q's DINOv2+T5, once per step) | q_decoder
  (batched Q transformer over N) | norm_agg (residual: BC-unnorm/Q-renorm of
  candidates, softmax/top-K aggregation, bookkeeping)
by wrapping bound methods on the loaded instances from this script — the
planner/policy source is untouched. bc_denoise directly answers whether
``predict_n_action_chunks`` batches the N denoising trajectories: if it loops,
bc_denoise scales ~linearly with N; if batched, sublinearly.

Usage (single GPU):
  python scripts/profile_planning.py --benchmark libero \
      --bc-checkpoint /storage/project/r-agarg35-0/shared/awm/fastwam_checkpoint \
      --q-checkpoint  <libero Q pretrained_model dir> \
      --output outputs/profile/libero_l40s.json
"""

import argparse
import json
import logging
import os
import socket
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch

logger = logging.getLogger("profile_planning")

# Control rates and replan intervals of the simulated benchmarks
# (corl_reviews/STATUS.md; n_action_steps also lives in the BC checkpoint config).
BENCHMARKS = {
    "libero": {
        "control_rate_hz": 30.0,
        "task": "put both the alfredo sauce and the cream cheese box in the basket",
        "deployed_n": 64,
        "n_elites": 16,
    },
    "robotwin": {
        "control_rate_hz": 25.0,
        "task": "use the hammer to beat the block on the table once",
        "deployed_n": 32,
        "n_elites": 8,
    },
}


@dataclass
class Condition:
    name: str
    mode: str                       # "bc_only" | "planner"
    bc_steps: int = 10              # policy.config.num_inference_steps
    planner_type: str = ""
    n_samples: int = 0
    n_iters: int = 1
    n_elites: int = 0
    num_diffusion_steps: int | None = None
    noise_std: float = 0.3
    noise_smooth_sigma_t: float | None = None
    temperature: float = 1.0
    deployed: bool = False
    note: str = ""


def build_conditions(benchmark: str) -> list[Condition]:
    b = BENCHMARKS[benchmark]
    n_dep, k_dep = b["deployed_n"], b["n_elites"]
    conds: list[Condition] = []

    # A. BC-only ladder. 10 is the resolved eval default on both benchmarks.
    for s in (3, 5, 10, 20):
        conds.append(
            Condition(
                name=f"bc_only_s{s}", mode="bc_only", bc_steps=s,
                deployed=(s == 10), note="BC baseline (resolved default)" if s == 10 else "",
            )
        )

    # B. Shipping planner: bc_diffusion_mppi at 3 diffusion steps, N ladder.
    for n in (8, 16, 32, 64):
        conds.append(
            Condition(
                name=f"bcdiff_mppi_n{n}_s3", mode="planner",
                planner_type="bc_diffusion_mppi", n_samples=n, n_elites=k_dep,
                num_diffusion_steps=3, deployed=(n == n_dep),
                note="shipping config" if n == n_dep else "",
            )
        )
    if benchmark == "robotwin":
        # conversations.md labels the best RoboTwin row "5 steps" but the run
        # dir Ignat pointed at resolves to 3 — time both, flag the discrepancy.
        conds.append(
            Condition(
                name=f"bcdiff_mppi_n{n_dep}_s5", mode="planner",
                planner_type="bc_diffusion_mppi", n_samples=n_dep, n_elites=k_dep,
                num_diffusion_steps=5,
                note="secondary: 5-step label in conversations.md",
            )
        )

    # C. Argmax aggregation at the deployed N — confirms aggregation is free.
    conds.append(
        Condition(
            name=f"bcdiff_argmax_n{n_dep}_s3", mode="planner",
            planner_type="bc_diffusion_argmax", n_samples=n_dep, n_elites=k_dep,
            num_diffusion_steps=3,
        )
    )

    # D. The submitted paper's planner: temporal-smoothed MPPI, 64 x 3 = 192
    # Q-evals — the number the AC and ZGT5 are asking about.
    conds.append(
        Condition(
            name="mppi_n64_it3_smooth", mode="planner",
            planner_type="mppi", n_samples=64, n_iters=3,
            noise_std=0.3, noise_smooth_sigma_t=2.0,
            note="paper's 192 Q-evals",
        )
    )
    # E. Same with one iteration — marginal cost of one MPPI iteration.
    conds.append(
        Condition(
            name="mppi_n64_it1_smooth", mode="planner",
            planner_type="mppi", n_samples=64, n_iters=1,
            noise_std=0.3, noise_smooth_sigma_t=2.0,
        )
    )
    return conds


# ── Synthetic observation ────────────────────────────────────────────────────

def _visual_chw(shape: list[int]) -> tuple[int, int, int]:
    """Return (C, H, W) from a VISUAL feature shape stored as CHW or HWC."""
    if len(shape) != 3:
        raise ValueError(f"Unexpected visual shape {shape}")
    if shape[0] == 3:
        return shape[0], shape[1], shape[2]
    if shape[2] == 3:
        return shape[2], shape[0], shape[1]
    raise ValueError(f"Cannot infer layout of visual shape {shape}")


def build_observation(bc_cfg, q_cfg, task: str, generator: torch.Generator) -> dict:
    """Synthetic post-env-processor observation batch (batch=1, CPU, float [0,1]).

    Shared keys (BC inputs) take shapes from the BC checkpoint config — these
    match what the eval scripts configure the env to emit. Q-only camera keys
    (e.g. RoboTwin's three raw cameras) take shapes from the Q checkpoint config.
    """
    obs: dict = {}
    for key, ft in bc_cfg.input_features.items():
        if ft.type.name == "VISUAL":
            c, h, w = _visual_chw(list(ft.shape))
            obs[key] = torch.rand(1, c, h, w, generator=generator)
        elif ft.type.name == "STATE":
            obs[key] = torch.zeros(1, *ft.shape)
    for key in q_cfg.camera_keys:
        if key in obs:
            continue
        ft = q_cfg.input_features[key]
        c, h, w = _visual_chw(list(ft.shape))
        obs[key] = torch.rand(1, c, h, w, generator=generator)
    obs["task"] = [task]
    return obs


# ── Timing helpers ───────────────────────────────────────────────────────────

def timed_pass(fn, warmup: int, iters: int) -> list[float]:
    """Wall-clock ms per call, cuda.synchronize around every timed region."""
    times: list[float] = []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3
        if i >= warmup:
            times.append(dt_ms)
    return times


def stats(times_ms: list[float]) -> dict:
    a = np.asarray(times_ms)
    return {
        "iters": len(times_ms),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


class ComponentTimer:
    """Wrap bound methods on live instances to accumulate GPU-synced time.

    Instrumentation lives entirely in this script; the planner/policy modules
    are untouched. Nested wraps (bc_denoise inside bc_sampling) are fine — the
    inner syncs serialize work that is already sequential.
    """

    def __init__(self):
        self.buckets: dict[str, float] = defaultdict(float)
        self._patched: list[tuple[object, str, object]] = []

    def wrap(self, obj, attr: str, key: str) -> None:
        orig = getattr(obj, attr)

        def wrapped(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            self.buckets[key] += (time.perf_counter() - t0) * 1e3
            return out

        # Instance-attribute patch shadows the class method for this object only.
        object.__setattr__(obj, attr, wrapped)
        self._patched.append((obj, attr, orig))

    def restore(self) -> None:
        for obj, attr, _orig in reversed(self._patched):
            object.__delattr__(obj, attr)
        self._patched.clear()

    def snapshot_and_reset(self) -> dict[str, float]:
        snap = dict(self.buckets)
        self.buckets.clear()
        return snap


def decomposition_pass(fn, policy, q_policy, warmup: int, iters: int) -> dict:
    """Per-component median ms across iters. norm_agg = total - top-level parts."""
    timer = ComponentTimer()
    timer.wrap(policy, "predict_action_chunk", "bc_sampling")
    timer.wrap(policy, "predict_n_action_chunks", "bc_sampling")
    timer.wrap(policy, "predict_n_action_chunks_partial", "bc_sampling")
    timer.wrap(policy.model, "_predict_action_noise_with_cache", "bc_denoise")
    timer.wrap(q_policy, "encode_obs_context", "obs_encode")
    timer.wrap(q_policy.q_online, "forward_with_context", "q_decoder")

    per_iter: list[dict[str, float]] = []
    try:
        for i in range(warmup + iters):
            timer.buckets.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            total = (time.perf_counter() - t0) * 1e3
            if i >= warmup:
                snap = timer.snapshot_and_reset()
                snap["total"] = total
                per_iter.append(snap)
    finally:
        timer.restore()

    keys = sorted({k for d in per_iter for k in d})
    out = {k: float(np.median([d.get(k, 0.0) for d in per_iter])) for k in keys}
    # bc_denoise is a subset of bc_sampling; exclude it from the residual.
    top_level = ("bc_sampling", "obs_encode", "q_decoder")
    out["norm_agg"] = out["total"] - sum(out.get(k, 0.0) for k in top_level)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    p.add_argument("--bc-checkpoint", required=True)
    p.add_argument("--q-checkpoint", required=True)
    p.add_argument("--output", required=True, help="Path for the JSON results file")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--decomp-iters", type=int, default=30)
    p.add_argument("--decomp-warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conditions", default="", help="Comma-separated subset of condition names")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--relaxed",
        action="store_true",
        help="Standard inference settings (TF32 on, non-deterministic algorithms, "
        "cudnn.benchmark) instead of the strict lerobot-eval parity settings. "
        "Approximates deployment-mode latency; success rates were NOT produced this way.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Keep per-chunk planner INFO logs out of the timed loop's stdout.
    logging.getLogger("lerobot.policies.fastwam.planning").setLevel(logging.WARNING)

    if args.relaxed:
        # Deployment-style inference settings — NOT how eval success rates were
        # produced; use only as the secondary "standard inference" latency row.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
    else:
        # Replicate lerobot_eval.eval_main runtime settings exactly.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.fastwam.planning import FastWAMPlanner
    from lerobot.policies.act_simple.planning import PlanningConfig
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)
    # eval_main wraps the whole rollout in torch.no_grad() (no autocast:
    # use_amp defaults to False for these checkpoints); replicate globally.
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    bench = BENCHMARKS[args.benchmark]

    logger.info("Loading BC policy from %s", args.bc_checkpoint)
    bc_cfg = PreTrainedConfig.from_pretrained(args.bc_checkpoint)
    bc_cfg.pretrained_path = args.bc_checkpoint
    bc_cfg.device = str(device)
    policy_cls = get_policy_class(bc_cfg.type)
    policy = policy_cls.from_pretrained(pretrained_name_or_path=args.bc_checkpoint, config=bc_cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=bc_cfg,
        pretrained_path=args.bc_checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    logger.info("Loading Q-function from %s", args.q_checkpoint)
    base_planning_cfg = PlanningConfig(q_checkpoint_path=args.q_checkpoint)
    base_planner = FastWAMPlanner.from_checkpoints(
        cfg=base_planning_cfg,
        bc_post=postprocessor,
        bc_chunk_size=int(bc_cfg.chunk_size),
        device=device,
    )
    ctx = base_planner.ctx
    q_policy = ctx.q_policy

    gen = torch.Generator().manual_seed(args.seed)
    raw_obs = build_observation(bc_cfg, q_policy.config, bench["task"], gen)
    batch = preprocessor(raw_obs)
    logger.info(
        "Observation batch: %s",
        {k: tuple(v.shape) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
    )

    conditions = build_conditions(args.benchmark)
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        unknown = wanted - {c.name for c in conditions}
        if unknown:
            raise SystemExit(f"Unknown condition names: {sorted(unknown)}")
        conditions = [c for c in conditions if c.name in wanted]

    n_action_steps = int(bc_cfg.n_action_steps)
    budget_ms = n_action_steps / bench["control_rate_hz"] * 1e3

    results: list[dict] = []
    for cond in conditions:
        policy.config.num_inference_steps = cond.bc_steps
        if cond.mode == "bc_only":
            fn = lambda: policy.predict_action_chunk(batch)  # noqa: E731
        else:
            pcfg = PlanningConfig(
                q_checkpoint_path=args.q_checkpoint,
                planner_type=cond.planner_type,
                n_samples=cond.n_samples,
                n_iters=cond.n_iters,
                n_elites=cond.n_elites,
                noise_std=cond.noise_std,
                noise_smooth_sigma_t=cond.noise_smooth_sigma_t,
                temperature=cond.temperature,
                num_diffusion_steps=cond.num_diffusion_steps,
            )
            planner = FastWAMPlanner(cfg=pcfg, ctx=ctx, generator=None)
            fn = lambda: planner.plan(policy, batch)  # noqa: B023,E731

        logger.info("[%s] timing pass (%d warmup + %d iters)", cond.name, args.warmup, args.iters)
        e2e = stats(timed_pass(fn, args.warmup, args.iters))
        entry = {
            "condition": asdict(cond),
            "e2e": e2e,
            "fraction_of_budget": e2e["median_ms"] / budget_ms,
        }
        if cond.mode == "planner":
            logger.info("[%s] decomposition pass", cond.name)
            entry["components_ms"] = decomposition_pass(
                fn, policy, q_policy, args.decomp_warmup, args.decomp_iters
            )
        results.append(entry)
        logger.info(
            "[%s] median %.1f ms  p95 %.1f ms  (%.0f%% of %d ms budget)",
            cond.name, e2e["median_ms"], e2e["p95_ms"],
            100 * entry["fraction_of_budget"], budget_ms,
        )

    out = {
        "benchmark": args.benchmark,
        "gpu": torch.cuda.get_device_name(device),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bc_checkpoint": args.bc_checkpoint,
        "q_checkpoint": args.q_checkpoint,
        "settings": {
            "deterministic_algorithms": not args.relaxed,
            "allow_tf32": args.relaxed,
            "cudnn_benchmark": args.relaxed,
            "autocast": False,
            "seed": args.seed,
            "note": (
                "standard inference settings (deployment-style)"
                if args.relaxed
                else "matches lerobot_eval.eval_main runtime settings"
            ),
        },
        "control_rate_hz": bench["control_rate_hz"],
        "n_action_steps": n_action_steps,
        "replan_budget_ms": budget_ms,
        "chunk_size": int(bc_cfg.chunk_size),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output)

    # Human-readable summary.
    print(f"\n=== {args.benchmark} on {out['gpu']} — replan budget {budget_ms:.0f} ms ===")
    print(f"{'condition':<28} {'median':>9} {'p95':>9} {'% budget':>9}  components (median ms)")
    for r in results:
        c, e = r["condition"], r["e2e"]
        comp = r.get("components_ms", {})
        comp_str = "  ".join(
            f"{k}={v:.0f}" for k, v in comp.items() if k != "total"
        )
        print(
            f"{c['name']:<28} {e['median_ms']:>7.1f}ms {e['p95_ms']:>7.1f}ms "
            f"{100 * r['fraction_of_budget']:>8.0f}%  {comp_str}"
        )


if __name__ == "__main__":
    main()
