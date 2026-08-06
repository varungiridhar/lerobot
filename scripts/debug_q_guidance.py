"""Sweep Q-gradient guidance scales on a single observation.

Loads FastWAM + the Q function once, then plans the same chunk at several
``q_guidance_scale`` values (and optionally several diffusion-step counts),
reporting for each:

  q_mean/q_max   Q over the N sampled candidates
  q_sel          Q of the chunk the planner actually returns
  spread         mean per-dim std across candidates — the "trajectory collapse"
                 metric; the unguided baseline sits around 0.02-0.04
  d(base)        RMS distance of the selected chunk from the unguided one, i.e.
                 how far guidance dragged the plan. Large values mean Q is being
                 exploited off-manifold rather than usefully re-ranked.

By default the observation is synthetic (random pixels), which validates the
mechanics — gradient exists, flows, and moves the samples — but says nothing
about task performance, since Q's output on noise is meaningless. Use a real
rollout observation via ``--obs-npz`` to judge whether the Q landscape is
actually informative.

    python scripts/debug_q_guidance.py --steps 1,3,5 --scales 0,0.05,0.1,0.3

Needs a GPU node: FastWAM is ~4 min to load and does not fit on the login node.
"""

import argparse
import time

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act_simple.planning import _score_candidates_fast
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.planning import FastWAMPlanner, plan_chunk_fastwam
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

FASTWAM_CKPT = "/storage/project/r-agarg35-0/shared/fastwam/hf_checkpoint_robotwin"
Q_ROOT = (
    "/storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/q_checkpoints_backup/"
    "qf_robotwin_ddp_20260526_220319"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fastwam-ckpt", default=FASTWAM_CKPT)
    p.add_argument("--q-ckpt", default=None, help="full path; overrides --q-step")
    p.add_argument("--q-step", default="015000", help="checkpoint step under Q_ROOT")
    p.add_argument("--scales", default="0,0.05,0.1,0.3,1.0")
    p.add_argument("--steps", default="3", help="comma-separated diffusion step counts")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--obs-npz", default=None, help="real observation (npz of batch keys)")
    p.add_argument("--task", default="adjust the bottle")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_ckpt = args.q_ckpt or f"{Q_ROOT}/{args.q_step}/pretrained_model"

    t0 = time.time()
    cfg = PreTrainedConfig.from_pretrained(args.fastwam_ckpt)
    cfg.device = str(dev)
    policy = FastWAMPolicy.from_pretrained(args.fastwam_ckpt, config=cfg).to(dev).eval()
    _pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.fastwam_ckpt,
        preprocessor_overrides={"device_processor": {"device": str(dev)}},
    )
    print(f"FastWAM loaded in {time.time() - t0:.0f}s")

    if args.obs_npz:
        import numpy as np

        raw = np.load(args.obs_npz, allow_pickle=True)
        batch = {k: torch.as_tensor(raw[k]).to(dev) for k in raw.files if k != "task"}
        batch["task"] = [str(raw["task"])] if "task" in raw.files else [args.task]
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        batch = {
            f"{OBS_IMAGES}.image": torch.rand(1, 3, 384, 320, generator=g).to(dev),
            f"{OBS_IMAGES}.cam_high": torch.rand(1, 3, 480, 640, generator=g).to(dev),
            f"{OBS_IMAGES}.cam_left_wrist": torch.rand(1, 3, 480, 640, generator=g).to(dev),
            f"{OBS_IMAGES}.cam_right_wrist": torch.rand(1, 3, 480, 640, generator=g).to(dev),
            OBS_STATE: torch.zeros(1, 14).to(dev),
            "task": [args.task],
        }

    pcfg = cfg.planning
    pcfg.q_checkpoint_path = q_ckpt
    pcfg.planner_type = "bc_diffusion_mppi"
    pcfg.n_samples = args.n_samples
    pcfg.n_elites = 8
    pcfg.temperature = 1.0

    t0 = time.time()
    planner = FastWAMPlanner.from_checkpoints(
        cfg=pcfg, bc_post=post, bc_chunk_size=int(cfg.chunk_size), device=dev
    )
    print(f"Q loaded in {time.time() - t0:.0f}s")

    img_feats = {k: batch[k] for k in planner.ctx.q_camera_keys}
    img_feats["task"] = batch["task"]
    h = planner.ctx.horizon
    action_dim = int(cfg.action_dim)

    def score(chunk: torch.Tensor) -> float:
        """Q of one (1, h, A) chunk, through the path the planner itself scores with."""
        single = planner.ctx.q_pre({ACTION: torch.zeros(1, h, action_dim, device=dev), **img_feats})
        obs_ctx = planner.ctx.q_policy.encode_obs_context(single)
        return float(_score_candidates_fast(chunk, obs_ctx, planner.ctx, img_feats))

    header = f"{'steps':>6} {'scale':>8} {'q_mean':>9} {'q_max':>9} {'q_sel':>9} {'spread':>9} {'d(base)':>9} {'sec':>6}"
    print("\n" + header)
    print("-" * len(header))
    for st in [int(x) for x in args.steps.split(",")]:
        pcfg.num_diffusion_steps = st
        base = None
        for s in [float(x) for x in args.scales.split(",")]:
            pcfg.q_guidance_scale = s
            torch.manual_seed(args.seed)  # identical initial noise across scales
            t0 = time.time()
            result, spread, _q_vals, cands = plan_chunk_fastwam(policy, batch, planner.ctx, pcfg)
            dt = time.time() - t0
            div = float(cands.float().std(dim=0).mean())
            if base is None:
                base = result.clone()
            dist = float((result - base).float().pow(2).mean().sqrt())
            print(
                f"{st:>6} {s:>8.3f} {spread[2]:>9.4f} {spread[1]:>9.4f} {score(result):>9.4f} "
                f"{div:>9.5f} {dist:>9.5f} {dt:>6.1f}"
            )
        print()


if __name__ == "__main__":
    main()
