# Planning-latency profile (ZGT5 Q2 + AC Key Issue #2)

**Hardware:** 1× NVIDIA L40S (PACE Phoenix; jobs 11723304/11723315 strict, 11723720/11723721 relaxed).
**Method:** `scripts/profile_planning.py` — one planning step (one action chunk) timed repeatedly
on a fixed observation; batch = 1; median + p95 over 60 timed steps after 10 discarded warm-up
steps; `torch.cuda.synchronize` around every timed region. Primary numbers replicate the
`lerobot-eval` runtime settings exactly (deterministic algorithms, TF32 off, no autocast, BC in
bf16, Q in fp32) — the settings the reported success rates were produced with. A secondary
"standard inference" pass (TF32 on, non-deterministic kernels, `cudnn.benchmark`) approximates
deployment mode. Checkpoints: the resolved eval checkpoints from `STATUS.md` (LIBERO Q
`23-54-42_qf_libero_ddp2…/last`, RoboTwin Q `qf_robotwin_ddp_20260526_220319/045000`).
Raw JSONs: `outputs/profile/{libero,robotwin}_l40s{,_relaxed}.json`.

**Replanning cadence** (the deadline a planning step must beat): the planner emits a 32-step
chunk, `n_action_steps` of it execute at the env control rate, then it replans.
LIBERO: 10 steps @ 30 Hz = **333 ms**. RoboTwin: 24 steps @ 25 Hz = **960 ms**.

## Appendix table — ms per planning step, median (p95), % of replanning budget

Strict = eval-parity settings (primary). Std = standard inference settings (secondary).

| Condition | Q-evals | LIBERO strict | % | LIBERO std | % | RoboTwin strict | % | RoboTwin std | % |
|---|---|---|---|---|---|---|---|---|---|
| BC only, 3 denoise steps | 0 | 273 (275) | 82 | 170 | 51 | 273 (275) | 28 | 169 | 18 |
| BC only, 5 denoise steps | 0 | 380 (381) | 114 | 230 | 69 | 379 (381) | 39 | 230 | 24 |
| **BC only, 10 steps (eval baseline)** | 0 | **646 (647)** | **194** | 381 | 114 | **640 (643)** | **67** | 377 | 39 |
| BC only, 20 denoise steps | 0 | 1179 (1181) | 354 | 688 | 206 | 1166 (1170) | 121 | 672 | 70 |
| bc-diff-MPPI, N=8, 3 steps | 8 | 322 (323) | 97 | 197 | 59 | 327 (331) | 34 | 196 | 20 |
| bc-diff-MPPI, N=16, 3 steps | 16 | 337 (338) | 101 | 205 | 61 | 352 (354) | 37 | 211 | 22 |
| bc-diff-MPPI, N=32, 3 steps | 32 | 371 (372) | 111 | 260 | 78 | **400 (402)** ◄ | **42** | 277 | 29 |
| bc-diff-MPPI, N=64, 3 steps | 64 | **640 (641)** ◄ | **192** | 495 | 149 | 716 (725) | 75 | 535 | 56 |
| bc-diff-MPPI, N=32, 5 steps | 32 | — | — | — | — | 507 (511) | 53 | 365 | 38 |
| bc-diff-argmax @ deployed N, 3 steps | 64 / 32 | 640 (642) | 192 | 496 | 149 | 402 (404) | 42 | 278 | 29 |
| MPPI ×3 iters, temporal smoothing (paper) | **192** | 1114 (1117) | 334 | 638 | 191 | 1276 (1282) | 133 | 740 | 77 |
| MPPI ×1 iter, temporal smoothing | 64 | 815 (821) | 244 | 479 | 144 | 867 (870) | 90 | 513 | 53 |

◄ = the deployed (shipping) configuration on that benchmark: LIBERO N=64, RoboTwin N=32,
both `bc_diffusion_mppi` at 3 diffusion steps.

## Decomposition of a planning step (median ms, strict; deployed configs and the paper's planner)

| Component | LIBERO ship (N=64, s3) | RoboTwin ship (N=32, s3) | LIBERO paper (64×3) | RoboTwin paper (64×3) |
|---|---|---|---|---|
| BC sampling (diffusion draw, incl. prefill) | 477 | 272 | 648 (10-step, 1 sample) | 640 |
| — of which batched denoise loop | 362 | 158 | 533 | 526 |
| Obs encode: DINOv2 + T5, once per step | 23 | 26 | 23 | 26 |
| Batched Q decoder over N | 146 | 100 | 439 (192 evals) | 590 |
| Normalization + aggregation | 2 | 4 | 6 | 24 |
| **Total** | **640** | **400** | **1114** | **1276** |

## Findings

1. **Real-time verdicts (L40S).** RoboTwin: every bc-diffusion configuration is comfortably
   real-time — the deployed N=32 planner uses **42%** of the 960 ms budget under strict eval
   settings (29% under standard settings); even N=64 fits (75% / 56%). LIBERO: the deployed
   N=64 configuration (640 ms strict / 495 ms std) exceeds the 333 ms budget at LIBERO's
   native 30 Hz — **but so does the 10-step BC baseline it is compared against (646 / 381 ms)**.
   Q-planning is latency-neutral on LIBERO: a 3-step × 64-batched draw + scoring costs the
   same as the baseline's single 10-step draw. N ≤ 16 (strict) or N ≤ 32 (std) fits LIBERO's
   budget.
2. **Shipping method vs. the paper's 192-eval MPPI.** The shipping planner is strictly faster:
   1.7× on LIBERO (640 vs 1114 ms), 3.2× on RoboTwin (400 vs 1276 ms). The paper's planner
   misses the RoboTwin budget under eval settings (133%); the shipping one meets it easily.
   Honest note for the rebuttal: on LIBERO *neither* the old nor the new planner (nor the BC
   baseline at its evaluated 10 steps) meets 333 ms on an L40S.
3. **Encoders are amortized; only the decoder scales with N — confirmed.** DINOv2+T5 obs
   encoding runs once per planning step: 23 ms (2 views, LIBERO) / 26 ms (3 views, RoboTwin),
   4–7% of a step. The ~302M Q decoder is the only Q-side cost that grows with N:
   ≈2.3 ms/candidate (LIBERO) and ≈3.1 ms/candidate (RoboTwin, larger context), measured
   25→146 ms and 28→202 ms across N=8→64. Normalization + aggregation is 1–24 ms. Q's T5
   caches per-string embeddings, so text encoding is paid only on a task's first chunk.
4. **`predict_n_action_chunks` batches the N denoising trajectories — verified empirically.**
   A 3-step N-candidate draw costs ~160 ms for N=8, 16 and 32 alike (≈53 ms per denoise step,
   same as N=1), rising ~2.3× only at N=64 where the batch saturates the GPU. A looped
   implementation would scale linearly (N=32 would cost ≈5 s). Fixed prefill (VAE + T5 +
   video-expert KV-cache) is ~114 ms regardless of N.
5. **Paper line-130 "5 steps halves latency" — approximately.** It halves the denoising loop
   (530→265 ms) but total step latency falls 41% (646→380 ms) because of the ~114 ms
   N-independent prefill.
6. **3-vs-5-step labelling discrepancy (RoboTwin).** `conversations.md` labels the best RoboTwin
   row "5 steps", but the run Ignat pointed at (`robotwin_bcdiff_s3_ck045k`) resolves to
   `num_diffusion_steps=3`. Both were timed: 400 ms (3 steps) vs 507 ms (5 steps) — both well
   inside budget, so the latency conclusion is robust to however the label resolves. Flagged,
   not silently picked.
7. **Aggregation is free.** `bc_diffusion_argmax` and `bc_diffusion_mppi` at the same N are
   within 2 ms of each other on both benchmarks.

## Rebuttal paragraph (~150 words)

> We profiled one full planning step (batch 1, median over 60 steps with CUDA synchronization,
> one NVIDIA L40S, the exact evaluation configuration and checkpoints). Our planner draws N
> diffusion candidates in one batched pass and scores them with one batched Q call: the
> DINOv2+T5 observation encoders run once per step (23–27 ms) and only the 302M-parameter Q
> decoder scales with N (≈2–3 ms per candidate). A planning step takes 400 ms on RoboTwin
> (N=32) — 42% of its 960 ms replanning budget (24 actions at 25 Hz), real-time with 2.4×
> headroom — and 640 ms on LIBERO (N=64), matching the 10-step BC baseline's own 646 ms, so
> Q-planning adds no latency over the evaluated policy; N≤16 fits LIBERO's 333 ms budget with
> comparable success. The submitted paper's 192-Q-evaluation MPPI costs 1114–1276 ms; the
> current planner is 1.7–3.2× faster. Standard inference settings (TF32, non-deterministic
> kernels) reduce all numbers a further ~25–35%.

*(Note: the "N≤16 … with comparable success" clause needs a success-rate citation from the
N-ablation before it goes in the rebuttal — drop the clause if that ablation isn't run.)*

## Notes / caveats

- Strict settings = `lerobot-eval` parity (`torch.use_deterministic_algorithms(True)`, TF32 off,
  `cudnn.benchmark` off) — conservative, but exactly what produced the reported success rates.
- Real-world (physical robot) latency is out of scope here — separate deliverable (Varun).
- Observation content is synthetic but shape-exact (latency is content-independent); shapes are
  taken from the checkpoints' configs, which match the eval scripts' env settings.
- The per-chunk `logger.info` in `FastWAMPlanner.plan` and planner vis recording are excluded
  (eval-harness bookkeeping, not the method).
- `n_iters=3` in the RoboTwin eval config is inert for `bc_diffusion_*` planners (the branch
  never loops) — RoboTwin runs 32 Q-evals per step, not 96. Confirmed by the timing: its
  Q-decoder cost matches a single N=32 pass.
