# Task prompt — Latency / real-time feasibility

Hand this to the downstream agent verbatim. Answers AC Key Issue #2 and reviewer ZGT5's Q2.

---

Task: profile planning latency for the Q-Planning CoRL rebuttal (ZGT5 Q2 + AC Key Issue #2).

READ FIRST: `corl_reviews/STATUS.md` — reviewer map, the method change since submission,
the resolved per-benchmark configs, control rates, replan budgets, checkpoint paths.
Do not trust `corl_reviews/qplanning_paper.pdf` for the current method.

CONFIG RULE: whatever the eval script sets wins; if a value is absent from the script,
fall back to the `config.json` of the checkpoint the eval points at. Both benchmarks
resolve to `num_inference_steps=10` for the BC policy and `num_diffusion_steps=3` for
candidate sampling — already worked out in STATUS.md, don't re-derive it.

SCOPE: a microbenchmark, not an eval campaign. Time ONE planning step (one action chunk)
repeatedly on a single fixed observation. No episodes, no success rates. Short study.

MEASURE — median + p95 over >=50 planning steps, batch=1, `torch.cuda.synchronize` around
every timed region, >=10 warm-up iterations discarded. Run the full set on BOTH LIBERO
and RoboTwin; the appendix table carries both.

  A. BC-only (`use_planning=false`) at denoising steps {3, 5, 10, 20}. 10 is the resolved
     default and the baseline ZGT5 asked for; the ladder also tests the paper's line-130
     claim that 5 steps halves latency.
  B. `bc_diffusion_mppi` — the shipping method — at N in {8, 16, 32, 64}, 3 diffusion steps.
     The deployed N differs per benchmark (LIBERO 64, RoboTwin 32); mark those rows.
     Also run RoboTwin at 5 diffusion steps — see the labelling discrepancy in STATUS.md.
  C. `bc_diffusion_argmax` at the deployed N — same cost as B, confirms aggregation is free.
  D. `mppi` + temporal smoothing, N=64 x n_iters=3 = the paper's 192 Q-evals. This is the
     number the reviewers asked about; it must be in the table.
  E. `mppi`, N=64 x n_iters=1, for the marginal cost of one MPPI iteration.

DECOMPOSE B and D into: BC sampling (diffusion) | obs encode (DINOv2+T5, once per step)
| batched Q decoder over N | normalization + aggregation overhead. The headline claim is
that the encoders are amortized and only the ~302M decoder scales with N — prove or
disprove it. Also check empirically whether `predict_n_action_chunks` batches the N
denoising trajectories or loops them; the answer changes how B scales with N.

HARDWARE: one GPU, named in the output — L40s as primary. See `corl_reviews/rules.md`
for SLURM.

REPORT: ms/planning-step for every condition on both benchmarks, plus each as a fraction
of its replan budget (LIBERO 10 steps @ 30 Hz = 333 ms; RoboTwin 24 steps @ 25 Hz =
960 ms). State plainly which configurations beat real time at the benchmark's own control
rate and which don't. If the shipping method is slower than the old 192-eval MPPI, say
so — the rebuttal needs the true number, not a favourable one. Deliver a ~150-word
rebuttal paragraph and an appendix-ready table covering both benchmarks.

Real-world latency is out of scope — a real-robot demo is being run separately by the
first author. Keep this on the simulated benchmarks.

Add instrumentation as a standalone script (`scripts/profile_planning.py`); don't leave
timing code in the planner hot path.
