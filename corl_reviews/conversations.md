> **Read `STATUS.md` first.** This file is the raw team plan, written informally as the discussion
> happened. Two things in it are superseded:
> - line 24 "no real world results — no time; reviewers don't care that much" → **Varun is running
>   real-world demos.** Also, the "real robot would be awesome" line is AzZT's (`r1`), not `r2`'s.
> - the reviewer tags `[r1,r2,r3,m]` are never defined here; the mapping is in `STATUS.md`
>   (r1=AzZT, r2=ZGT5, r3=svm5, m=AC rV3G).

# Plan of attack

Ignat Georgiev  [1:07 PM]
ok 5 days to go before rebuttal is due. The rebuttal comments are good IMO and we can get this in *cracks hands*

Overall two changes in direction:

Instead of doing temporal-smoothed MPPI we now do BC samples + 1-step MPPI
Instead of selling this as "BC + Q function" we should sell this as "Self-improvement BC"


Weaknesses

Missing baselines (most important) [r2,r3,m] - this must be done in online self-improvement setting!!
best of N sampling @Anant Khandelwal 
filtered SFT @Anant Khandelwal
one closely related baseline. We should compare to Steering Your Diffusion Policy with Latent Space Reinforcement Learning as it is the most different and I already implemented it to show that it doesn't work @Ignat 
we can also compare to Imitation Bootstrapped Reinforcement Learning @Anant Khandelwal can you grab this?

Breadth of self-improvement. [m] Extend the online self-improvement result beyond LIBERO-10 to the other suites / RoboTwin to show it is not specific to one benchmark. Already running this @Ignat
Limited novelty [r3,m] - this is mostly BS, some is justified. 1 paper has nothing to do with us. 1 paper came out 5 weeks ago. 1 paper doesn't even do multi-task, let alone SOTA performance. This is just paraphrasing and convincing people. Will call in the big guns @animesh 
More benchmarks e.g. MolmoSpace [r2]  - no time
The temporal MPPI section was a bit too light. Expand on what MPPI is [r1] - no mppi anymore
no real world results [r2] - no time; reviewers don't care that much apparently


Questions

Latency / real-time feasibility [r2,m].Report wall-clock cost per planning step given ~192 Q-evaluations, and discuss real-robot viability; a real-robot demonstration would substantially strengthen the paper. @Varun Giri 
Can it work with a very undertrained policy? [r2] - not entirely sure how to do this actually? we don't have a fastwam checkpoints :(
Bootstrapping target.Justify the one-step-shifted bootstrap in Eq. 2 versus bootstrapping with MPPI-selected actions, and address the mismatch between the value being learned (BC’s) and the policy actually deployed (the Q-guided planner).
Can it be applied beyond BC? [r1] - not important

## Some follow up questions by the rest of the members in the team

> Q: do we disclose this in the paper. I feel this would be too big of the shift for reviewers, especially since some of them have explicitly cited our use of temporal smoothing.

A: @Varun Giri yes we should disclose this in the paper. It's a small change and only one subsection. I think it will be fine

For "a very undertrained policy", it was decided that we should use diffusion poolicy to speed up iteration.


# Ignat on reproducing robotwin results

Most importantly, I've been looking into the reproducibility results of robotwin recently and found that I actually misrepresented the method. The thing that ended up working best is actually  low-diffusion-step BC samples + single-step MPPI. Called bc_diffusion_mppi in the tables below. I'd like to be faithful to the paper, drop the Temporal-smoothed MPPI and just do MPPI from candidates which should give us more stability. Let's commit to the 3 step diffusion as it gives us the best results so far

LIBERO-10
┌──────────────────────────────────────┬─────────┬─────────┬─────────────┬─────────┐
│          sampling / planner          │  succ   │  rate%  │ vs baseline │ ep_len* │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────┤
│ bc_diffusion_mppi, 3 steps           │ 190/200 │ 95.0    │ +5.0        │ 286.5   │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────┤
│ bc_diffusion_mppi, 5 steps           │ 183/200 │ 91.5    │ +1.5        │ 297.9   │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────┤
│ mppi (noise-perturb, std 0.3)        │ 181/200 │ 90.5    │ +0.5        │ 286.4   │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────┤
│ baseline (no Q, BC only)             │ 180/200 │ 90.0    │ —           │ 296.8   │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────┤
│ mppi + smoothing σ_t=8               │ 148/200 │ 74.0    │ −16.0       │ 335.9   │
├──────────────────────────────────────┼─────────┼─────────┼─────────────┼─────────┤
│ Q-guidance 0.01 / 0.02 / 0.05 / 0.10 │ —       │ pending │             │         │
└──────────────────────────────────────┴─────────┴─────────┴─────────────┴─────────┘

RoboTwin (47 tasks)
┌───────────────────────────────────┬───────┬─────────────┬────────┐
│        sampling / planner         │ succ% │ vs baseline │ ep_len │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 5 steps, Q 45k │ 84.0  │ +1.3        │ 278.7  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 2 steps, Q 15k │ 83.4  │ +0.6        │ 285.0  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 3 steps, Q 15k │ 83.4  │ +0.6        │ 276.5  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 5 steps, Q 15k │ 83.0  │ +0.2        │ 276.8  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ baseline (no Q, BC only)          │ 82.8  │ —           │ 278.5  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 5 steps, Q 30k │ 82.2  │ −0.5        │ 278.7  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 3 steps, Q 30k │ 81.8  │ −1.0        │ 289.7  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ + Q-guidance 0.02                 │ 80.1  │ −2.7        │ 294.6  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ bc_diffusion_mppi, 1 step, Q 15k  │ 78.9  │ −3.8        │ 300.0  │
├───────────────────────────────────┼───────┼─────────────┼────────┤
│ + Q-guidance 0.05                 │ 65.5  │ −17.2       │ 369.1  │
└───────────────────────────────────┴───────┴─────────────┴────────┘

Ignat Georgiev  [12:17 PM]
@Varun Giri we can commit to the 45k Q checkpoint for robotwin. Results are decent and more training doesn't hurt. Q functions can't really overfit (edited)
Robotwin results /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot/outputs/eval/robotwin_bcdiff_s3_ck045k
libero results /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot/outputs/eval/bench_bcdiff_s3_libero_10
(for libero just change the bench name and the results exist for the other ones too)
e.g. /storage/home/hcoda1/7/igeorgiev3/r-agarg35-0/lerobot/outputs/eval/bench_bcdiff_s3_libero_spatial
I updated the robotwin eval script to work with the best q function we have https://github.com/varungiridhar/lerobot/commit/609e70592bc3622d07fe27f20243812eba70d3f3 cc @Anant Khandelwal
Script to eval libero https://github.com/varungiridhar/lerobot/blob/609e70592bc3622d07fe27f20243812eba70d3f3/scripts/eval_fastwam_q_libero.sh
(all are on `planning branch)
