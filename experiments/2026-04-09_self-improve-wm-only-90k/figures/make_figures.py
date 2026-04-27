"""Generate figures for the WM-only self-improvement experiment on the 50ep/90k checkpoint.

Run locally — never on SLURM.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"

# ----- load -----
phase_a = pd.read_csv(DATA / "phase_a_gbp_sweep.csv")
phase_b = pd.read_csv(DATA / "phase_b_wm_finetune.csv")

bc_baseline = float(phase_a.loc[phase_a["experiment_name"] == "E0-bc-baseline-wm90k", "bc_pc_success"].iloc[0])
gbp_best = float(phase_a.loc[phase_a["experiment_name"] == "G6-gbp-lr0.3-ni5-acc0.1", "plan_pc_success"].iloc[0])

# ----- Figure 1: Phase A — GBP sweep landscape -----
gbp_only = phase_a.dropna(subset=["plan_pc_success", "gbp_lr", "gbp_n_iters", "gbp_action_cost_coef"]).copy()
gbp_only["plan_pc_success"] = gbp_only["plan_pc_success"].astype(float)
gbp_only["gbp_lr"] = gbp_only["gbp_lr"].astype(float)
gbp_only["gbp_n_iters"] = gbp_only["gbp_n_iters"].astype(int)
gbp_only["gbp_action_cost_coef"] = gbp_only["gbp_action_cost_coef"].astype(float)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)

# (a) lr sweep with ni=10, acc=0.1
sub = gbp_only[(gbp_only["gbp_n_iters"] == 10) & (gbp_only["gbp_action_cost_coef"] == 0.1)].sort_values("gbp_lr")
axes[0].plot(sub["gbp_lr"], sub["plan_pc_success"], "o-", color="C0", lw=2, ms=8)
axes[0].axhline(bc_baseline, color="grey", ls="--", label=f"BC baseline ({bc_baseline:.1f}%)")
axes[0].set_xlabel("GBP learning rate")
axes[0].set_ylabel("Success rate (%)")
axes[0].set_title("lr sweep (n_iters=10, acc=0.1)")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="lower right")

# (b) n_iters sweep with lr=0.3, acc=0.1
sub = gbp_only[(gbp_only["gbp_lr"] == 0.3) & (gbp_only["gbp_action_cost_coef"] == 0.1)].sort_values("gbp_n_iters")
axes[1].plot(sub["gbp_n_iters"], sub["plan_pc_success"], "s-", color="C1", lw=2, ms=8)
axes[1].axhline(bc_baseline, color="grey", ls="--")
axes[1].set_xlabel("GBP n_iters")
axes[1].set_title("n_iters sweep (lr=0.3, acc=0.1)")
axes[1].grid(alpha=0.3)

# (c) action_cost_coef sweep with lr=0.3, ni=10
sub = gbp_only[(gbp_only["gbp_lr"] == 0.3) & (gbp_only["gbp_n_iters"] == 10)].sort_values("gbp_action_cost_coef")
axes[2].plot(sub["gbp_action_cost_coef"], sub["plan_pc_success"], "^-", color="C2", lw=2, ms=8)
axes[2].axhline(bc_baseline, color="grey", ls="--")
axes[2].set_xlabel("GBP action_cost_coef")
axes[2].set_title("action_cost sweep (lr=0.3, ni=10)")
axes[2].grid(alpha=0.3)

plt.suptitle("Phase A — GBP hyperparameter sweep (50ep/90k base, 250 eval episodes)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "phase_a_gbp_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIG / 'phase_a_gbp_sweep.png'}")

# ----- Figure 2: Phase B — heatmap of plan eval success -----
ep_values = [5, 10, 50, 100]
ft_values = [100, 500, 1000, 5000]

heatmap = np.zeros((len(ep_values), len(ft_values)))
for i, ep in enumerate(ep_values):
    for j, ft in enumerate(ft_values):
        row = phase_b[(phase_b["n_collect_episodes"] == ep) & (phase_b["finetune_steps"] == ft)]
        heatmap[i, j] = float(row["plan_pc_success"].iloc[0])

fig, ax = plt.subplots(figsize=(7.2, 5.0))
im = ax.imshow(heatmap, cmap="RdYlGn", vmin=2.0, vmax=6.0, aspect="auto")
ax.set_xticks(range(len(ft_values)))
ax.set_xticklabels(ft_values)
ax.set_yticks(range(len(ep_values)))
ax.set_yticklabels(ep_values)
ax.set_xlabel("finetune_steps")
ax.set_ylabel("n_collect_episodes")
for i in range(len(ep_values)):
    for j in range(len(ft_values)):
        ax.text(j, i, f"{heatmap[i, j]:.1f}", ha="center", va="center",
                color="black" if heatmap[i, j] > 4.0 else "white", fontsize=11, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, label="Plan eval success (%)")
ax.set_title(
    f"Phase B — WM-only finetune sweep (plan eval %)\n"
    f"GBP G6 baseline = {gbp_best:.1f}% | BC baseline = {bc_baseline:.1f}%",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(FIG / "phase_b_plan_eval_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIG / 'phase_b_plan_eval_heatmap.png'}")

# ----- Figure 3: Phase B — line plot of plan eval vs finetune steps, by n_collect_episodes -----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

# (a) Plan eval
ax = axes[0]
for ep in ep_values:
    sub = phase_b[phase_b["n_collect_episodes"] == ep].sort_values("finetune_steps")
    ax.plot(sub["finetune_steps"], sub["plan_pc_success"].astype(float), "o-",
            lw=1.8, ms=7, label=f"{ep} episodes")
ax.axhline(gbp_best, color="black", ls="--", lw=1.5, label=f"GBP G6 baseline ({gbp_best:.1f}%)")
ax.axhline(bc_baseline, color="grey", ls=":", lw=1.5, label=f"BC baseline ({bc_baseline:.1f}%)")
ax.set_xscale("log")
ax.set_xlabel("finetune_steps")
ax.set_ylabel("Plan eval success (%)")
ax.set_title("Plan eval after WM-only finetune")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)

# (b) BC eval (should be flat at 4.8%)
ax = axes[1]
for ep in ep_values:
    sub = phase_b[phase_b["n_collect_episodes"] == ep].sort_values("finetune_steps")
    ax.plot(sub["finetune_steps"], sub["bc_pc_success"].astype(float), "o-",
            lw=1.8, ms=7, label=f"{ep} episodes")
ax.axhline(bc_baseline, color="grey", ls=":", lw=1.5, label=f"BC baseline ({bc_baseline:.1f}%)")
ax.set_xscale("log")
ax.set_xlabel("finetune_steps")
ax.set_ylabel("BC eval success (%)")
ax.set_title("BC eval after WM-only finetune (should be invariant)")
ax.grid(alpha=0.3)
ax.set_ylim(0, 10)
ax.legend(loc="upper right", fontsize=9)

plt.suptitle("Phase B — WM-only finetune sweep on 50ep/90k base", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "phase_b_lines.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIG / 'phase_b_lines.png'}")

# ----- Figure 4: Phase B — avg max reward (more sensitive than success rate) -----
fig, ax = plt.subplots(figsize=(7.2, 4.8))
for ep in ep_values:
    sub = phase_b[phase_b["n_collect_episodes"] == ep].sort_values("finetune_steps")
    ax.plot(sub["finetune_steps"], sub["plan_avg_max_reward"].astype(float), "o-",
            lw=1.8, ms=7, label=f"{ep} episodes")
g6_reward = float(phase_a.loc[phase_a["experiment_name"] == "G6-gbp-lr0.3-ni5-acc0.1", "plan_avg_max_reward"].iloc[0])
ax.axhline(g6_reward, color="black", ls="--", lw=1.5, label=f"GBP G6 baseline ({g6_reward:.4f})")
ax.set_xscale("log")
ax.set_xlabel("finetune_steps")
ax.set_ylabel("Plan eval avg max reward")
ax.set_title("Phase B — Plan eval avg max reward (more sensitive than success)")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "phase_b_avg_max_reward.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {FIG / 'phase_b_avg_max_reward.png'}")

print("All figures generated.")
