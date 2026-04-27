"""Figure generators for the e2e self-improvement experiment.

Run locally (not on SLURM). Outputs go to ./figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).parent
DATA = EXP_DIR / "data"
FIG = EXP_DIR / "figures"
FIG.mkdir(exist_ok=True)

# Baselines (from prior 2026-04-09_self-improve-wm-only-90k experiment, same base)
BC_BASELINE = 4.8
G6_BASELINE = 5.2
BC_REW_BASELINE = 0.3602
G6_REW_BASELINE = 0.3496


def _mask_crash(df: pd.DataFrame) -> pd.DataFrame:
    # Replace crashed cells' metrics with NaN for heatmap display
    df = df.copy()
    crashed = df["status"] == "crash"
    for col in ["plan_pc_success", "plan_avg_max_reward"]:
        df.loc[crashed, col] = np.nan
    return df


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot(index="n_collect", columns="ft_steps", values=value_col).sort_index(
        ascending=False
    )


def _annotate_heatmap(ax, pivot, fmt="{:.1f}", cmap_min=None, cmap_max=None):
    for i, nc in enumerate(pivot.index):
        for j, ft in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, "OOM", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
            else:
                # Pick text color based on background
                color = "white" if (cmap_min is not None and cmap_max is not None
                                     and val < (cmap_min + cmap_max) / 2) else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color=color, fontsize=10)


def phase_a_heatmap():
    df = pd.read_csv(DATA / "phase_a.csv")
    df = _mask_crash(df)

    bc_piv = _pivot(df, "bc_pc_success")
    plan_piv = _pivot(df, "plan_pc_success")

    vmin = 0.0
    vmax = max(bc_piv.max().max(), plan_piv.max().max(), BC_BASELINE, G6_BASELINE) + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, piv, title, baseline, baseline_label in [
        (axes[0], bc_piv, f"BC eval success (%)\nbaseline {BC_BASELINE}%", BC_BASELINE, "BC=4.8%"),
        (axes[1], plan_piv, f"GBP-G6 plan eval success (%)\nbaseline {G6_BASELINE}%", G6_BASELINE, "G6=5.2%"),
    ]:
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([str(r) for r in piv.index])
        ax.set_xlabel("finetune_steps")
        ax.set_ylabel("n_collect_episodes")
        ax.set_title(title)
        _annotate_heatmap(ax, piv, cmap_min=vmin, cmap_max=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.axhline(baseline, color="black", linewidth=2, linestyle="--")

    fig.suptitle("Phase A — e2e single-iteration sweep (50ep/90k base)\n"
                 "Every cell underperforms the unfinetuned BC / G6 baseline",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "phase_a_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def phase_a_lines():
    df = pd.read_csv(DATA / "phase_a.csv")
    df = df[df["status"] == "keep"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, metric, baseline, baseline_label, ylim in [
        (axes[0], "bc_pc_success", BC_BASELINE, "BC baseline (4.8%)", (0, 7)),
        (axes[1], "plan_pc_success", G6_BASELINE, "G6 baseline (5.2%)", (0, 7)),
    ]:
        for n_collect, sub in df.groupby("n_collect"):
            sub = sub.sort_values("ft_steps")
            ax.plot(sub["ft_steps"], sub[metric], "o-", label=f"n_collect={n_collect}")
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1,
                   label=baseline_label)
        ax.set_xscale("log")
        ax.set_xlabel("finetune_steps (log)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_title(metric)

    fig.suptitle("Phase A — e2e single-iteration: BC and plan success vs ft_steps",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "phase_a_lines.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def phase_ab_combined_heatmap():
    """Combine Phase A + Phase B into a single (n_collect, ft_steps) grid."""
    a = pd.read_csv(DATA / "phase_a.csv")
    b_path = DATA / "phase_b.csv"
    if not b_path.exists():
        print("Phase B CSV not yet available — skipping combined heatmap.")
        return
    b = pd.read_csv(b_path)
    # Both files have the same schema except b lacks n_iters (it's always 1 here)
    df = pd.concat([a[a["status"] == "keep"], b[b["status"] == "keep"]], ignore_index=True)

    bc_piv = df.pivot_table(index="n_collect", columns="ft_steps",
                            values="bc_pc_success", aggfunc="mean").sort_index(ascending=False)
    plan_piv = df.pivot_table(index="n_collect", columns="ft_steps",
                              values="plan_pc_success", aggfunc="mean").sort_index(ascending=False)

    vmin = 0.0
    vmax = max(bc_piv.max().max(), plan_piv.max().max(), BC_BASELINE, G6_BASELINE) + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, piv, title in [
        (axes[0], bc_piv, f"BC eval success (%) — baseline {BC_BASELINE}%"),
        (axes[1], plan_piv, f"GBP-G6 plan eval (%) — baseline {G6_BASELINE}%"),
    ]:
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([str(r) for r in piv.index])
        ax.set_xlabel("finetune_steps")
        ax.set_ylabel("n_collect_episodes")
        ax.set_title(title)
        _annotate_heatmap(ax, piv, cmap_min=vmin, cmap_max=vmax)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Phase A + B combined — single-iter e2e sweep (50ep/90k base)", fontsize=11)
    fig.tight_layout()
    out = FIG / "phase_ab_combined_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def phase_c_figure():
    c_path = DATA / "phase_c.csv"
    if not c_path.exists():
        print("Phase C CSV not yet available — skipping.")
        return
    df = pd.read_csv(c_path)
    df = df[df["status"] == "keep"].copy()
    if df.empty:
        return

    # Group by (n_collect, ft_steps) → line of success vs n_iters
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, baseline, title in [
        (axes[0], "bc_pc_success", BC_BASELINE, "BC eval vs n_iters"),
        (axes[1], "plan_pc_success", G6_BASELINE, "GBP-G6 plan eval vs n_iters"),
    ]:
        for (nc, ft), sub in df.groupby(["n_collect", "ft_steps"]):
            sub = sub.sort_values("n_iters")
            ax.plot(sub["n_iters"], sub[metric], "o-", label=f"n={nc}, ft={ft}")
        ax.axhline(baseline, color="black", linestyle="--", linewidth=1,
                   label=f"baseline {baseline}%")
        ax.set_xlabel("n_iters")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(0, max(7, df[metric].max() + 1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Phase C — multi-iteration sweep", fontsize=11)
    fig.tight_layout()
    out = FIG / "phase_c_lines.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    phase_a_heatmap()
    phase_a_lines()
    phase_ab_combined_heatmap()
    phase_c_figure()
