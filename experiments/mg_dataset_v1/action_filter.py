"""Loader-time action filtering for the v1 dataset.

Background: investigate_action_noise.py showed that the `play` bucket (BC
diffusion rollouts) has 3–6× more high-frequency energy in actions than the
MG-warped buckets (q5, q3_termjitter). That's diffusion's per-step denoising
stochasticity leaking into the saved actions — invisible in mean step-delta,
but a structural difference in the action distribution that would bias Q
training.

This module applies a Savitzky–Golay filter (`window_length=5`, `polyorder=2`)
to actions of `play`-bucket trajectories *only*, on read. The filter is a
local quadratic LS smoother; with window=5 it averages over the current
sample and 2 on each side, which on a 20 Hz control loop kills content above
roughly 2 Hz while preserving slower task-driven motion.

Disk data stays raw so the filter can be ablated on/off without recollection.

Public API:
    DEFAULT_SAVGOL_KWARGS      → kwargs we picked
    savgol_filter_actions(a)   → smooth a (T, A) array, return same shape
    maybe_filter_actions(a, bucket, apply_to=("play",))
                               → no-op unless `bucket` is in `apply_to`
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import savgol_filter

# --- Hyperparameters ---------------------------------------------------------
# window_length=5 (must be odd, >= polyorder + 2): 5 samples = 250 ms at 20 Hz.
# polyorder=2 (quadratic local fit; preserves curvature of real motion).
# mode="nearest": pad the boundary by replicating the last/first value, which
# avoids the spurious endpoints that "interp" or "constant" would produce on
# action signals that don't go to zero at the edges.
DEFAULT_SAVGOL_KWARGS: dict = {
    "window_length": 5,
    "polyorder": 2,
    "mode": "nearest",
}

DEFAULT_FILTER_BUCKETS: tuple[str, ...] = ("play",)


def savgol_filter_actions(actions: np.ndarray, **savgol_kwargs) -> np.ndarray:
    """Apply Savitzky–Golay along the time axis of an (T, A) action array.

    Each action dim is smoothed independently. Trajectories shorter than
    `window_length` are returned unchanged (filter is undefined for them).
    """
    if actions.ndim != 2:
        raise ValueError(f"expected (T, action_dim), got shape {actions.shape}")
    kwargs = {**DEFAULT_SAVGOL_KWARGS, **savgol_kwargs}
    T = actions.shape[0]
    if T < kwargs["window_length"]:
        return actions
    return savgol_filter(actions, axis=0, **kwargs)


def maybe_filter_actions(
    actions: np.ndarray,
    bucket: str,
    apply_to: Iterable[str] = DEFAULT_FILTER_BUCKETS,
    **savgol_kwargs,
) -> np.ndarray:
    """Pass-through unless `bucket` is one we want to filter (default: 'play').

    Use this in the dataloader: read the `bucket` attr off the demo group,
    pass it in here, and you get raw or filtered actions transparently.
    """
    if bucket in set(apply_to):
        return savgol_filter_actions(actions, **savgol_kwargs)
    return actions
