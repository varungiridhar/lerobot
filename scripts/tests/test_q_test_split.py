#!/usr/bin/env python
"""Unit / static checks for the Q-function train-test-split refactor.

These tests don't need a GPU. They verify:
* The new config fields are present with the right defaults.
* ``_compute_test_metrics`` is importable from lerobot_train and has the
  expected kwargs.
* ``QValueLabelDataset`` accepts ``holdout_fraction`` / ``holdout_seed``,
  the wrapper still constructs cleanly with ``holdout_fraction=0.0``,
  and the public accessors behave as documented.
* Holdout construction is deterministic given matching
  ``(holdout_seed, holdout_fraction, repo_ids, bucket_overrides)`` and
  produces non-overlapping train/test frame index sets.

Run as ``python scripts/tests/test_q_test_split.py``. Each check is wrapped
in try/except; the script exits 0 iff all pass.
"""
from __future__ import annotations

import inspect
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_failures: list[str] = []


def _check(name: str, fn):
    try:
        fn()
        logger.info(f"PASS: {name}")
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"FAIL: {name}")
        _failures.append(f"{name}: {exc!r}")


# ---------------------------------------------------------------- fakes ----


class _FakeMeta:
    """Stand-in for ``LeRobotDatasetMetadata`` exposing only ``.episodes``."""

    def __init__(self, ep_from: list[int], ep_to: list[int]):
        # The wrapper indexes ``meta.episodes["dataset_from_index"]`` etc.
        self.episodes = {
            "dataset_from_index": list(ep_from),
            "dataset_to_index": list(ep_to),
        }


class _FakeSub:
    """Stand-in for a single ``LeRobotDataset`` inside a MultiLeRobotDataset.

    Implements the minimum surface the wrapper touches: ``num_frames``,
    ``repo_id``, ``meta.episodes``.
    """

    def __init__(self, repo_id: str, episode_lengths: list[int]):
        self.repo_id = repo_id
        starts = [0]
        for length in episode_lengths:
            starts.append(starts[-1] + length)
        self.meta = _FakeMeta(ep_from=starts[:-1], ep_to=starts[1:])
        self.num_frames = starts[-1]


class _FakeMulti:
    """Stand-in for ``MultiLeRobotDataset`` with just enough surface."""

    def __init__(self, subs: list[_FakeSub]):
        self._datasets = subs
        self.repo_ids = [s.repo_id for s in subs]
        self.num_frames = sum(s.num_frames for s in subs)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):  # not used by these checks
        raise NotImplementedError


_PATCHED = False


def _patch_multi_isinstance():
    """Make ``isinstance(_FakeMulti(...), MultiLeRobotDataset)`` evaluate True
    so the wrapper's MultiLeRobotDataset code paths fire on our fakes.

    Idempotent: re-running across tests does not stack tuples on tuples.
    """
    global _PATCHED
    if _PATCHED:
        return
    import lerobot.policies.q_function.q_value_labels as qvl

    qvl.MultiLeRobotDataset = (qvl.MultiLeRobotDataset, _FakeMulti)  # type: ignore[assignment]
    _PATCHED = True


# ------------------------------------------------------------------ tests --


def test_config_fields_present():
    import dataclasses

    from lerobot.configs.train import TrainPipelineConfig

    fields = {f.name: f for f in dataclasses.fields(TrainPipelineConfig)}
    for name, expected_default in (
        ("test_split_ratio", 0.0),
        ("test_freq", 0),
        ("test_n_batches", 1),
    ):
        assert name in fields, f"{name} missing from TrainPipelineConfig"
        default = fields[name].default
        assert default == expected_default, (
            f"{name}: expected default {expected_default!r}, got {default!r}"
        )


def test_compute_test_metrics_signature():
    from lerobot.scripts.lerobot_train import _compute_test_metrics

    sig = inspect.signature(_compute_test_metrics)
    expected = {"policy", "test_dl_iter", "n_batches", "preprocessor", "accelerator"}
    missing = expected - set(sig.parameters)
    assert not missing, f"_compute_test_metrics missing params: {missing}"


def test_qvld_accepts_holdout_kwargs_and_no_holdout_default():
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    sub = _FakeSub("repo/a", episode_lengths=[5, 7, 6, 8])
    _patch_multi_isinstance()
    multi = _FakeMulti([sub])

    # holdout disabled (default)
    wrapper = QValueLabelDataset(
        multi,
        h=4,
        step_reward=0.0,
        terminal_bonuses={"q5": 1.0},
        bucket_overrides={"repo/a": "q5"},
    )
    assert wrapper.train_frame_indices == [], (
        "train_frame_indices should be empty when holdout disabled"
    )
    assert wrapper.test_frame_indices == [], (
        "test_frame_indices should be empty when holdout disabled"
    )
    assert not wrapper.has_holdout, "has_holdout should be False with holdout_fraction=0.0"


def test_qvld_holdout_split_sane_and_deterministic():
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    _patch_multi_isinstance()

    def build():
        subs = [
            _FakeSub("repo/a", episode_lengths=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            _FakeSub("repo/b", episode_lengths=[5] * 20),
        ]
        multi = _FakeMulti(subs)
        return QValueLabelDataset(
            multi,
            h=4,
            step_reward=0.0,
            terminal_bonuses={"q5": 1.0, "play": 0.0},
            bucket_overrides={"repo/a": "q5", "repo/b": "play"},
            holdout_fraction=0.1,
            holdout_seed=1000,
        )

    w1 = build()
    w2 = build()
    assert w1.train_frame_indices == w2.train_frame_indices, "split not deterministic (train)"
    assert w1.test_frame_indices == w2.test_frame_indices, "split not deterministic (test)"
    assert w1.has_holdout, "expected has_holdout=True"

    train_set = set(w1.train_frame_indices)
    test_set = set(w1.test_frame_indices)
    assert not (train_set & test_set), "train and test frame indices overlap"
    total_frames = w1.dataset.num_frames
    assert len(train_set) + len(test_set) == total_frames, (
        f"train+test frame count != total_frames ({len(train_set)} + {len(test_set)} != {total_frames})"
    )

    # ~10% holdout per repo (10 eps -> 1 test; 20 eps -> 2 test).
    # 1 test ep in repo/a (any 1 of [3..12] frames) + 2 test eps in repo/b (5 frames each)
    # so test set size is between (3+5+5)=13 and (12+5+5)=22.
    assert 13 <= len(test_set) <= 22, f"unexpected test set size: {len(test_set)}"


def test_q_vis_public_api():
    """q_vis exposes the test-set visualization API; the BC-rollout API is gone."""
    import inspect

    from lerobot.policies.q_function import q_vis

    for name in ("log_q_test_visualizations", "compute_episode_q_values", "make_episode_video"):
        assert hasattr(q_vis, name), f"q_vis missing public symbol: {name}"
    for stale in ("log_q_rollout_visualizations", "rollout_with_q_logging"):
        assert not hasattr(q_vis, stale), f"stale BC-rollout symbol {stale} still in q_vis"

    # log_q_test_visualizations takes the dataset + Q preprocessor, no BC/env.
    params = set(inspect.signature(q_vis.log_q_test_visualizations).parameters)
    expected = {"policy", "dataset", "preprocessor", "step", "wandb_logger"}
    missing = expected - params
    assert not missing, f"log_q_test_visualizations missing params: {missing}"
    for banned in ("bc_policy", "envs", "eval_env"):
        assert banned not in params, f"log_q_test_visualizations should not take {banned}"


def test_lerobot_train_log_q_visualizations_signature():
    """_log_q_visualizations is the test-set form: (policy, dataset, preprocessor, ...)."""
    import inspect

    from lerobot.scripts.lerobot_train import _log_q_visualizations

    params = set(inspect.signature(_log_q_visualizations).parameters)
    expected = {"policy", "dataset", "preprocessor", "step", "wandb_logger", "device"}
    missing = expected - params
    assert not missing, f"_log_q_visualizations missing params: {missing}"
    assert "bc_policy" not in params, "_log_q_visualizations should no longer take bc_policy"


def test_q_eval_bc_policy_path_removed():
    """The BC-eval config field is gone now that env eval is removed."""
    import dataclasses

    from lerobot.configs.train import TrainPipelineConfig

    names = {f.name for f in dataclasses.fields(TrainPipelineConfig)}
    assert "q_eval_bc_policy_path" not in names, "q_eval_bc_policy_path should be removed"


def test_wandb_logger_accepts_test_mode():
    """WandBLogger.log_dict must accept mode='test' (used for test-split metrics).

    Regression guard: log_dict hard-codes an allowed-mode set; 'test' has to be
    in it or the training loop crashes at the first test_freq step when wandb
    is enabled.
    """
    from lerobot.rl.wandb_utils import WandBLogger

    class _FakeWandb:
        def __init__(self):
            self.logged = []

        def log(self, data, step=None):
            self.logged.append((data, step))

        def define_metric(self, *a, **k):
            pass

    # Bypass __init__ (it calls wandb.init); set only what log_dict reads.
    lg = object.__new__(WandBLogger)
    lg._wandb = _FakeWandb()
    lg._wandb_custom_step_key = None

    lg.log_dict({"loss": 4.0, "td_ce_loss": 4.0}, step=50, mode="test")
    assert lg._wandb.logged, "log_dict(mode='test') logged nothing"
    keys = [next(iter(d)) for d, _ in lg._wandb.logged]
    assert all(k.startswith("test/") for k in keys), f"keys not test/-prefixed: {keys}"

    # train + eval must still work.
    for m in ("train", "eval"):
        lg.log_dict({"x": 1.0}, step=1, mode=m)

    # An unknown mode must still raise.
    raised = False
    try:
        lg.log_dict({"x": 1.0}, step=1, mode="bogus")
    except ValueError:
        raised = True
    assert raised, "unknown mode should still raise ValueError"


def test_qvld_holdout_rejects_all_success():
    """Holdout requires bucket info, which the all_success path doesn't compute."""
    from lerobot.policies.q_function.q_value_labels import QValueLabelDataset

    sub = _FakeSub("repo/a", episode_lengths=[5, 7, 6])
    _patch_multi_isinstance()
    multi = _FakeMulti([sub])

    raised = False
    try:
        QValueLabelDataset(
            multi,
            h=4,
            step_reward=0.0,
            terminal_bonuses={"q5": 1.0},
            reward_mode="all_success",
            holdout_fraction=0.1,
        )
    except ValueError as e:
        assert "non-all_success" in str(e), f"unexpected error message: {e}"
        raised = True
    assert raised, "expected ValueError when combining holdout with all_success"


# ------------------------------------------------------------------ main --

if __name__ == "__main__":
    _check("config_fields_present", test_config_fields_present)
    _check("compute_test_metrics_signature", test_compute_test_metrics_signature)
    _check(
        "qvld_accepts_holdout_kwargs_and_no_holdout_default",
        test_qvld_accepts_holdout_kwargs_and_no_holdout_default,
    )
    _check(
        "qvld_holdout_split_sane_and_deterministic",
        test_qvld_holdout_split_sane_and_deterministic,
    )
    _check("wandb_logger_accepts_test_mode", test_wandb_logger_accepts_test_mode)
    _check("q_vis_public_api", test_q_vis_public_api)
    _check("lerobot_train_log_q_visualizations_signature",
           test_lerobot_train_log_q_visualizations_signature)
    _check("q_eval_bc_policy_path_removed", test_q_eval_bc_policy_path_removed)
    _check("qvld_holdout_rejects_all_success", test_qvld_holdout_rejects_all_success)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)
