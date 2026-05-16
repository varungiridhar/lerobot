"""GPU-friendly image preprocessing + LeRobot processor pipeline for the Q-function.

* ``DINOv2ImagePreprocessor`` is an in-model preprocessor used only when
  ``vision_backbone == "dinov2"`` (the live path). It resizes + ImageNet-
  normalizes batched images on-device for DINOv2.
* ``make_q_function_pre_post_processors`` is the standard LeRobot factory
  function that ``policies.factory.make_pre_post_processors`` discovers via
  introspection. Mirrors the ACT processor: normalize action (and any state
  in the batch), batch+device steps. Visual keys are present in
  ``input_features`` but the cached path emits them under ``_preencoded``
  suffixes so the normalizer's "skip missing keys" logic leaves them alone.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.q_function.configuration_q_function import QFunctionConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# Keys produced by QValueLabelDataset that the standard ``batch_to_transition``
# would drop (since they don't start with ``observation.`` and aren't in the
# canonical complementary-data set). We smuggle them through complementary_data
# so the symmetric ``transition_to_batch`` round-trip preserves them.
_Q_KEYS = (
    "q_reward_chunk_first",
    "q_reward_pad_first",
    "q_bootstrap_valid",
    "q_bucket_index",
    "task",   # language instruction string; must survive batch↔transition without normalisation
)


def _q_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    transition = batch_to_transition(batch)
    extras = {k: batch[k] for k in _Q_KEYS if k in batch}
    if extras:
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})
        comp.update(extras)
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
    return transition


def _q_transition_to_batch(transition: EnvTransition) -> dict[str, Any]:
    # ``transition_to_batch`` already merges complementary_data into the batch,
    # so the q_* keys come back automatically.
    return transition_to_batch(transition)


# Standard ImageNet stats, as used by DINOv2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2ImagePreprocessor:
    """Resize + ImageNet-normalise a batch of images for DINOv2.

    Not an ``nn.Module``: it holds no learnable parameters, only lazily-created
    buffers for mean/std (moved onto the correct device on first call).
    """

    def __init__(self, resize_h: int, resize_w: int):
        self.resize_h = int(resize_h)
        self.resize_w = int(resize_w)
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    def _lazy_buffers(self, ref: Tensor) -> tuple[Tensor, Tensor]:
        if self._mean is None or self._mean.device != ref.device or self._mean.dtype != ref.dtype:
            self._mean = torch.tensor(_IMAGENET_MEAN, device=ref.device, dtype=ref.dtype).view(1, 3, 1, 1)
            self._std = torch.tensor(_IMAGENET_STD, device=ref.device, dtype=ref.dtype).view(1, 3, 1, 1)
        return self._mean, self._std

    def __call__(self, images: Tensor) -> Tensor:
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W) images, got shape {tuple(images.shape)}")
        if images.shape[-2] != self.resize_h or images.shape[-1] != self.resize_w:
            images = F.interpolate(
                images, size=(self.resize_h, self.resize_w),
                mode="bilinear", align_corners=False, antialias=True,
            )
        mean, std = self._lazy_buffers(images)
        return (images - mean) / std


def _load_action_stats_override(path: str) -> dict[str, torch.Tensor]:
    """Load ``action.mean`` and ``action.std`` from a saved processor safetensors.

    Used when ``QFunctionConfig.action_stats_path`` is set: lets Q's normalizer
    use stats from an external source (typically BC's q5-only postprocessor)
    instead of the dataset-derived superset stats. Returns a dict in the
    ``{"action": {"mean": ..., "std": ...}}`` shape NormalizerProcessorStep expects.
    """
    from safetensors.torch import load_file
    weights = load_file(path)
    out = {}
    for stat in ("mean", "std"):
        key = f"action.{stat}"
        if key not in weights:
            raise ValueError(
                f"action_stats_path={path!r} is missing {key!r}. "
                "Point at a safetensors file containing action.mean and action.std."
            )
        out[stat] = weights[key]
    return out


def make_q_function_pre_post_processors(
    config: QFunctionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pre/post processors for the Q-function policy.

    Mirrors ACT's pipeline: rename → batch → device → normalize.
    The Q-function trains via the standard ``lerobot-train`` loop and consumes
    a batch dict containing (in cached mode) ``{cam}_preencoded`` features +
    ``action`` (chunk of length 2h) + the four Q keys injected by
    ``QValueLabelDataset``. Vision keys in ``input_features`` are absent from
    the batch in cached mode; the normalizer skips missing keys.

    If ``config.action_stats_path`` is set, the action ``mean/std`` from that
    file overrides the dataset-derived stats — used to align Q's normalization
    frame with BC's (when BC was trained on a smaller bucket).
    """
    if config.action_stats_path:
        override = _load_action_stats_override(config.action_stats_path)
        if dataset_stats is None:
            dataset_stats = {}
        dataset_stats = {**dataset_stats}
        existing_action = dict(dataset_stats.get("action", {}))
        existing_action.update(override)  # override only mean/std; preserve quantiles etc.
        dataset_stats["action"] = existing_action

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_q_batch_to_transition,
            to_output=_q_transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
