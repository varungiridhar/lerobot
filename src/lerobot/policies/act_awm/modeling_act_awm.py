#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from transformers import AutoModel, AutoProcessor

from lerobot.policies.act_awm.configuration_act_awm import ACTAWMConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTAWMPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer with AWM Policy
    """

    config_class = ACTAWMConfig
    name = "act_awm"

    def __init__(
        self,
        config: ACTAWMConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.expert_trajectories = None
        if config.expert_trajectory_path:
            import pickle
            with open(config.expert_trajectory_path, "rb") as f:
                traj_data = pickle.load(f)
            self.expert_trajectories = {
                seed: (traj, done_idx) 
                for seed, traj, done_idx in zip(
                    traj_data["seeds"], 
                    traj_data["trajectories"], 
                    traj_data["done_indices"]
                )
            }

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            if self.expert_trajectories is not None and "seed" in batch:
                seeds = batch["seed"].cpu().tolist()
                policy_device = next(self.parameters()).device
                goal_observations = []
                for seed in seeds:
                    if seed not in self.expert_trajectories:
                        raise ValueError(f"Seed {seed} not found in expert trajectories")
                    trajectory, done_idx = self.expert_trajectories[seed]
                    goal_obs = {k: v[done_idx].to(policy_device) for k, v in trajectory.items()}
                    goal_observations.append(goal_obs)
                self._goal_observations = goal_observations
            else:
                self._goal_observations = None
            
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        
        if self.config.gradient_based_planning and self._goal_observations is not None:
            # TODO: Implement gradient-based planning here
            # self._goal_observations contains goal observations for each env in the batch
            # Each goal_obs is a dict with keys like 'observation.image', 'observation.state'
            # Optimize actions to minimize delta between predicted future state and goal state
            pass
        
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, future_state_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        loss_dict = {}
        total_loss = 0.0
        
        # ============================================
        # POLICY LOSS (Phase 1 or Joint Training)
        # ============================================
        if self.config.train_policy:
            l1_loss = (
                F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
            ).mean()
            loss_dict["l1_loss"] = l1_loss.item()
            total_loss = total_loss + l1_loss
            
            if self.config.use_vae:
                # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
                # each dimension independently, we sum over the latent dimension to get the total
                # KL-divergence per batch element, then take the mean over the batch.
                # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
                mean_kld = (
                    (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
                )
                loss_dict["kld_loss"] = mean_kld.item()
                total_loss = total_loss + mean_kld * self.config.kl_weight
        
        # ============================================
        # AWM LOSS (Phase 2 or Joint Training)
        # ============================================
        if self.config.train_awm and future_state_hat is not None:
            # Get target: encoder output for final observation (t+H)
            # observation_delta_indices = [0, chunk_size] gives us observations at t and t+H
            future_state_target = self.model._get_future_state_target(batch, batch_size=actions_hat.shape[0])
            
            # future_state_hat: (encoder_seq_len, B, D)
            # future_state_target: (encoder_seq_len, B, D)
            awm_loss = F.mse_loss(future_state_hat, future_state_target)
            loss_dict["awm_loss"] = awm_loss.item()
            total_loss = total_loss + awm_loss * self.config.future_state_weight

        return total_loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: ACTAWMConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Note: SigLIP2 is no longer used for AWM.
        # self.siglip2_model = AutoModel.from_pretrained(self.config.siglip2_model_path).eval()
        # self.siglip2_processor = AutoProcessor.from_pretrained(self.config.siglip2_model_path)

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        self.action_encoder = nn.Sequential(
            nn.Linear(self.config.action_feature.shape[0], config.dim_model),
            nn.ReLU(),
            nn.Linear(config.dim_model, config.dim_model)
        )

        from copy import copy
        awm_config = copy(config)
        awm_config.n_decoder_layers = config.n_awm_decoder_layers
        self.awm_decoder = ACTDecoder(awm_config)
        
        self.future_queries_base = nn.Parameter(torch.randn(1, 1, config.dim_model)) # TODO: is this the right way to do stuff
        
        self.awm_action_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)        
        max_encoder_len = 2000
        self.awm_query_pos_embed = nn.Embedding(max_encoder_len, config.dim_model) # TODO: is this the right way to do stuff

        # Output heads.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()
        self._freeze_parameters()  # Freeze components based on training flags

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters(), self.awm_decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _freeze_parameters(self):
        """Freeze parameters based on training flags."""
        # Co-training mode: both policy and AWM train together
        if self.config.train_policy and self.config.train_awm:
            print("[INFO] Co-training mode: Both policy and AWM components active")
            if self.config.detach_encoder_for_awm:
                print("[WARNING] detach_encoder_for_awm=True in co-training mode - AWM won't improve encoder!")
            return
        
        if not self.config.train_policy:
            # Freeze all policy components
            components_to_freeze = [
                self.encoder,
                self.decoder,
                self.action_head,
            ]
            if self.config.use_vae:
                components_to_freeze.extend([
                    self.vae_encoder,
                    self.vae_encoder_cls_embed,
                    self.vae_encoder_robot_state_input_proj,
                    self.vae_encoder_action_input_proj,
                    self.vae_encoder_latent_output_proj,
                ])
            if self.config.image_features:
                components_to_freeze.append(self.backbone)
            if self.config.robot_state_feature:
                components_to_freeze.append(self.encoder_robot_state_input_proj)
            if self.config.env_state_feature:
                components_to_freeze.append(self.encoder_env_state_input_proj)
            
            components_to_freeze.extend([
                self.encoder_latent_input_proj,
                self.encoder_1d_feature_pos_embed,
            ])
            if self.config.image_features:
                components_to_freeze.extend([
                    self.encoder_img_feat_input_proj,
                    self.encoder_cam_feat_pos_embed,
                ])
            components_to_freeze.append(self.decoder_pos_embed)
            
            for component in components_to_freeze:
                for param in component.parameters():
                    param.requires_grad = False
            
            print("[INFO] Frozen policy components for AWM training")
        
        if not self.config.train_awm:
            # Freeze all AWM components
            awm_components = [
                self.action_encoder,
                self.awm_decoder,
                self.awm_action_pos_embed,
                self.awm_query_pos_embed,
            ]
            
            for component in awm_components:
                for param in component.parameters():
                    param.requires_grad = False
            
            # Also freeze the future_queries_base parameter
            self.future_queries_base.requires_grad = False
            
            print("[INFO] Frozen AWM components for policy training")

    def _get_future_state_target(self, batch: dict[str, Tensor], batch_size: int) -> Tensor:
        """Extract encoder output for final observation (t+H) with zero latent.
        
        This creates the supervision target for AWM training.
        
        Args:
            batch: Training batch containing observations
            batch_size: Batch size
            
        Returns:
            Tensor of shape (encoder_seq_len, B, D) representing the encoded future state
        """
        with torch.no_grad():
            # Extract final observations (t+H)
            final_obs_images = None
            if OBS_IMAGES in batch:
                final_obs_images = [img[:, 1] for img in batch[OBS_IMAGES]]
            
            final_obs_state = None
            if OBS_STATE in batch:
                final_obs_state = batch[OBS_STATE][:, 1]
            
            final_obs_env_state = None
            if OBS_ENV_STATE in batch:
                final_obs_env_state = batch[OBS_ENV_STATE][:, 1]
            
            # Zero latent (no VAE at test time)
            device = final_obs_images[0].device if final_obs_images else (
                final_obs_state.device if final_obs_state is not None else batch[OBS_ENV_STATE].device
            )
            latent_zero = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)
            
            # Prepare encoder input for final observation
            final_encoder_in_tokens = [self.encoder_latent_input_proj(latent_zero)]
            final_encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
            
            # Robot state token
            if self.config.robot_state_feature:
                final_encoder_in_tokens.append(self.encoder_robot_state_input_proj(final_obs_state))
            
            # Environment state token
            if self.config.env_state_feature:
                final_encoder_in_tokens.append(self.encoder_env_state_input_proj(final_obs_env_state))
            
            # Image tokens
            if self.config.image_features:
                for img in final_obs_images:
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)
                    
                    # Rearrange features to (sequence, batch, dim)
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    
                    final_encoder_in_tokens.extend(list(cam_features))
                    final_encoder_in_pos_embed.extend(list(cam_pos_embed))
            
            # Stack all tokens
            final_encoder_in_tokens = torch.stack(final_encoder_in_tokens, axis=0)
            final_encoder_in_pos_embed = torch.stack(final_encoder_in_pos_embed, axis=0)
            
            # Encode future observation
            final_encoder_out = self.encoder(final_encoder_in_tokens, pos_embed=final_encoder_in_pos_embed)
            
        return final_encoder_out  # (encoder_seq_len, B, D)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        # Ensure observations have temporal dimension (B, T, ...)
        # Training: dataset provides (B, T=2, ...) via observation_delta_indices
        # Eval: environment provides (B, ...), so add T=1 dimension
        batch = dict(batch)  # shallow copy to avoid modifying original
        if OBS_IMAGES in batch:
            batch[OBS_IMAGES] = [
                img if img.ndim == 5 else img.unsqueeze(1)  # (B,C,H,W) -> (B,1,C,H,W)
                for img in batch[OBS_IMAGES]
            ]
        if OBS_STATE in batch and batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)  # (B, state_dim) -> (B, 1, state_dim)
        if OBS_ENV_STATE in batch and batch[OBS_ENV_STATE].ndim == 2:
            batch[OBS_ENV_STATE] = batch[OBS_ENV_STATE].unsqueeze(1)

        # Extract current observations (timestep 0) from temporal dimension
        current_obs_images = [img[:, 0] for img in batch[OBS_IMAGES]] if OBS_IMAGES in batch else None
        current_obs_state = batch[OBS_STATE][:, 0] if OBS_STATE in batch else None
        current_obs_env_state = batch[OBS_ENV_STATE][:, 0] if OBS_ENV_STATE in batch else None

        batch_size = current_obs_images[0].shape[0] if current_obs_images else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        # Only run VAE encoder during policy training (Phase 1)
        if self.config.use_vae and ACTION in batch and self.training and self.config.train_policy:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(current_obs_state)
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=current_obs_state.device if current_obs_state is not None else batch[ACTION].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            device = current_obs_state.device if current_obs_state is not None else (
                current_obs_images[0].device if current_obs_images else current_obs_env_state.device
            )
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(device)

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(current_obs_state))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(current_obs_env_state))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in current_obs_images:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # ============================================
        # STAGE 1: TRANSFORMER ENCODER (Always needed)
        # ============================================
        # Encoder is always needed (for policy actions in Phase 1, and AWM targets in Phase 2)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        
        # ============================================
        # STAGE 2: POLICY DECODER (Actions Only)
        # ============================================
        # Only run policy decoder during policy training or inference (Phase 1)
        if self.config.train_policy or not self.training:
            # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
            decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )
            decoder_out = decoder_out.transpose(0, 1)
            actions = self.action_head(decoder_out)
        else:
            # During AWM-only training, return dummy actions (not used for loss)
            actions = torch.zeros(
                (batch_size, self.config.chunk_size, self.config.action_feature.shape[0]),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )

        # ============================================
        # STAGE 3: AWM DECODER (Future State Prediction)
        # ============================================
        # AWM predicts the full encoder output sequence for t+H observation
        # Uses cross-attention decoder: [actions + future_queries] attend to encoder_out
        
        future_state_pred = None
        if self.config.train_awm:
            # Encode actions for AWM
            if self.training and ACTION in batch:
                actions_for_awm = batch[ACTION]
            else:
                actions_for_awm = actions
            
            action_embeddings = self.action_encoder(actions_for_awm)  # (B, chunk_size, D)
            
            encoder_seq_len = encoder_out.shape[0]
            
            future_queries = self.future_queries_base.expand(encoder_seq_len, batch_size, self.config.dim_model) # (encoder_seq_len, B, D): dynamic queries to match encoder sequence length
            
            action_embeddings_seq = action_embeddings.transpose(0, 1) # (chunk_size, B, D)
            
            awm_decoder_in = torch.cat([action_embeddings_seq, future_queries], dim=0) # (chunk_size + encoder_seq_len, B, D)
            
            action_pos = self.awm_action_pos_embed.weight[:self.config.chunk_size].unsqueeze(1)  # (chunk_size, 1, D)
            query_pos = self.awm_query_pos_embed.weight[:encoder_seq_len].unsqueeze(1)  # (encoder_seq_len, 1, D)
            awm_pos = torch.cat([action_pos, query_pos], dim=0)  # (chunk_size + encoder_seq_len, 1, D)
            
            # - Co-training (train_policy=True, train_awm=True, detach=False): AWM loss helps refine encoder
            # - Two-stage (train_policy=False, train_awm=True, detach=True): AWM trains on frozen encoder
            encoder_for_awm = encoder_out.detach() if self.config.detach_encoder_for_awm else encoder_out
            encoder_pos_for_awm = encoder_in_pos_embed.detach() if self.config.detach_encoder_for_awm else encoder_in_pos_embed
            
            awm_decoder_out = self.awm_decoder(
                awm_decoder_in,
                encoder_for_awm,
                decoder_pos_embed=awm_pos,
                encoder_pos_embed=encoder_pos_for_awm,
            ) # (chunk_size + encoder_seq_len, B, D)
            
            future_state_pred = awm_decoder_out[self.config.chunk_size:] # (encoder_seq_len, B, D)

        return actions, future_state_pred, (mu, log_sigma_x2)

    @torch.no_grad()
    def get_siglip2_embeddings(self, images, pool_size=None, use_cls_token=False):
        """Extract raw patch tokens from SigLIP-2 (FLARE-style, excluding [CLS] token)."""
        # Denormalize from ImageNet stats.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        denormalized_images = images * std + mean
        
        image_list = [denormalized_images[i] for i in range(denormalized_images.shape[0])]
        inputs = self.siglip2_processor(images=image_list, return_tensors="pt", do_rescale=False).to(images.device)
        vision_outputs = self.siglip2_model.vision_model(**inputs)
        
        if use_cls_token:
            # Return pooled representation (single vector per image)
            # Shape: (B, D) where D is siglip2_embedding_dim (typically 768 or 1152)
            return vision_outputs.pooler_output
        else:
            # Return patch tokens
            patch_tokens = vision_outputs.last_hidden_state
            
            # Optional spatial pooling to reduce token count.
            if pool_size is not None and pool_size > 1:
                B, N, D = patch_tokens.shape
                H = W = int(N ** 0.5)
                tokens_spatial = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
                pooled = torch.nn.functional.avg_pool2d(tokens_spatial, kernel_size=pool_size)
                _, _, H_new, W_new = pooled.shape
                patch_tokens = pooled.permute(0, 2, 3, 1).reshape(B, H_new * W_new, D)
            
            return patch_tokens


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTAWMConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTAWMConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(
        self,
        x,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTAWMConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTAWMConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
