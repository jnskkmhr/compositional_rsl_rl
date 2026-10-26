# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Complete IsaacLab AME agent config; policy, critic, and elevation_map are observation groups."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.components.network import AttentionMapNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class MyNetworksCfg(NetworkCfg):
    """Share one actor-visible AME term across the actor and critic."""

    map_encoder: AttentionMapNetworkCfg | None = None


@configclass
class MyCompositionalPPORunnerCfg(CompositionalRunnerCfg):
    """Select this class with the task's existing rsl_rl_cfg_entry_point."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    networks = MyNetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        map_encoder=AttentionMapNetworkCfg(
            input=["elevation_map", "policy"],
            map_input="elevation_map",
            map_shape=(33, 21, 3),
            output_dim=64,
            feature_dim=64,
            num_heads=16,
            cnn_channels=16,
            cnn_downsample=True,
            cnn_normalization="batch",
            attach_global=False,
            hidden_dims=[],
            obs_normalization=False,
            frozen=False,
            pretrained_cfg=None,
        ),
    )
    loss = LossesCfg()  # PPO trains the encoder; no auxiliary labels are required.
    algorithm = CompositionalPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    )
    logger = "wandb"
    wandb_project = "attention_map_ppo"
    experiment_name = "attention_map_ppo"
