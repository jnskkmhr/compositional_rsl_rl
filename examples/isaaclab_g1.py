# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Full IsaacLab task config. The environment must return privileged_state with shape [num_envs, 3]."""

from compositional_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rl.loss import LossesCfg, SupervisedLossCfg
from compositional_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg
from compositional_rl.runner import CompositionalRunnerCfg
from isaaclab.utils.configclass import configclass


@configclass
class G1NetworksCfg(NetworkCfg):
    """Actor and critic are peers of the learned state estimator."""

    state_estimator: MLPNetworkCfg | None = None


@configclass
class G1LossesCfg(LossesCfg):
    """Named loss terms are independent of network construction."""

    estimator_loss: SupervisedLossCfg | None = None


@configclass
class G1RoughCompositionalPPORunnerCfg(CompositionalRunnerCfg):
    """Select through the task's existing rsl_rl_cfg_entry_point; keep train/play scripts unchanged."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500

    # Raw observation dependencies are inferred from each network term's input list.
    networks = G1NetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "state_estimator", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "privileged", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        state_estimator=MLPNetworkCfg(
            input=["policy"],
            output_dim=3,
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=False,
            frozen=False,
            pretrained_cfg=None,
        ),
    )

    loss = G1LossesCfg(
        estimator_loss=SupervisedLossCfg(
            trains=["state_estimator"],
            prediction="state_estimator",
            target="privileged_state",
            loss_fn="mse",
            optimizer="joint",
            weight=1.0,
            # For separate supervision: optimizer="separate", learning_rate=1.0e-3.
        ),
    )

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
        # Enable your RslRlSymmetryCfg only when its observation transform handles all supplied groups.
        # Upstream RSL-RL disallows symmetry for recurrent actor/critic models.
        symmetry_cfg=None,
    )

    logger = "wandb"
    wandb_project = "g1_29dof_rigid_rough_ppo"
    experiment_name = "g1_29dof_rigid_rough_compositional_ppo"
