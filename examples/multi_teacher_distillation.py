# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO plus two frozen teachers for a 29-action locomotion/manipulation student.

The environment supplies policy, critic, locomotion_teacher_obs, and manipulation_teacher_obs.
Teacher observation layouts and action conventions must match the source policies exactly.
This example assumes student actions are legs (12), waist (3), left arm (7), right arm (7).
"""

from dataclasses import dataclass

from compositional_rsl_rl.loss import DistillationLossCfg, LossesCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg, TeacherPolicyCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg


@dataclass(kw_only=True)
class TeacherNetworksCfg(NetworkCfg):
    locomotion_teacher: TeacherPolicyCfg
    manipulation_teacher: TeacherPolicyCfg


@dataclass(kw_only=True)
class TeacherLossesCfg(LossesCfg):
    locomotion: DistillationLossCfg
    manipulation: DistillationLossCfg


def make_config(locomotion_checkpoint: str, manipulation_checkpoint: str) -> CompositionalRunnerCfg:
    """Configure independent actor checkpoints and a PPO-trained student with two imitation losses."""
    return CompositionalRunnerCfg(
        networks=TeacherNetworksCfg(
            actor=ActorHeadCfg(input=["policy"], hidden_dims=[512, 256, 128]),
            critic=CriticHeadCfg(input=["critic"], hidden_dims=[512, 256, 128]),
            locomotion_teacher=TeacherPolicyCfg(
                input=["locomotion_teacher_obs"],
                output_dim=15,
                hidden_dims=[512, 256, 128],
                pretrained_cfg={"path": locomotion_checkpoint, "state_dict_key": "actor_state_dict"},
            ),
            manipulation_teacher=TeacherPolicyCfg(
                input=["manipulation_teacher_obs"],
                output_dim=14,
                hidden_dims=[512, 256, 128],
                pretrained_cfg={"path": manipulation_checkpoint, "state_dict_key": "actor_state_dict"},
            ),
        ),
        loss=TeacherLossesCfg(
            locomotion=DistillationLossCfg(
                target="locomotion_teacher",
                prediction_indices=list(range(15)),
                weight=1.0,
            ),
            manipulation=DistillationLossCfg(
                target="manipulation_teacher",
                prediction_indices=list(range(15, 29)),
                weight=1.0,
            ),
        ),
        experiment_name="multi_teacher_distillation",
    )
