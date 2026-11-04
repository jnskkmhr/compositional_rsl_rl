# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose networks while retaining the native SuperMimic learning algorithm."""

from __future__ import annotations

import copy

import torch
from rsl_rl.algorithms import SuperMimic
from rsl_rl.algorithms.supermimic import SuperMimicActor
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State
from rsl_rl.env import VecEnv
from rsl_rl.modules import WorldModel
from tensordict import TensorDict
from torch import nn

from compositional_rsl_rl.network import NetworkManager
from compositional_rsl_rl.network.network import GraphModel


class CompositionalSuperMimic(SuperMimic):
    """Build disjoint actor, world-model and value graphs using ordinary network terms."""

    @staticmethod
    def construct_models(
        task: DifferentiableTask,
        groups: dict[str, list[str]],
        cfg: dict,
        env: VecEnv,
        actor_obs: torch.Tensor,
        actor_goal: torch.Tensor,
        critic_features: torch.Tensor,
        state: State,
        dynamics_context: torch.Tensor,
        device: str,
    ) -> tuple[SuperMimicActor, WorldModel, nn.Module]:
        """Expose normalized policy/dynamics inputs to configurable feature terms and heads.

        ``policy`` is the encoded goal followed by flattened observation history;
        ``goal`` is the normalized goal before encoding. ``dynamics`` contains
        normalized state, action and context inputs; ``events`` omits the action.
        ``critic`` contains task-owned imagined value features.
        """
        if cfg.get("loss"):
            raise ValueError("SuperMimic auxiliary loss composition is not supported.")
        networks_cfg = copy.deepcopy(cfg["networks"])
        history_length = cfg.get("history_length", 5)
        goal_dim = actor_goal.shape[-1]
        if goal_dim and not networks_cfg.get("goal_encoder"):
            raise ValueError(
                "A goal-conditioned SuperMimic policy requires networks.goal_encoder."
            )
        latent_dim = networks_cfg["goal_encoder"]["output_dim"] if goal_dim else 0
        feature_dim = task.representation.features(state).shape[-1]
        dynamics_dim = feature_dim + env.num_actions + dynamics_context.shape[-1]
        for name, width in (
            ("world_model", task.representation.increment_dim),
            ("event_model", task.num_labels),
        ):
            if name == "event_model" and not width:
                networks_cfg[name] = None
                continue
            if not networks_cfg.get(name):
                raise ValueError(f"SuperMimic requires networks.{name}.")
            configured = networks_cfg[name].get("output_dim")
            if configured is not None and configured != width:
                raise ValueError(f"networks.{name}.output_dim must be {width} or None.")
            networks_cfg[name]["output_dim"] = width
        example = TensorDict(
            {
                "policy": actor_obs.new_zeros(
                    env.num_envs, actor_obs.shape[-1] * history_length + latent_dim
                ),
                "goal": actor_goal,
                "dynamics": actor_obs.new_zeros(env.num_envs, dynamics_dim),
                "events": actor_obs.new_zeros(
                    env.num_envs, feature_dim + dynamics_context.shape[-1]
                ),
                "critic": critic_features,
            },
            batch_size=[env.num_envs],
        )
        networks = NetworkManager(
            example, networks_cfg, env.num_actions, stochastic_actor=False
        ).to(device)
        if any(
            term.is_recurrent or term.distribution is not None
            for term in networks.terms.values()
        ):
            raise ValueError(
                "SuperMimic network terms must be feed-forward and deterministic."
            )
        heads = {
            "actor": ["actor"],
            "critic": ["critic"],
            "world_model": ["world_model"],
        }
        if goal_dim:
            heads["actor"].append("goal_encoder")
        if task.num_labels:
            heads["world_model"].append("event_model")
        ownership = {
            name: {node for head in branch for node in networks.dependencies(head)}
            - networks.frozen
            for name, branch in heads.items()
        }
        names = list(ownership)
        for i, name in enumerate(names):
            for other in names[i + 1 :]:
                if ownership[name] & ownership[other]:
                    raise ValueError(
                        "SuperMimic optimizers cannot share trainable network terms."
                    )
        for name in ownership["actor"] | ownership["world_model"]:
            if networks_cfg[name].get("obs_normalization"):
                raise ValueError(
                    "Actor and world-model input normalization is owned by SuperMimic."
                )
        with torch.no_grad():
            networks.terms["actor"].mlp.output.weight.mul_(0.01)
            networks.terms["actor"].mlp.output.bias.zero_()
            networks.terms["world_model"].mlp.output.weight.zero_()
            networks.terms["world_model"].mlp.output.bias.zero_()
        actor = SuperMimicActor(
            task,
            groups,
            actor_obs.shape[-1],
            goal_dim,
            env.num_envs,
            env.num_actions,
            history_length=history_length,
            latent_dim=latent_dim,
            encoder=_Graph(networks, "goal_encoder", "goal") if goal_dim else None,
            decoder=_Graph(networks, "actor", "policy"),
        ).to(device)
        model = WorldModel(
            task.representation,
            feature_dim,
            env.num_actions,
            dynamics_context.shape[-1],
            num_labels=task.num_labels,
            net=_Graph(networks, "world_model", "dynamics"),
            label_head=_Graph(networks, "event_model", "events")
            if task.num_labels
            else None,
        ).to(device)
        return actor, model, _Graph(networks, "critic", "critic")


class _Graph(GraphModel):
    """Adapt normalized tensor inputs and update only this optimizer's value moments."""

    def __init__(self, manager: NetworkManager, name: str, input_name: str) -> None:
        super().__init__(manager, name)
        if self.obs_groups != [input_name]:
            raise ValueError(
                f"The {name} graph must consume only the synthetic '{input_name}' input."
            )
        self.input_name = input_name

    def forward(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            obs = TensorDict({self.input_name: obs}, batch_size=obs.shape[:-1])
        return super().forward(obs)

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        values = obs.clone(recurse=False)
        for name, model in self.terms.items():
            if name not in self.manager.frozen:
                model.update_normalization(values)
            if name != self.name:
                values[name] = model(values)
