# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composed SuperMimic update, checkpoint and deployment contracts."""

import torch
from rsl_rl.algorithms import SuperMimic
from rsl_rl.algorithms.supermimic_task import (
    DifferentiableTask,
    EuclideanStateRepresentation,
    State,
)
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from compositional_rsl_rl.algorithm import CompositionalSuperMimic


class ScalarTask(DifferentiableTask):
    """A Euclidean system with privileged context and one successor event."""

    num_labels = 1

    def __init__(self, env: VecEnv, device: str) -> None:
        """Bind a one-dimensional Euclidean state."""
        self.representation = EuclideanStateRepresentation(1, error_floor=0.01)

    def actor_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Expose position and the most recent action."""
        return torch.cat((state[0], prev_actions[..., -1, :]), -1)

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Use the explicit deployable goal unchanged."""
        return goal

    def dynamics_context(
        self, state: State, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Expose privileged context and both action predecessors to dynamics."""
        return torch.cat((context, prev_actions.flatten(-2)), -1)

    def critic_obs(
        self,
        state: State,
        goal: torch.Tensor,
        context: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Build privileged scalar-value inputs."""
        return torch.cat((state[0], goal, context, prev_actions.flatten(-2)), -1)

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold exogenous goals and context fixed."""
        return goal, context

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reward goal tracking and penalize action magnitude."""
        target = goal[..., :1] if goal.shape[-1] else torch.zeros_like(next_state[0])
        reward = (
            1
            - (next_state[0] - target).square().squeeze(-1)
            - 0.01 * actions.square().sum(-1)
        )
        return reward if labels is None else reward + 0.01 * labels[..., 0]

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Keep the synthetic physical system alive in imagination."""
        return torch.zeros_like(next_state[0][..., 0])


class ScalarEnv(VecEnv):
    """Owned, pre-reset observations with optional interval pushes and nonzero reset actions."""

    def __init__(
        self,
        device: str = "cpu",
        goal_dim: int = 1,
        context_dim: int = 1,
        reset_action: float = 0.0,
        pushes: bool = False,
    ) -> None:
        """Allocate a small vectorized system with explicit physical observation groups."""
        self.num_envs, self.num_actions = 4, 1
        self.device, self.cfg = device, {}
        self.max_episode_length = 5
        self.episode_length_buf = torch.zeros(4, dtype=torch.long, device=device)
        self.position = torch.zeros(4, 1, device=device)
        self.goal = torch.ones(4, goal_dim, device=device)
        self.context = torch.zeros(4, context_dim, device=device)
        self.previous = torch.full((4, 1), reset_action, device=device)
        self.previous_previous = self.previous.clone()
        self.reset_action, self.pushes = reset_action, pushes

    def get_observations(self) -> TensorDict:
        """Return owned physical observations and reported action predecessors."""
        return TensorDict(
            {
                "state": self.position.clone(),
                "goal": self.goal.clone(),
                "context": self.context.clone(),
                "labels": (self.position > 0).float(),
                "prev_action": self.previous.clone(),
                "prev_prev_action": self.previous_previous.clone(),
            },
            batch_size=[4],
            device=self.device,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance, capture physical successors, then apply pushes and resets."""
        self.position.add_(0.2 * actions)
        self.previous_previous.copy_(self.previous)
        self.previous.copy_(actions)
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= self.max_episode_length
        final = self.get_observations()
        rewards = 1 - (self.position[:, 0] - 1).square()
        if self.pushes:
            self.position[~dones] += 10
        self.position[dones] = 0
        self.episode_length_buf[dones] = 0
        self.previous[dones] = self.reset_action
        self.previous_previous[dones] = self.reset_action
        return (
            self.get_observations(),
            rewards,
            dones,
            {
                "final_obs": final,
                "final_obs_every_step": True,
                "time_outs": dones.clone(),
            },
        )


def _cfg(**options: object) -> dict:
    algorithm = {
        "class_name": "SuperMimic",
        "task_cfg": {"class_name": ScalarTask},
        "updates_per_step": 1.0,
        "replay_buffer_size": 64,
        "warmup_steps": 4,
        "world_model_window": 3,
        "num_world_model_updates": 1,
        "world_model_batch_size": 8,
        "world_model_warmup_iterations": 0,
        "horizon": 3,
        "policy_batch_size": 4,
        "critic_epochs": 1,
        "critic_minibatches": 2,
    }
    algorithm.update(options)
    return {
        "algorithm": algorithm,
        "num_steps_per_env": 3,
        "save_interval": 100,
        "multi_gpu": None,
        "obs_groups": {
            key: [key]
            for key in (
                "state",
                "goal",
                "context",
                "labels",
                "prev_action",
                "prev_prev_action",
            )
        },
        "actor": {
            "class_name": "rsl_rl.algorithms.supermimic:SuperMimicActor",
            "history_length": 2,
            "encoder_hidden_dims": (8,),
            "decoder_hidden_dims": (8,),
        },
        "world_model": {
            "class_name": "WorldModel",
            "hidden_dims": (16,),
            "label_hidden_dims": (8,),
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": (8,),
            "obs_normalization": True,
        },
    }


def _collect(
    learner: SuperMimic, env: ScalarEnv, count: int, inference: bool = True
) -> None:
    with torch.inference_mode(inference), torch.no_grad():
        obs = env.get_observations()
        for _ in range(count):
            actions = learner.act(obs)
            obs, reward, done, extras = env.step(actions)
            learner.process_env_step(obs, reward, done, extras)


def _learner(env: ScalarEnv | None = None, **options: object) -> SuperMimic:
    env = ScalarEnv() if env is None else env
    return SuperMimic.construct_algorithm(
        env.get_observations(), env, _cfg(**options), env.device
    )


def config():
    cfg = _cfg()
    cfg["algorithm"]["class_name"] = CompositionalSuperMimic
    cfg["history_length"] = 2
    cfg["networks"] = {
        "actor": {"input": ["policy"], "hidden_dims": [8], "distribution_cfg": None},
        "goal_encoder": {"input": ["goal"], "hidden_dims": [8], "output_dim": 3},
        "world_model": {"input": ["dynamics"], "hidden_dims": [16]},
        "event_model": {"input": ["events"], "hidden_dims": [8]},
        "critic": {"input": ["critic"], "hidden_dims": [8], "obs_normalization": True},
    }
    for term in cfg["networks"].values():
        term["class_name"] = "MLPModel"
    return cfg


def test_composed_update_and_export(tmp_path):
    env = ScalarEnv()
    learner = CompositionalSuperMimic.construct_algorithm(
        env.get_observations(), env, config(), "cpu"
    )
    _collect(learner, env, 4)
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    metrics = learner.update()
    assert metrics and all(
        torch.isfinite(torch.tensor(value)) for value in metrics.values()
    )
    assert any(
        not torch.equal(before[k], v) for k, v in learner.actor.state_dict().items()
    )
    exported = learner.actor.as_jit().eval()
    script = torch.jit.script(exported)
    history = torch.randn(4, 2, learner.actor.obs_dim)
    goal = torch.randn(4, 1)
    with torch.no_grad():
        expected = learner.actor.forward_history(history, goal)
        torch.testing.assert_close(script(history, goal), expected)
    saved = learner.save()
    restored = CompositionalSuperMimic.construct_algorithm(
        env.get_observations(), env, config(), "cpu"
    )
    restored.load(saved, None, True)
    torch.testing.assert_close(restored.actor.forward_history(history, goal), expected)
