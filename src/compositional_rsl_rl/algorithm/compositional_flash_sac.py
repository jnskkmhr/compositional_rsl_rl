# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Network/loss composition over the single native RSL-RL FlashSAC update loop."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict
from torch import nn

from compositional_rsl_rl.loss import LossManager
from compositional_rsl_rl.network import NetworkManager
from rsl_rl.algorithms import FlashSAC
from rsl_rl.algorithms.flash_sac import FlashSACPolicy
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config


class CompositionalFlashSAC(FlashSAC):
    """FlashSAC with explicit actor, critic and separate auxiliary optimizer ownership."""

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> FlashSAC:
        """Build disjoint trainable branches and retain the base algorithm's optimization schedule."""
        if cfg.get("torch_compile_mode"):
            raise ValueError("CompositionalFlashSAC currently supports eager training.")
        if cfg.get("networks") is None:
            raise ValueError(
                "CompositionalFlashSAC requires an explicit networks configuration."
            )
        alg_cfg = copy.deepcopy(cfg["algorithm"])
        alg_cfg.pop("class_name")
        action_input = alg_cfg.pop("critic_action_input", "critic")
        if action_input not in cfg["networks"]:
            raise ValueError("critic_action_input must name a configured network term.")
        alg_cfg = resolve_symmetry_config(alg_cfg, env)
        networks = NetworkManager(
            obs,
            cfg["networks"],
            env.num_actions,
            head_dims={"actor": env.num_actions, "critic": 1},
            stochastic_actor=False,
            extra_input_dims={action_input: env.num_actions},
        ).to(device)
        if any(model.is_recurrent for model in networks.terms.values()):
            raise ValueError("CompositionalFlashSAC does not support recurrent replay.")
        from rsl_rl.modules import EmpiricalNormalization

        if any(
            isinstance(getattr(model, "obs_normalizer", None), EmpiricalNormalization)
            for model in networks.terms.values()
        ):
            raise ValueError(
                "FlashSAC graph terms require obs_normalization=False; native heads use batch normalization."
            )
        if {"actor", "critic"} & networks.frozen:
            raise ValueError("FlashSAC actor and critic heads must be trainable.")
        obs_keys = set(obs.keys())
        next_keys = {"next_" + key for key in obs_keys}
        loss_keys = obs_keys | next_keys
        if obs_keys & next_keys:
            raise ValueError(
                "Observation names collide with next-observation auxiliary labels."
            )
        losses = LossManager(
            networks, cfg.get("loss") or {}, loss_keys, joint_phases=("actor", "critic")
        )
        owned = {head: set(networks.dependencies(head)) for head in ("actor", "critic")}
        for term in losses.terms.values():
            if term["prediction"] in ("actor", "critic") or term["target"] in (
                "actor",
                "critic",
            ):
                raise ValueError(
                    "SAC auxiliary losses require explicit auxiliary prediction/target terms."
                )
            if term["mode"] in owned:
                owned[term["mode"]].update(term["trains"])
        trainable = {
            head: owned[head] - networks.frozen - networks.separate for head in owned
        }
        if trainable["actor"] & trainable["critic"]:
            raise ValueError(
                "Actor and critic cannot share trainable graph terms; share only frozen terms."
            )
        if action_input not in networks.dependencies(
            "critic"
        ) or action_input in networks.dependencies("actor"):
            raise ValueError(
                "critic_action_input must belong exclusively to the critic graph."
            )
        if any(
            action_input in networks.dependencies(name) for name in networks.separate
        ):
            raise ValueError(
                "Action-conditioned terms cannot be detached by a separate optimizer."
            )
        models = {
            head: FlashSACGraph(
                networks, head, owned[head], action_input if head == "critic" else None
            )
            for head in owned
        }
        actor_distribution = networks.terms["actor"].distribution
        if not callable(getattr(actor_distribution, "rsample_with_log_prob", None)):
            raise ValueError(
                "FlashSAC requires a squashed Gaussian actor distribution with differentiable sampling."
            )
        FlashSAC.validate_models(
            models["actor"],
            models["critic"],
            alg_cfg.get("num_bins", 101),
            alg_cfg.get("value_max", 5.0),
        )
        # Separate terms may supply detached features, but are not owned by the SAC optimizers.
        algorithm = CompositionalFlashSAC(
            models["actor"],
            models["critic"],
            env.num_actions,
            {head: networks.observation_groups(head) for head in owned},
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )
        algorithm.networks = networks
        algorithm.losses = losses
        for head in owned:
            params = [
                p
                for name in networks.order
                if name in trainable[head]
                for p in networks.terms[name].parameters()
                if p.requires_grad
            ]
            if len({id(p) for p in params}) != len(params):
                raise ValueError("A graph parameter has multiple optimizer owners.")
            algorithm.optimizers[head] = torch.optim.Adam(
                params,
                lr=algorithm.initial_learning_rate,
                fused=algorithm.device.type == "cuda",
            )
            algorithm.schedulers[head] = torch.optim.lr_scheduler.LambdaLR(
                algorithm.optimizers[head], algorithm._learning_rate_factor
            )
        actor_ids = {
            id(p)
            for group in algorithm.optimizers["actor"].param_groups
            for p in group["params"]
        }
        critic_ids = {
            id(p)
            for group in algorithm.optimizers["critic"].param_groups
            for p in group["params"]
        }
        if actor_ids & critic_ids:
            raise ValueError(
                "A parameter cannot belong to both actor and critic optimizers."
            )
        return algorithm

    def actor_observations(self, obs: TensorDict) -> TensorDict:
        """Keep named observations for graph evaluation."""
        return obs

    def critic_observations(self, obs: TensorDict) -> TensorDict:
        """Keep named observations for graph evaluation."""
        return obs

    def _loss_observations(self, batch: TensorDict) -> TensorDict:
        observations = batch["observations"].clone(recurse=False)
        for key, value in batch["next_observations"].items():
            observations["next_" + key] = value
        return observations

    def auxiliary_loss(self, phase: str, batch: TensorDict) -> torch.Tensor:
        """Attach each joint auxiliary loss to its explicitly selected SAC update."""
        joint, _, _ = self.losses.compute(self._loss_observations(batch), phase=phase)
        return sum(joint, torch.zeros((), device=self.device))

    def update_auxiliary(self, batch: TensorDict) -> dict[str, torch.Tensor]:
        """Train separate auxiliary terms from the sampled replay observations."""
        _, separate, metrics = self.losses.compute(
            self._loss_observations(batch), phase="separate"
        )
        for optimizer in self.losses.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if separate:
            sum(separate.values()).backward()
            for optimizer in self.losses.optimizers.values():
                self.reduce_parameters(optimizer)
                optimizer.step()
        return {
            f"auxiliary/{key}": torch.tensor(value, device=self.device)
            for key, value in metrics.items()
        }

    def broadcast_parameters(self) -> None:
        """Include frozen and separately optimized graph terms in initial synchronization."""
        super().broadcast_parameters(
            (self.networks, self.target_critic, self.temperature)
        )

    def synchronize_buffers(self) -> None:
        """Share graph and target running statistics after all optimizer updates."""
        super().synchronize_buffers(
            (self.networks, self.target_critic, self.temperature)
        )

    def save(self) -> dict:
        """Include graph and optimizer ownership so incompatible resumes fail explicitly."""
        state = super().save()
        state.update(
            networks=self.networks.state_dict(),
            network_cfg=self.networks.cfg,
            critic_action_input=self.critic.action_input,
            loss_cfg=self.losses.cfg,
            loss_optimizers={
                key: value.state_dict() for key, value in self.losses.optimizers.items()
            },
        )
        return state

    def load(
        self, state: dict, load_cfg: dict | None = None, strict: bool = True
    ) -> bool:
        """Restore matching composition checkpoints and auxiliary optimizer state."""
        if state.get("critic_action_input") != self.critic.action_input:
            raise ValueError(
                "FlashSAC critic action routing differs from the checkpoint."
            )
        if (
            state.get("network_cfg") != self.networks.cfg
            or state.get("loss_cfg") != self.losses.cfg
        ):
            raise ValueError(
                "FlashSAC graph or loss ownership differs from the checkpoint."
            )
        restore = super().load(state, load_cfg, strict)
        if load_cfg is None or (load_cfg.get("actor") and load_cfg.get("critic")):
            self.networks.load_state_dict(state["networks"], strict=strict)
        if load_cfg is None or load_cfg.get("optimizer"):
            for key, optimizer in self.losses.optimizers.items():
                optimizer.load_state_dict(state["loss_optimizers"][key])
        return restore

    def get_policy(self) -> FlashSACPolicy:
        """Export the actor branch without critic or auxiliary-only dependencies."""
        actor = FlashSACGraph(
            self.networks, "actor", set(self.networks.dependencies("actor"))
        )
        return FlashSACPolicy(
            actor,
            self.obs_groups["actor"],
            self.action_support,
            concatenate_observations=False,
        )


class FlashSACGraph(NetworkManager):
    """Own one graph branch, sharing its registered terms with the composition manager."""

    def __init__(
        self,
        manager: NetworkManager,
        head: str,
        owned: set[str],
        action_input: str | None = None,
    ) -> None:
        """Retain branch ownership and the optional action-input connection."""
        nn.Module.__init__(self)
        self.head = head
        self.action_input = action_input
        self.order = [name for name in manager.order if name in owned]
        self.terms = nn.ModuleDict({name: manager.terms[name] for name in self.order})
        self.inputs = {name: manager.inputs[name] for name in self.order}
        self.obs_shapes = manager.obs_shapes
        self.frozen = manager.frozen & owned
        self.separate = manager.separate & owned

    @property
    def value_support(self) -> torch.Tensor:
        """Expose the configured critic's categorical support."""
        return self.terms[self.head].distribution.bin_values.flatten()

    def as_inference(self) -> nn.Module:
        """Prepare the owned actor branch through its optional inference adapter."""
        graph = copy.deepcopy(self)
        for name, term in graph.terms.items():
            if callable(getattr(term, "as_inference", None)):
                graph.terms[name] = term.as_inference()
        return graph

    def _raw_output(
        self,
        observations: TensorDict,
        training: bool,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = observations.clone(recurse=False)
        for name in self.dependencies(self.head):
            model = self.terms[name]
            extra = actions if name == self.action_input else None
            kwargs = {"extra_input": extra} if name == self.action_input else {}
            if name == self.head:
                return model.mlp(model.get_latent(values, **kwargs))
            if callable(getattr(model, "forward_with_mode", None)):
                result = model.forward_with_mode(
                    values, training=training and name not in self.frozen, **kwargs
                )
            else:
                result = model(values, **kwargs)
            values[name] = result.detach() if name in self.separate else result
        raise RuntimeError("The graph has no output head.")

    def get_mean_and_std(
        self, observations: TensorDict, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate feature terms and obtain pre-tanh actor distribution parameters."""
        output = self._raw_output(observations, training)
        return self.terms[self.head].distribution.get_mean_and_std(output)

    def forward(
        self,
        observations: TensorDict,
        actions: torch.Tensor | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supply actions at the configured critic term, then evaluate ordinary output distributions."""
        output = self._raw_output(observations, training, actions)
        distribution = self.terms[self.head].distribution
        if self.head == "actor":
            actions, log_prob = distribution.rsample_with_log_prob(output)
            return actions, {"log_prob": log_prob}
        values, log_prob = distribution.evaluate(output)
        # Named terms keep batch first; the core Q learner expects ensemble first.
        if values.ndim == 2:
            values, log_prob = values.unsqueeze(1), log_prob.unsqueeze(1)
        return values.squeeze(-1).transpose(0, 1), {
            "log_prob": log_prob.transpose(0, 1)
        }
