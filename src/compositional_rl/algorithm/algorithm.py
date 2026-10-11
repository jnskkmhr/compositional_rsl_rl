# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL 5.5.1 integration. Rollout/GAE/runner APIs remain upstream implementations.

The update loop is adapted from that pinned version because upstream has no loss hook. Keep this single
integration point synchronized when upgrading the dependency; task-specific algorithms are unnecessary.
"""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict
from torch import nn

from compositional_rl.loss import LossManager
from compositional_rl.network import GraphModel, NetworkManager
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets, resolve_class, resolve_obs_groups, resolve_optimizer


class CompositionalPPO(PPO):
    """PPO over peer network terms, with explicit joint/separate supervised losses."""

    def __init__(
        self,
        networks: NetworkManager,
        losses: dict,
        storage: RolloutStorage,
        **kwargs: object,
    ) -> None:
        """Create unique parameter ownership and retain the standard PPO rollout interface."""
        device = kwargs.get("device", "cpu")
        networks.to(device)
        # Initialize upstream facilities with distinct heads, avoiding shared-parameter duplication.
        super().__init__(networks.terms["actor"], networks.terms["critic"], storage, **kwargs)
        self.networks = networks
        self.losses = LossManager(networks, losses, set(storage.observations.keys()))
        self.actor = self._raw_actor = GraphModel(networks, "actor")
        self.critic = self._raw_critic = GraphModel(networks, "critic")
        active = set(networks.dependencies("actor")) | set(networks.dependencies("critic"))
        for term in self.losses.terms.values():
            if term["mode"] == "joint":
                active.update(term["trains"])
        active -= networks.separate | networks.frozen
        self.ppo_parameters = [
            p for name in networks.order if name in active for p in networks.terms[name].parameters() if p.requires_grad
        ]
        self.optimizer_parameter_names = [
            f"{name}.{key}"
            for name in networks.order
            if name in active
            for key, p in networks.terms[name].named_parameters()
            if p.requires_grad
        ]
        if not self.ppo_parameters:
            raise ValueError("PPO requires at least one trainable policy/value parameter.")
        self.optimizer = resolve_optimizer(kwargs.get("optimizer", "adam"))(self.ppo_parameters, lr=self.learning_rate)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct through the unmodified upstream OnPolicyRunner."""
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("multi_gpu", None)
        if cfg.get("networks") is None:
            cfg["algorithm"]["class_name"] = "PPO"
            return PPO.construct_algorithm(obs, env, cfg, device)
        _, alg_cfg = resolve_class(cfg["algorithm"])
        if alg_cfg.pop("share_cnn_encoders", False):
            raise ValueError("Share a named network term through input references instead of share_cnn_encoders.")
        networks = NetworkManager(obs, cfg["networks"], env.num_actions)
        obs_groups = dict(cfg.get("obs_groups") or {})
        for head in ("actor", "critic"):
            obs_groups[head] = networks.observation_groups(head)
        defaults = ["actor", "critic"] + (["rnd_state"] if alg_cfg.get("rnd_cfg") else [])
        obs_groups = resolve_obs_groups(obs, obs_groups, defaults)
        alg_cfg = resolve_rnd_config(alg_cfg, obs, obs_groups, env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm = CompositionalPPO(
            networks, cfg.get("loss") or {}, storage, device=device, multi_gpu_cfg=cfg.get("multi_gpu"), **alg_cfg
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def train_mode(self) -> None:
        """Train all terms, including loss-only heads; frozen terms remain in eval mode."""
        self.networks.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set every term to evaluation mode for inference."""
        self.networks.eval()
        if self.rnd:
            self.rnd.eval()

    def broadcast_parameters(self) -> None:
        """Broadcast every network once, including separately optimized and auxiliary-only terms."""
        models = [self.networks] + ([self.rnd] if self.rnd else [])
        for model in models:
            for value in model.state_dict().values():
                torch.distributed.broadcast(value, src=0)

    def reduce_parameters(self) -> None:
        """Reduce shared gradients once before any optimizer steps."""
        parameters = list(self.networks.parameters())
        if self.rnd:
            parameters.extend(self.rnd.parameters())
        reduce_gradients_in_buckets(parameters, self.gpu_world_size, self.grad_reduce_bucket_mb)

    def save(self) -> dict:
        """Keep standard checkpoint entries and add auxiliary-only networks and optimizer state."""
        checkpoint = super().save()
        checkpoint.update(
            network_state_dict=self.networks.state_dict(),
            loss_optimizer_state_dict={
                name: optimizer.state_dict() for name, optimizer in self.losses.optimizers.items()
            },
            optimizer_parameter_names=self.optimizer_parameter_names,
            network_inputs=self.networks.inputs,
            frozen_networks=sorted(self.networks.frozen),
            loss_ownership={name: (sorted(term["trains"]), term["mode"]) for name, term in self.losses.terms.items()},
        )
        return checkpoint

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore all registered terms and reject incompatible optimizer ownership on resume."""
        restore_optimizer = load_cfg is None or load_cfg.get("optimizer", False)
        if restore_optimizer:
            expected = self.save()
            for key in ("optimizer_parameter_names", "network_inputs", "frozen_networks", "loss_ownership"):
                if loaded_dict.get(key) != expected[key]:
                    raise ValueError(f"Checkpoint {key} differs; load weights without optimizer state for transfer.")
        if load_cfg is None or load_cfg.get("networks", False):
            self.networks.load_state_dict(loaded_dict["network_state_dict"], strict=strict)
        result = super().load(loaded_dict, load_cfg, strict)
        if restore_optimizer:
            for name, optimizer in self.losses.optimizers.items():
                optimizer.load_state_dict(loaded_dict["loss_optimizer_state_dict"][name])
        return result

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_loss_terms = {}
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Preserve supervision before symmetry mutates the minibatch. All backwards precede any step.
            with torch.amp.autocast(
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                joint_losses, separate_losses, loss_metrics = self.losses.compute(batch)

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # Recompute actions log prob and entropy for current batch of transitions
                # Note: We need to do this because we updated the policy with new parameters
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
                distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy[:original_batch_size]

                # Compute KL divergence and adapt the learning rate
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                        kl_mean = torch.mean(kl)

                        # Reduce the KL divergence across all GPUs
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        # Update the learning rate only on the main process
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        # Update the learning rate for all GPUs
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        # Update the learning rate for all parameter groups
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
                surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
                surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                for joint_loss in joint_losses:
                    loss = loss + joint_loss

                # RND loss
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

                # Symmetry loss
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                    if self.symmetry.use_mirror_loss:
                        loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.networks.zero_grad(set_to_none=True)
            loss.backward()
            for separate_loss in separate_losses.values():
                separate_loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            for optimizer in self.losses.optimizers.values():
                parameters = [p for group in optimizer.param_groups for p in group["params"]]
                nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()
            for optimizer in self.losses.optimizers.values():
                optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            for name, value in loss_metrics.items():
                mean_loss_terms[name] = mean_loss_terms.get(name, 0.0) + value
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Update the normalizers
        obs = self.storage.observations.flatten(0, 1)
        self.networks.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        loss_dict.update({f"loss/{name}": value / num_updates for name, value in mean_loss_terms.items()})

        # Clear the storage
        self.storage.clear()

        return loss_dict
