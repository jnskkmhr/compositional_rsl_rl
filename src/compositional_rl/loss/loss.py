# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loss terms with explicit gradient ownership and optimizer selection."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn.functional as functional

from compositional_rl.network import NetworkManager
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, unpad_trajectories


class LossManager:
    """Resolve supervised loss terms and disjoint separate-optimizer ownership."""

    def __init__(self, networks: NetworkManager, cfg: dict, obs_keys: set[str]) -> None:
        """Validate all references before rollout and partition optimizer ownership."""
        self.networks = networks
        self.cfg = {name: copy.deepcopy(term) for name, term in cfg.items() if term is not None}
        self.terms = {}
        self.optimizers = {}
        self.owners = {}
        functions = {
            "mse": functional.mse_loss,
            "huber": functional.huber_loss,
            "bce": functional.binary_cross_entropy,
            "bce_with_logits": functional.binary_cross_entropy_with_logits,
        }
        for name, cfg in self.cfg.items():
            term = copy.deepcopy(cfg)
            weight = float(term.pop("weight", 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Loss '{name}' requires a finite nonnegative weight.")
            trains = term.pop("trains")
            prediction = term.pop("prediction")
            target = term.pop("target")
            if prediction not in networks.terms or target not in obs_keys | set(networks.terms):
                raise ValueError(f"Loss '{name}' has an unknown prediction or target.")
            if not trains or len(set(trains)) != len(trains):
                raise ValueError(f"Loss '{name}' requires a nonempty, unique trains list.")
            ancestors = set(networks.dependencies(prediction))
            if not set(trains).issubset(ancestors) or set(trains) & networks.frozen:
                raise ValueError(f"Loss '{name}' trains must select unfrozen prediction dependencies.")
            mode = term.pop("optimizer", "joint")
            learning_rate = term.pop("learning_rate", None)
            if mode not in ("joint", "separate"):
                raise ValueError(f"Unknown optimizer mode '{mode}' for '{name}'.")
            if mode == "joint" and learning_rate is not None:
                raise ValueError("Joint loss learning_rate comes from the PPO optimizer.")
            if mode == "separate":
                if learning_rate is None or not math.isfinite(learning_rate) or learning_rate <= 0:
                    raise ValueError(f"Separate loss '{name}' requires a positive learning_rate.")
                for network in trains:
                    if network in self.owners:
                        raise ValueError(f"Separate losses cannot share optimizer ownership of '{network}'.")
                    self.owners[network] = name
                parameters = [p for network in trains for p in networks.terms[network].parameters() if p.requires_grad]
                self.optimizers[name] = torch.optim.Adam(parameters, lr=learning_rate)
            fn = term.pop("loss_fn", "mse")
            func = functions[fn] if isinstance(fn, str) and fn in functions else resolve_callable(fn)
            if not callable(func) or isinstance(func, torch.nn.Module):
                raise TypeError(
                    f"Loss '{name}' requires a stateless criterion; register trainable modules as networks."
                )
            params = term.pop("params", {})
            if term:
                raise ValueError(f"Unknown loss options for '{name}': {list(term)}.")
            self.terms[name] = dict(
                trains=set(trains),
                prediction=prediction,
                target=target,
                mode=mode,
                weight=weight,
                func=func,
                params=params,
            )
        for name, term in self.terms.items():
            if term["mode"] == "joint" and term["trains"] & self.owners.keys():
                raise ValueError(f"Joint loss '{name}' selects a separately owned network.")
        if {"actor", "critic"} & self.owners.keys():
            raise ValueError("Actor and critic heads must remain owned by the PPO optimizer.")
        networks.separate = set(self.owners)

    def _predict(self, name: str, batch: RolloutStorage.Batch, trains: set[str]) -> torch.Tensor:
        state = batch.hidden_states[0 if name == "actor" else 1] if name in ("actor", "critic") else None
        return self.networks.predict(name, batch.observations, batch.masks, state, train_names=trains)

    def compute(
        self, batch: RolloutStorage.Batch
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
        """Evaluate original, unaugmented minibatches and exclude recurrent padding from supervision."""
        joint = []
        separate = {}
        metrics = {}
        for name, term in self.terms.items():
            if term["weight"] == 0:
                continue
            prediction = self._predict(term["prediction"], batch, term["trains"])
            if term["target"] not in batch.observations:
                with torch.no_grad():
                    target = self._predict(term["target"], batch, set())
            else:
                target = batch.observations[term["target"]]
                if batch.masks is not None:
                    target = unpad_trajectories(target, batch.masks)
            target = target.detach()
            if prediction.shape != target.shape:
                raise ValueError(f"Loss '{name}' prediction {prediction.shape} and target {target.shape} must match.")
            value = term["func"](prediction, target, **term["params"])
            if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value).item():
                raise ValueError(f"Loss '{name}' must return a finite scalar tensor.")
            if not value.requires_grad:
                raise ValueError(f"Loss '{name}' has no gradient to its selected trains.")
            metrics[name] = value.detach().item()
            weighted = value * term["weight"]
            if term["mode"] == "joint":
                joint.append(weighted)
            else:
                separate[name] = weighted
        return joint, separate, metrics
