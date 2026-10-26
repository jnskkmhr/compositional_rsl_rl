# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loss terms with explicit gradient ownership and optimizer selection."""

from __future__ import annotations

import copy
import math

import torch
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict
from torch.nn import functional

from compositional_rsl_rl.network import NetworkManager


class LossManager:
    """Resolve supervised loss terms and disjoint separate-optimizer ownership."""

    def __init__(
        self,
        networks: NetworkManager,
        cfg: dict,
        obs_keys: set[str],
        *,
        joint_phases: tuple[str, ...] = ("joint",),
    ) -> None:
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
        for name, term_cfg in self.cfg.items():
            term = copy.deepcopy(term_cfg)
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
            if mode not in (*joint_phases, "separate"):
                raise ValueError(f"Unknown optimizer mode '{mode}' for '{name}'.")
            if mode in joint_phases and learning_rate is not None:
                raise ValueError("Joint loss learning_rate comes from its algorithm optimizer.")
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
            self.terms[name] = {
                "trains": set(trains),
                "prediction": prediction,
                "target": target,
                "mode": mode,
                "weight": weight,
                "func": func,
                "params": params,
            }
        for name, term in self.terms.items():
            if term["mode"] in joint_phases and term["trains"] & self.owners.keys():
                raise ValueError(f"Joint loss '{name}' selects a separately owned network.")
        if {"actor", "critic"} & self.owners.keys():
            raise ValueError("Actor and critic heads must remain owned by the algorithm optimizers.")
        networks.separate = set(self.owners)

    def _predict(self, name: str, batch: RolloutStorage.Batch | TensorDict, trains: set[str]) -> torch.Tensor:
        if isinstance(batch, TensorDict):
            return self.networks.predict(name, batch, train_names=trains)
        state = batch.hidden_states[0 if name == "actor" else 1] if name in ("actor", "critic") else None
        return self.networks.predict(name, batch.observations, batch.masks, state, train_names=trains)

    def compute(
        self, batch: RolloutStorage.Batch | TensorDict, phase: str = "joint"
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
        """Evaluate original, unaugmented minibatches and exclude recurrent padding from supervision."""
        observations = batch if isinstance(batch, TensorDict) else batch.observations
        masks = None if isinstance(batch, TensorDict) else batch.masks
        joint = []
        separate = {}
        metrics = {}
        for name, term in self.terms.items():
            if term["weight"] == 0 or term["mode"] not in (phase, "separate"):
                continue
            if phase != "joint" and term["mode"] != phase:
                continue
            prediction = self._predict(term["prediction"], batch, term["trains"])
            if term["target"] not in observations:
                with torch.no_grad():
                    target = self._predict(term["target"], batch, set())
            else:
                target = observations[term["target"]]
                if masks is not None:
                    target = unpad_trajectories(target, masks)
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
            if term["mode"] != "separate":
                joint.append(weighted)
            else:
                separate[name] = weighted
        return joint, separate, metrics
