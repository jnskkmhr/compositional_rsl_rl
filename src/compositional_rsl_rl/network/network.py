# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named network terms, dependency resolution, and RSL-RL policy views."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models import CNNModel, MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_class, unpad_trajectories


def load_pretrained(module: nn.Module, cfg: dict) -> None:
    """Load a complete term including normalizers, using explicit checkpoint key/prefix selection."""
    cfg = copy.deepcopy(cfg)
    state = torch.load(cfg.pop("path"), map_location="cpu", weights_only=True)
    key = cfg.pop("state_dict_key", None)
    if key is not None:
        state = state[key]
    prefix = cfg.pop("prefix", "")
    if prefix:
        state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if not state:
            raise ValueError(f"No pretrained parameters match '{prefix}'.")
    module.load_state_dict(state, **cfg)


class NetworkManager(nn.Module):
    """Own every network exactly once and resolve input dependencies independently of config order.

    Terms use the RSL-RL MLPModel/CNNModel/RNNModel constructor protocol. Recurrent actor and critic heads
    use upstream rollout-state storage. Intermediate terms are feed-forward: shared recurrent state would
    require a different rollout-storage protocol and is rejected explicitly.
    """

    def __init__(
        self,
        obs: TensorDict,
        cfg: dict,
        num_actions: int,
        *,
        head_dims: dict[str, int] | None = None,
        stochastic_actor: bool = True,
        extra_input_dims: dict[str, int] | None = None,
    ) -> None:
        """Validate the graph, infer input dimensions, and construct configured model terms."""
        super().__init__()
        self.cfg = {name: copy.deepcopy(term) for name, term in cfg.items() if term is not None}
        if not {"actor", "critic"}.issubset(self.cfg):
            raise ValueError("networks must define actor and critic terms.")
        self.inputs = {}
        self.frozen = set()
        self.separate = set()
        for name, term in self.cfg.items():
            if not name or "." in name or (name in obs and name not in ("actor", "critic")):
                raise ValueError(f"Invalid or observation-colliding network name '{name}'.")
            inputs = term.get("input")
            if not isinstance(inputs, (list, tuple)) or not inputs or not all(isinstance(x, str) for x in inputs):
                raise ValueError(f"Network '{name}' requires a nonempty input list.")
            self.inputs[name] = list(inputs)
            for source in inputs:
                if source in ("actor", "critic") and source not in obs:
                    raise ValueError("Actor and critic are terminal heads and cannot feed another network.")
                if source not in self.cfg and source not in obs:
                    raise ValueError(f"Unknown input '{source}' for network '{name}'.")
        self.order = []
        visiting = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Network dependency cycle at '{name}'.")
            if name in self.order:
                return
            visiting.add(name)
            for source in self.inputs[name]:
                if source in self.cfg and source not in obs:
                    visit(source)
            visiting.remove(name)
            self.order.append(name)

        for name in self.cfg:
            visit(name)
        self.terms = nn.ModuleDict()
        sample = obs[:2].detach().to("cpu").clone()
        self.obs_shapes = {name: tuple(value.shape[1:]) for name, value in sample.items()}
        for name in self.order:
            term = copy.deepcopy(self.cfg[name])
            term.pop("input")
            frozen = term.pop("frozen", False)
            pretrained = term.pop("pretrained_cfg", None)
            output_dim = term.pop("output_dim", None)
            if name in ("actor", "critic"):
                expected = (head_dims or {"actor": num_actions, "critic": 1})[name]
                if output_dim is not None and output_dim != expected:
                    raise ValueError(f"Network '{name}' output_dim must be {expected} or None.")
                output_dim = expected
            elif output_dim is None:
                raise ValueError(f"Network '{name}' requires output_dim.")
            extra_dim = (extra_input_dims or {}).get(name, 0)
            if extra_dim:
                term["extra_input_dim"] = extra_dim
            model_class, kwargs = resolve_class(term)
            with torch.device(sample.device):
                model = model_class(sample, {name: self.inputs[name]}, name, output_dim, **kwargs)
            if not isinstance(model, MLPModel) and not (head_dims is not None and name in head_dims):
                raise TypeError("Network classes must implement the RSL-RL MLPModel protocol.")
            if model.is_recurrent and name not in ("actor", "critic"):
                raise ValueError("Recurrent intermediate terms are unsupported; use recurrent actor/critic heads.")
            if stochastic_actor and name == "actor" and model.distribution is None:
                raise ValueError("The actor requires distribution_cfg for PPO.")
            if model.distribution is not None and name != "actor" and (stochastic_actor or name != "critic"):
                raise ValueError("Only actor and off-policy critic heads can have an output distribution.")
            if pretrained is not None:
                load_pretrained(model, pretrained)
            if frozen:
                self.frozen.add(name)
                model.requires_grad_(False)
            self.terms[name] = model
            # Only intermediate outputs are needed for downstream dimension inference.
            if name not in ("actor", "critic"):
                model.eval()
                with torch.no_grad():
                    if extra_dim:
                        sample[name] = model(
                            sample, extra_input=torch.zeros(*sample.batch_size, extra_dim, device=sample.device)
                        )
                    else:
                        sample[name] = model(sample)
        self.train()

    def dependencies(self, name: str) -> list[str]:
        """Return this term and its ancestors once, in execution order."""
        required = {name}
        for node in reversed(self.order):
            if node in required:
                required.update(
                    source for source in self.inputs[node] if source in self.terms and source not in self.obs_shapes
                )
        return [node for node in self.order if node in required]

    def observation_groups(self, name: str) -> list[str]:
        """Return only raw observations needed by the selected graph."""
        required = {source for node in self.dependencies(name) for source in self.inputs[node]}
        return [key for key in self.obs_shapes if key in required]

    def _call(
        self,
        name: str,
        obs: TensorDict,
        train_names: set[str] | None = None,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        model = self.terms[name]
        kwargs = {
            "masks": masks,
            "hidden_state": hidden_state,
            "stochastic_output": stochastic_output,
        }
        if train_names is not None and name not in train_names:
            # Detached weights still propagate gradients to selected upstream terms.
            parameters = {key: value.detach() for key, value in model.named_parameters()}
            return torch.func.functional_call(model, parameters, (obs,), kwargs)
        return model(obs, **kwargs)

    def features(
        self,
        name: str,
        obs: TensorDict,
        train_names: set[str] | None = None,
        for_ppo: bool = False,
    ) -> TensorDict:
        """Evaluate feed-forward ancestors, retaining padded batch dimensions for recurrent heads."""
        values = obs.clone(recurse=False)
        for node in self.dependencies(name):
            if node == name:
                break
            output = self._call(node, values.reshape(-1), train_names)
            if for_ppo and node in self.separate:
                output = output.detach()
            values[node] = output.reshape(*obs.batch_size, *output.shape[1:])
        return values

    def predict(
        self,
        name: str,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        train_names: set[str] | None = None,
        for_ppo: bool = False,
    ) -> torch.Tensor:
        """Evaluate one term with optional loss-specific gradient routing."""
        values = self.features(name, obs, train_names, for_ppo)
        model = self.terms[name]
        if model.is_recurrent:
            return self._call(name, values, train_names, masks, hidden_state, stochastic_output)
        # Flatten arbitrary batch dimensions, including padded trajectories, for CNN inputs.
        model_obs = values.reshape(-1) if isinstance(model, CNNModel) else values
        result = self._call(name, model_obs, train_names, stochastic_output=stochastic_output)
        if isinstance(model, CNNModel):
            result = result.reshape(*obs.batch_size, *result.shape[1:])
        # Distribution parameters must have the same layout as the stored actions.
        if masks is not None:
            result = unpad_trajectories(result, masks)
        return result

    def train(self, mode: bool = True) -> NetworkManager:
        """Keep frozen terms and their normalization/dropout layers in evaluation mode."""
        super().train(mode)
        for name in self.frozen:
            self.terms[name].eval()
        return self

    @torch.no_grad()
    def update_normalization(self, obs: TensorDict) -> None:
        """Update each term once after all optimizers have consumed the rollout."""
        values = obs.clone(recurse=False)
        for name in self.order:
            model = self.terms[name]
            if name not in self.frozen:
                model.update_normalization(values)
            if name not in ("actor", "critic"):
                values[name] = model(values)


class GraphModel(MLPModel):
    """An actor/critic view satisfying the unchanged RSL-RL runner and storage APIs."""

    def __init__(self, manager: NetworkManager, name: str) -> None:
        """Register graph dependencies for state/device traversal without copying their parameters."""
        nn.Module.__init__(self)
        object.__setattr__(self, "manager", manager)
        self.name = name
        self.terms = nn.ModuleDict({key: manager.terms[key] for key in manager.dependencies(name)})
        self.is_recurrent = self.terms[name].is_recurrent
        self.obs_groups = manager.observation_groups(name)

    @property
    def head(self) -> MLPModel:
        """The existing RSL-RL head model."""
        return self.terms[self.name]

    @property
    def distribution(self) -> nn.Module | None:
        """Delegate distribution APIs inherited from MLPModel."""
        return self.head.distribution

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Evaluate the selected graph with PPO gradient isolation for separately optimized terms."""
        # Match upstream distribution layout when a feed-forward actor accompanies a recurrent critic.
        if masks is not None and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
            masks = None
        result = self.manager.predict(self.name, obs, masks, hidden_state, stochastic_output, for_ppo=True)
        if stochastic_output:
            self._distribution_batch_shape = result.shape[:-1]
        return result

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Distribution parameters in rollout layout, including CNN heads in recurrent batches."""
        return tuple(
            p.reshape(*self._distribution_batch_shape, p.shape[-1]) for p in self.head.output_distribution_params
        )

    @property
    def output_mean(self) -> torch.Tensor:
        """Action means in rollout layout."""
        return self.head.output_mean.reshape(*self._distribution_batch_shape, -1)

    @property
    def output_std(self) -> torch.Tensor:
        """Action standard deviations in rollout layout."""
        return self.head.output_std.reshape(*self._distribution_batch_shape, -1)

    @property
    def output_entropy(self) -> torch.Tensor:
        """Per-sample entropy in rollout layout."""
        return self.head.output_entropy.reshape(self._distribution_batch_shape)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Align flattened CNN distributions with the original action batch layout."""
        return self.head.get_output_log_prob(outputs.reshape_as(self.head.output_mean)).reshape(
            self._distribution_batch_shape
        )

    def train(self, mode: bool = True) -> GraphModel:
        """Honor frozen terms even when a caller changes mode directly on a policy view."""
        super().train(mode)
        for name, term in self.terms.items():
            if name in self.manager.frozen:
                term.eval()
        return self

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent head state at episode boundaries."""
        self.head.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the head's upstream-compatible recurrent state."""
        return self.head.get_hidden_state()

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent history using the upstream protocol."""
        self.head.detach_hidden_state(dones)

    def update_normalization(self, obs: TensorDict) -> None:
        """Leave normalization to the owning manager so shared terms update only once."""

    def as_jit(self) -> nn.Module:
        """Export the complete dependency graph with the upstream recurrent reset contract."""
        from .export import _TorchGraph

        return _TorchGraph(self.manager, self.name)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Export the complete dependency graph with explicit recurrent state inputs/outputs."""
        from .export import _OnnxGraph, _OnnxRecurrentGraph

        export_class = _OnnxRecurrentGraph if self.is_recurrent else _OnnxGraph
        return export_class(self.manager, self.name, verbose)
