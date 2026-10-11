# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL adapter for custom encoders of a single structured observation tensor."""

from __future__ import annotations

import copy
import math
import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState


class TensorNetwork(MLPModel):
    """Implement build_encoder to turn [batch, *input_shape] into [batch, feature_dim].

    The adapter supplies an MLP readout, distributions, arbitrary rollout batch
    dimensions, and flattened-input export. Raw tensor values are not normalized: an
    encoder defines their meaning, including any validity channels. Combine its output
    with vector observations through ordinary graph input references.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        feature_dim: int = 128,
        hidden_dims: tuple[int, ...] | list[int] = (128,),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Infer the structured input shape and construct the encoder and readout."""
        if len(obs_groups[obs_set]) != 1:
            raise ValueError("TensorNetwork requires exactly one input; compose vector inputs in another term.")
        if obs_normalization:
            raise ValueError("TensorNetwork does not normalize raw structured inputs; normalize inside your encoder.")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if not hidden_dims:
            raise ValueError("The RSL-RL MLP readout requires at least one hidden layer.")
        self.input_group = obs_groups[obs_set][0]
        self.input_shape = tuple(obs[self.input_group].shape[len(obs.batch_size) :])
        if not self.input_shape or any(size < 1 for size in self.input_shape):
            raise ValueError("TensorNetwork requires a nonempty input tensor.")
        self.feature_dim = feature_dim
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        self.encoder = self.build_encoder(self.input_shape, feature_dim, activation)

    def build_encoder(self, input_shape: tuple[int, ...], feature_dim: int, activation: str) -> nn.Module:
        """Return an encoder mapping a batch of structured inputs to feature vectors."""
        raise NotImplementedError

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        return obs_groups[obs_set], math.prod(self.input_shape)

    def _get_latent_dim(self) -> int:
        return self.feature_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Encode each observation independently, including padded recurrent batches."""
        inputs = obs[self.input_group].reshape(-1, *self.input_shape)
        features = self.encoder(inputs)
        return features.reshape(*obs.batch_size, self.feature_dim)

    def as_jit(self) -> nn.Module:
        """Export the encoder and readout using the graph's flattened input contract."""
        return _TensorExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Export deterministic inference with standard RSL-RL input metadata."""
        return _OnnxTensorExport(self, verbose)


class _TensorExport(nn.Module):
    def __init__(self, model: TensorNetwork) -> None:
        super().__init__()
        self.input_shape = list(model.input_shape)
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        shape = [obs.shape[0]]
        shape.extend(self.input_shape)
        return self.deterministic_output(self.mlp(self.encoder(obs.reshape(shape))))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxTensorExport(_TensorExport):
    is_recurrent = False

    def __init__(self, model: TensorNetwork, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = math.prod(model.input_shape)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
