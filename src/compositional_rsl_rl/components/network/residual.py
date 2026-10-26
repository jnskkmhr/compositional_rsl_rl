# Copyright (c) 2026 Holiday Robotics
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND BSD-3-Clause

"""Normalized layers from FlashSAC revision 11a276c; see licenses/dependencies/FlashSAC.txt."""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_class, unpad_trajectories


class UnitLinear(nn.Module):
    """Linear projection with unit-length output weight vectors."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the layer."""
        return self.w(x)  # type: ignore[no-any-return]

    def normalize_parameters(self) -> None:
        """Normalize input-feature weights to keep output vectors at unit length.

        Called after initialization and optimizer.step().
        """
        self.w.weight.copy_(functional.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    """Batch normalization with jointly normalized scale and bias."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Evaluate the model using the requested normalization mode."""
        return functional.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        """Normalize scale and bias parameters to sqrt(d)."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    """Root-mean-square normalization with normalized learned scale."""

    def __init__(self, input_dim: int, eps: float = 1e-6) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the layer."""
        return functional.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)

    def normalize_parameters(self) -> None:
        """Project learned parameters to the prescribed normalization."""
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class UnitEmbedding(nn.Module):
    """Batch-normalized input embedding."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Evaluate the model using the requested normalization mode."""
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class UnitResidualBlock(nn.Module):
    """Residual block with normalized weights and batch normalization."""

    def __init__(self, hidden_dim: int, expansion: int = 4) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Evaluate the model using the requested normalization mode."""
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = functional.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = functional.relu(x)
        x = x + residual
        return x


# -------------------------------------
# Ensembled layers for Double Critic
# -------------------------------------


class EnsembleUnitLinear(nn.Module):
    """Independent normalized linear projections evaluated in one batch."""

    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, B, in] @ [N, in, out] -> [N, B, out]
        """Evaluate the layer."""
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        """Project learned parameters to the prescribed normalization."""
        self.weight.copy_(functional.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    """Independent batch statistics for each critic ensemble member."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Evaluate the model using the requested normalization mode."""
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                batch_size = x.shape[1]
                # Cast to float32 for running stats (BatchNorm uses float32 even in AMP)
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (batch_size / (batch_size - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        """Project learned parameters to the prescribed normalization."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    """RMS normalization across features for each ensemble member."""

    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the layer."""
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        """Project learned parameters to the prescribed normalization."""
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleUnitEmbedding(nn.Module):
    """Parallel normalized embeddings for the double critic."""

    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Evaluate the model using the requested normalization mode."""
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleUnitResidualBlock(nn.Module):
    """Parallel normalized residual blocks for the double critic."""

    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4) -> None:
        """Initialize the layer dimensions and parameters."""
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Evaluate the model using the requested normalization mode."""
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = functional.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = functional.relu(x)
        x = x + residual
        return x


class _ResidualFeatures(nn.Module):
    """Residual feature extractor, independent of actor/critic output semantics.

    A single network consumes [batch, input_dim]. An ensemble consumes
    [ensemble, batch, input_dim] and preserves the ensemble dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 2, num_ensembles: int = 1) -> None:
        """Build the input embedding, residual blocks and final feature normalization."""
        super().__init__()
        if hidden_dim < 1 or num_blocks < 0 or num_ensembles < 1:
            raise ValueError("Residual dimensions and ensemble size must be positive; num_blocks must be nonnegative.")
        self.output_dim = hidden_dim
        if num_ensembles == 1:
            self.embedder = UnitEmbedding(input_dim, hidden_dim)
            self.encoder = nn.ModuleList([UnitResidualBlock(hidden_dim) for _ in range(num_blocks)])
            self.post_norm = UnitRMSNorm(hidden_dim)
        else:
            self.embedder = EnsembleUnitEmbedding(num_ensembles, input_dim, hidden_dim)
            self.encoder = nn.ModuleList([
                EnsembleUnitResidualBlock(num_ensembles, hidden_dim) for _ in range(num_blocks)
            ])
            self.post_norm = EnsembleUnitRMSNorm(num_ensembles, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        """Encode observations (and actions when supplied by a Q model)."""
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        return self.post_norm(x)

    def as_inference(self) -> nn.Module:
        """Return an owned backbone using exporter-supported RMS normalization."""
        backbone = copy.deepcopy(self)
        if isinstance(backbone.post_norm, UnitRMSNorm):
            backbone.post_norm = _ExportRMSNorm(backbone.post_norm)
        return backbone


class _ExportRMSNorm(nn.Module):
    """Primitive RMS normalization for inference exporters without fused RMSNorm support."""

    def __init__(self, source: UnitRMSNorm) -> None:
        super().__init__()
        self.register_buffer("weight", source.weight.detach().clone())
        self.eps = source.eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class ResidualNetwork(MLPModel):
    """A named residual feature term, with optional independent ensemble members."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "relu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_blocks: int = 2,
        num_ensembles: int = 1,
        extra_input_dim: int = 0,
    ) -> None:
        """Infer input width; output_dim is the residual feature width."""
        nn.Module.__init__(self)
        if hidden_dims or activation != "relu" or obs_normalization or distribution_cfg is not None:
            raise ValueError(
                "ResidualNetwork uses its own ReLU blocks and normalization; configure readouts separately."
            )
        if extra_input_dim < 0:
            raise ValueError("extra_input_dim must be nonnegative.")
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.extra_input_dim = extra_input_dim
        self.num_ensembles = num_ensembles
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()
        self.distribution = None
        self.mlp = _ResidualFeatures(self.obs_dim + extra_input_dim, output_dim, num_blocks, num_ensembles)

    def forward_with_mode(
        self, obs: TensorDict, training: bool, extra_input: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate features with explicit batch-normalization mode for replay updates."""
        x = self.get_latent(obs, extra_input=extra_input)
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if self.num_ensembles > 1:
            x = x.unsqueeze(0).expand(self.num_ensembles, -1, -1)
        x = self.mlp(x, training)
        if self.num_ensembles > 1:
            x = x.transpose(0, 1)
        return x.reshape(*shape, *x.shape[1:])

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        extra_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce features through the ordinary named-network interface."""
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
            if extra_input is not None:
                extra_input = unpad_trajectories(extra_input, masks)
        return self.forward_with_mode(obs, self.training, extra_input)

    def as_inference(self) -> nn.Module:
        """Convert RMS normalization without changing feature semantics."""
        model = copy.deepcopy(self)
        model.mlp = model.mlp.as_inference()
        return model

    def as_jit(self) -> nn.Module:
        """Export this feature term using flattened observations."""
        return _ResidualExport(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Export this feature term with the standard RSL-RL metadata."""
        return _ResidualExport(self, verbose)


class LinearNetwork(MLPModel):
    """Normalized linear projections into a configured output distribution.

    Vector inputs produce one readout. Inputs shaped [batch, ensemble, features]
    produce independent readouts per ensemble member. No residual layers are used.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        extra_input_dim: int = 0,
    ) -> None:
        """Construct only output projections; hidden feature networks are separate terms."""
        nn.Module.__init__(self)
        if hidden_dims or obs_normalization:
            raise ValueError("LinearNetwork requires hidden_dims=[] and obs_normalization=False.")
        self.obs_groups = obs_groups[obs_set]
        shapes = [obs[key].shape[1:] for key in self.obs_groups]
        if any(len(shape) not in (1, 2) or shape[:-1] != shapes[0][:-1] for shape in shapes):
            raise ValueError("LinearNetwork inputs must have matching vector or ensemble feature shapes.")
        if extra_input_dim < 0:
            raise ValueError("extra_input_dim must be nonnegative.")
        self.obs_dim = sum(shape[-1] for shape in shapes)
        self.extra_input_dim = extra_input_dim
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()
        self.distribution = None
        if distribution_cfg is not None:
            dist_class, kwargs = resolve_class(distribution_cfg)
            self.distribution = dist_class(output_dim, **kwargs)
        shape = self.distribution.input_dim if self.distribution is not None else output_dim
        ensemble_size = shapes[0][0] if len(shapes[0]) == 2 else 1
        self.mlp = _LinearProjection(self.obs_dim + extra_input_dim, shape, ensemble_size)


class _LinearProjection(nn.Module):
    """Independent normalized projections for each output parameter and ensemble member."""

    def __init__(self, input_dim: int, output_dim: int | list[int], num_ensembles: int) -> None:
        super().__init__()
        self.num_ensembles = num_ensembles
        self.structured_output = not isinstance(output_dim, int)
        count, width = (1, output_dim) if isinstance(output_dim, int) else output_dim
        self.layers = nn.ModuleList()
        self.biases = nn.ParameterList()
        for _ in range(count):
            if num_ensembles == 1:
                self.layers.append(UnitLinear(input_dim, width))
                self.biases.append(nn.Parameter(torch.zeros(width)))
            else:
                self.layers.append(EnsembleUnitLinear(num_ensembles, input_dim, width))
                self.biases.append(nn.Parameter(torch.zeros(num_ensembles, width)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_ensembles > 1:
            x = x.transpose(0, 1)
        outputs = []
        for layer, bias in zip(self.layers, self.biases):
            if self.num_ensembles == 1:
                # Fused linear+bias preserves the reference AMP projection arithmetic.
                outputs.append(functional.linear(x, layer.w.weight, bias))
            else:
                outputs.append(layer(x) + bias.unsqueeze(1))
        value = torch.stack(outputs, dim=-2) if self.structured_output else outputs[0]
        return value.transpose(0, 1) if self.num_ensembles > 1 else value


class _ResidualExport(nn.Module):
    is_recurrent = False

    def __init__(self, source: ResidualNetwork, verbose: bool = False) -> None:
        super().__init__()
        if source.extra_input_dim:
            raise ValueError("Export action-conditioned features through their Q model.")
        self.mlp = source.mlp.as_inference()
        self.num_ensembles = source.num_ensembles
        self.input_size = source.obs_dim
        self.verbose = verbose
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_ensembles > 1:
            x = x.unsqueeze(0).expand(self.num_ensembles, -1, -1)
        x = self.mlp(x, False)
        return x.transpose(0, 1) if self.num_ensembles > 1 else x

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["features"]
