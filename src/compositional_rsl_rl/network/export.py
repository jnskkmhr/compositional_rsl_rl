# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export only the actor dependency graph using upstream term exporters."""

from __future__ import annotations

import math
import torch
from torch import nn

from rsl_rl.models import CNNModel

from .network import NetworkManager


class _ObservationSlice(nn.Module):
    """Recover an observation tensor from the exported flattened input contract."""

    def __init__(self, start: int, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.start = start
        self.end = start + math.prod(shape)
        self.shape = list(shape)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        shape = [obs.shape[0]]
        shape.extend(self.shape)
        return obs[:, self.start : self.end].reshape(shape)


class _VectorTerm(nn.Module):
    """Route vector inputs to a feed-forward or recurrent upstream export."""

    def __init__(self, model: nn.Module, indices: list[int]) -> None:
        super().__init__()
        self.model = model
        self.indices = indices

    def forward(self, values: list[torch.Tensor]) -> torch.Tensor:
        return self.model(torch.cat([values[i].flatten(start_dim=1) for i in self.indices], dim=-1))

    @torch.jit.export
    def reset(self) -> None:
        self.model.reset()


class _ImageTerm(nn.Module):
    """Route vectors and image tensors to the upstream CNN export."""

    __constants__ = ("vector_indices", "image_indices")

    def __init__(self, model: CNNModel, names: list[str]) -> None:
        super().__init__()
        self.model = model.as_jit()
        self.vector_indices = [names.index(group) for group in model.obs_groups]
        self.image_indices = [names.index(group) for group in model.obs_groups_2d]

    def forward(self, values: list[torch.Tensor]) -> torch.Tensor:
        vectors = [values[i] for i in self.vector_indices]
        vector = torch.cat(vectors, dim=-1) if len(vectors) else values[0].new_empty((values[0].shape[0], 0))
        return self.model(vector, [values[i] for i in self.image_indices])

    @torch.jit.export
    def reset(self) -> None:
        self.model.reset()


class _GraphFeatures(nn.Module):
    """Reconstruct raw inputs and evaluate the selected network ancestors."""

    def __init__(self, manager: NetworkManager, head: str, include_head: bool = True) -> None:
        super().__init__()
        self.names = manager.observation_groups(head)
        self.input_size = 0
        slices = []
        for name in self.names:
            shape = manager.obs_shapes[name]
            slices.append(_ObservationSlice(self.input_size, shape))
            self.input_size += math.prod(shape)
        self.slices = nn.ModuleList(slices)
        terms = []
        for name in manager.dependencies(head):
            if name == head and not include_head:
                break
            model = manager.terms[name]
            if isinstance(model, CNNModel):
                terms.append(_ImageTerm(model, self.names))
            else:
                terms.append(_VectorTerm(model.as_jit(), [self.names.index(source) for source in manager.inputs[name]]))
            self.names.append(name)
        self.terms = nn.ModuleList(terms)

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        values = [part(obs) for part in self.slices]
        for term in self.terms:
            values.append(term(values))
        return values

    @torch.jit.export
    def reset(self) -> None:
        for term in self.terms:
            term.reset()


class _TorchGraph(nn.Module):
    """Complete deterministic policy; reset also resets any recurrent head."""

    def __init__(self, manager: NetworkManager, head: str) -> None:
        super().__init__()
        self.features = _GraphFeatures(manager, head)
        self.input_size = self.features.input_size
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.features(obs)[-1]

    @torch.jit.export
    def reset(self) -> None:
        self.features.reset()


class _OnnxGraph(_TorchGraph):
    is_recurrent = False

    def __init__(self, manager: NetworkManager, head: str, verbose: bool) -> None:
        super().__init__(manager, head)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _OnnxRecurrentGraph(nn.Module):
    is_recurrent = True

    def __init__(self, manager: NetworkManager, head: str, verbose: bool) -> None:
        super().__init__()
        self.features = _GraphFeatures(manager, head, include_head=False)
        self.indices = [self.features.names.index(name) for name in manager.inputs[head]]
        self.head = manager.terms[head].as_onnx(verbose)
        self.input_size = self.features.input_size
        self.verbose = verbose
        self.eval()

    def forward(
        self,
        obs: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        values = self.features(obs)
        return self.head(torch.cat([values[i] for i in self.indices], dim=-1), h_in, c_in)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.input_size), *self.head.get_dummy_inputs()[1:])

    @property
    def input_names(self) -> list[str]:
        return self.head.input_names

    @property
    def output_names(self) -> list[str]:
        return self.head.output_names
