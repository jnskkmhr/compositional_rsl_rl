# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent configuration for supervised loss terms."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class SupervisedLossCfg:
    """A prediction/target loss with explicit network and optimizer ownership."""

    trains: list[str]
    prediction: str
    target: str
    loss_fn: str | Callable = "mse"
    optimizer: str = "joint"
    learning_rate: float | None = None
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class LossesCfg:
    """Extend with named SupervisedLossCfg fields; None disables a term."""
