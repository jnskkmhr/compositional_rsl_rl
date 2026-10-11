# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateless criteria usable through SupervisedLossCfg.loss_fn."""

import torch
import torch.nn.functional as functional


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return mean cosine distance over final-axis feature vectors (zero vectors have distance one)."""
    return (1.0 - functional.cosine_similarity(prediction, target, dim=-1, eps=eps)).mean()
