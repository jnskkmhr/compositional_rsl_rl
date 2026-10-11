# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional PPO and its simulator-independent configuration."""

from .algorithm import CompositionalPPO
from .algorithm_cfg import CompositionalPPOAlgorithmCfg

__all__ = ["CompositionalPPO", "CompositionalPPOAlgorithmCfg"]
