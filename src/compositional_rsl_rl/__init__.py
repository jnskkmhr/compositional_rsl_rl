# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent network and loss composition for unmodified RSL-RL."""

from .algorithm import CompositionalPPO
from .network import NetworkManager

__all__ = ["CompositionalPPO", "NetworkManager"]
