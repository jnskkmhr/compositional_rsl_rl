# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility import; use compositional_ppo for new configurations."""

from .compositional_ppo import CompositionalPPO

__all__ = ["CompositionalPPO"]
