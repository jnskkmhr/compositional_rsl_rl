# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility import; use compositional_rsl_rl.components in new configurations."""

from compositional_rsl_rl.components.network.attention_map import (
    AttentionMapEncoder,
    AttentionMapNetwork,
)

__all__ = ["AttentionMapEncoder", "AttentionMapNetwork"]
