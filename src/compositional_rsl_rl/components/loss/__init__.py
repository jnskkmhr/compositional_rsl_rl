# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable stateless supervised criteria."""

from .loss import cosine_distance

__all__ = ["cosine_distance"]
