# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loss evaluation, optimizer ownership, and loss configurations."""

from .loss import LossManager
from .loss_cfg import DistillationLossCfg, LossesCfg, SupervisedLossCfg

__all__ = ["DistillationLossCfg", "LossManager", "LossesCfg", "SupervisedLossCfg"]
