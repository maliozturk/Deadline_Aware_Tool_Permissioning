# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC)
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Core/ToolTier.py
#  Purpose: Define the ToolTier configuration dataclass for J-mode support.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-05-17
#  Environment: Python 3.9.13
# =============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class ToolTier:
    """Configuration for a single tool tier in the J-mode menu.

    Tier 0 is the fastest/lightest, tier J-1 is the slowest/richest.
    """

    name: str
    index: int
    mean_service_time: float
    service_trace: List[float] = field(default_factory=list)
    mean_utility: float = 0.0
    utility_std: float = 0.0
    utility_min: float = 0.0
    utility_max: float = 1.2
