# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Core/Task.py
#  Purpose: Define the Task data model and timing helpers.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class Mode(str, Enum):
    SLOW = "slow"
    FAST = "fast"


@dataclass
class Task:

    task_id       : int
    arrival_time  : float
    deadline      : float

    high_priority_bool: bool = False

                             
    chosen_mode_mode_opt: Optional[Mode] = None
    chosen_tier_i32: Optional[int] = None

                        
    start_service_time_f64_opt : Optional[float] = None
    completion_time_f64_opt    : Optional[float] = None
    service_time_f64_opt       : Optional[float] = None                                        

                                                       
    mode_switches_i32        : int   = 0
    service_consumed_f64     : float = 0.0                                                     

    dropped_in_queue_bool    : bool           = False
    drop_time_f64_opt        : Optional[float] = None

    meta_dict_obj: dict = field(default_factory=dict)

    def Ttl_Remaining(self, now_f64: float) -> float:
        return self.deadline - now_f64

    def Is_Expired(self, now_f64: float) -> bool:
        return now_f64 >= self.deadline

    def Mark_Started(self, now_f64: float, mode_mode: Union[Mode, int] = Mode.FAST, num_tiers: int = 2) -> None:
        if self.start_service_time_f64_opt is None:
            self.start_service_time_f64_opt = now_f64
        if isinstance(mode_mode, int):
            self.chosen_tier_i32 = mode_mode
            # Map tier index to Mode for backward compat (J=2: 0→FAST, J-1→SLOW)
            self.chosen_mode_mode_opt = _tier_to_mode(mode_mode, num_tiers)
        else:
            self.chosen_mode_mode_opt = mode_mode
            self.chosen_tier_i32 = _mode_to_tier(mode_mode, num_tiers)

    def Mark_Completed(self, now_f64: float) -> None:
        self.completion_time_f64_opt = now_f64

    def Mark_Dropped(self, now_f64: float) -> None:
        self.dropped_in_queue_bool = True
        self.drop_time_f64_opt = now_f64

    @property
    def Waiting_Time(self) -> Optional[float]:
        if self.start_service_time_f64_opt is None:
            return None
        return self.start_service_time_f64_opt - self.arrival_time

    @property
    def Response_Time(self) -> Optional[float]:
        if self.completion_time_f64_opt is None:
            return None
        return self.completion_time_f64_opt - self.arrival_time

    def Completed_Before_Deadline(self) -> Optional[bool]:
        if self.completion_time_f64_opt is None:
            return None
        return self.completion_time_f64_opt <= self.deadline


def _tier_to_mode(tier_index: int, num_tiers: int = 2) -> Mode:
    """Map tier index to legacy Mode enum. Tier 0 → FAST, tier J-1 → SLOW."""
    if tier_index == 0:
        return Mode.FAST
    if tier_index == num_tiers - 1:
        return Mode.SLOW
    # For intermediate tiers (J>2), default to SLOW for backward compat
    return Mode.SLOW


def _mode_to_tier(mode: Mode, num_tiers: int = 2) -> int:
    """Map legacy Mode enum to tier index. FAST → 0, SLOW → J-1."""
    if mode == Mode.FAST:
        return 0
    return num_tiers - 1
