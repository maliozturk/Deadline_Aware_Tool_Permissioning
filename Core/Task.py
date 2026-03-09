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
from typing import Optional


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

    def Mark_Started(self, now_f64: float, mode_mode: Mode) -> None:
        if self.start_service_time_f64_opt is None:
            self.start_service_time_f64_opt = now_f64
        self.chosen_mode_mode_opt = mode_mode

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
