# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Models/Utility.py
#  Purpose: Implement utility models for task outcomes.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from Configurations import Utility_Config
from Core.Task import Mode, Task


class Utility_Model(Protocol):
    def Utility(self, task: Task) -> float:
        ...


@dataclass(frozen=True)
class Firm_Deadline_Quality_Utility:

    cfg_utility_config : Utility_Config
    rng_opt : Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng_opt is None:
            self.rng_opt = np.random.default_rng()

    def _Sample_Normal_Clipped(
        self,
        mean_f64: float,
        std_f64: float,
        min_f64_opt: Optional[float],
        max_f64_opt: Optional[float],
    ) -> float:
        if std_f64 <= 0:
            val_f64 = float(mean_f64)
        else:
            val_f64 = float(mean_f64)
            for _ in range(50):
                draw = float(self.rng_opt.normal(loc=mean_f64, scale=std_f64))
                if min_f64_opt is not None and draw < float(min_f64_opt):
                    continue
                if max_f64_opt is not None and draw > float(max_f64_opt):
                    continue
                val_f64 = draw
                break

        if min_f64_opt is not None:
            val_f64 = max(val_f64, float(min_f64_opt))
        if max_f64_opt is not None:
            val_f64 = min(val_f64, float(max_f64_opt))

        if val_f64 < 0.0:
            val_f64 = 0.0

        return float(val_f64)

    def _Sample_Success_Utility(self, task: Task) -> float:
        cached_val = task.meta_dict_obj.get("sampled_utility_f64")
        if isinstance(cached_val, (int, float)):
            return float(cached_val)

        mode_mode_opt = task.chosen_mode_mode_opt
        is_high_priority = bool(getattr(task, "high_priority_bool", False))
        if mode_mode_opt == Mode.SLOW:
            mean_f64 = (
                float(self.cfg_utility_config.high_priority_slow_success_utility_f64)
                if is_high_priority
                else float(self.cfg_utility_config.slow_success_utility_f64)
            )
            val_f64 = self._Sample_Normal_Clipped(
                mean_f64=mean_f64,
                std_f64=float(self.cfg_utility_config.slow_success_std_f64),
                min_f64_opt=self.cfg_utility_config.slow_success_min_f64,
                max_f64_opt=self.cfg_utility_config.slow_success_max_f64,
            )
        elif mode_mode_opt == Mode.FAST:
            mean_f64 = (
                float(self.cfg_utility_config.high_priority_fast_success_utility_f64)
                if is_high_priority
                else float(self.cfg_utility_config.fast_success_utility_f64)
            )
            val_f64 = self._Sample_Normal_Clipped(
                mean_f64=mean_f64,
                std_f64=float(self.cfg_utility_config.fast_success_std_f64),
                min_f64_opt=self.cfg_utility_config.fast_success_min_f64,
                max_f64_opt=self.cfg_utility_config.fast_success_max_f64,
            )
        else:
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

        task.meta_dict_obj["sampled_utility_f64"] = float(val_f64)
        return float(val_f64)

    def Utility(self, task: Task) -> float:
        if task.dropped_in_queue_bool:
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

        if task.completion_time_f64_opt is None:
                                                                  
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

        on_time_bool = task.completion_time_f64_opt <= task.deadline
        if not on_time_bool and self.cfg_utility_config.firm_deadline_bool:
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

                                                              
        return self._Sample_Success_Utility(task)
