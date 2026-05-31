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


# =========================================================================
#  CADTR Mission-Critical Utility Model
#  Added: Step 1 of CADTR migration
# =========================================================================

@dataclass(frozen=True)
class CADTR_Mission_Utility:
    """
    Physical Utility Model for Context-Aware Dynamic Tool Resolution.

    Additive, deterministic utility grounded in mission-critical semantics:
      - Tactical Tool  (Fast / Tier 0):   Base Survival Utility  (1.0)
      - Strategic Tool (Slow / Tier J-1): Base Survival + Intelligence Bonus  (1.0 + 0.5)
      - Missed Deadline:                  0.0  (Catastrophic Failure)

    A very small noise term (utility_noise_std_f64) prevents perfectly flat
    graphs while keeping the utility model essentially deterministic.
    """

    cfg_utility_config : Utility_Config
    rng_opt : Optional[np.random.Generator] = None

    def __post_init__(self) -> None:
        if self.rng_opt is None:
            object.__setattr__(self, "rng_opt", np.random.default_rng())

    def _Sample_Normal_Clipped(
        self,
        mean_f64: float,
        std_f64: float,
        min_f64: float,
        max_f64: float,
    ) -> float:
        if std_f64 <= 0.0:
            val_f64 = float(mean_f64)
        else:
            val_f64 = float(self.rng_opt.normal(loc=mean_f64, scale=std_f64))
        return float(max(min_f64, min(val_f64, max_f64)))

    def _Sample_Success_Utility(self, task: Task) -> float:
        # Return cached utility if already computed for this task
        cached_val = task.meta_dict_obj.get("sampled_utility_f64")
        if isinstance(cached_val, (int, float)):
            return float(cached_val)

        # Check for premium priority
        is_premium = bool(getattr(task, "high_priority_bool", False))
        multiplier = (
            float(self.cfg_utility_config.high_priority_multiplier_f64)
            if is_premium
            else 1.0
        )

        # Base Survival — granted to ALL successful tasks regardless of tool
        expected_utility = float(self.cfg_utility_config.base_survival_utility_f64) * multiplier

        # Strategic Intelligence Bonus — granted ONLY if the Strategic tool was used
        # Mode.SLOW = Strategic Tool; Mode.FAST = Tactical Tool
        is_strategic = (task.chosen_mode_mode_opt == Mode.SLOW)
        if is_strategic:
            expected_utility += (
                float(self.cfg_utility_config.strategic_logging_bonus_f64) * multiplier
            )

        # Add slight organic noise
        val_f64 = self._Sample_Normal_Clipped(
            mean_f64=expected_utility,
            std_f64=float(self.cfg_utility_config.utility_noise_std_f64),
            min_f64=float(self.cfg_utility_config.min_utility_f64),
            max_f64=float(self.cfg_utility_config.max_utility_f64),
        )

        task.meta_dict_obj["sampled_utility_f64"] = float(val_f64)
        return float(val_f64)

    def Utility(self, task: Task) -> float:
        """Evaluate realized utility under firm physical deadlines."""
        # Catastrophic failure: dropped in queue
        if task.dropped_in_queue_bool:
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

        # Catastrophic failure: never completed
        if task.completion_time_f64_opt is None:
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

        # Catastrophic failure: completed after deadline
        if task.completion_time_f64_opt > task.deadline:
            return float(self.cfg_utility_config.missed_deadline_utility_f64)

        # Task survived and completed on time
        return self._Sample_Success_Utility(task)

