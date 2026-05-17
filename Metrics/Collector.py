# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Metrics/Collector.py
#  Purpose: Collect per-task metrics and aggregate statistics.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from Core.Task import Mode, Task
from Models.Utility import Utility_Model


@dataclass
class SummaryStats:

    count_i32 : int   = 0
    mean_f64  : float = float("nan")
    p50_f64   : float = float("nan")
    p90_f64   : float = float("nan")
    p95_f64   : float = float("nan")
    p99_f64   : float = float("nan")

    @staticmethod
    def From_Samples(samples_list_f64: List[float]) -> "SummaryStats":
        if not samples_list_f64:
            return SummaryStats(count_i32=0)

        arr_f64 = np.asarray(samples_list_f64, dtype=float)

        return SummaryStats(
            count_i32=int(arr_f64.size),
            mean_f64=float(arr_f64.mean()),
            p50_f64=float(np.percentile(arr_f64, 50)),
            p90_f64=float(np.percentile(arr_f64, 90)),
            p95_f64=float(np.percentile(arr_f64, 95)),
            p99_f64=float(np.percentile(arr_f64, 99)),
        )


@dataclass
class MetricsCollector:

    utility_model_utility_model : Utility_Model
    warmup_time_f64             : float = 0.0
    num_tiers_i32               : int   = 2
    paugbeta_list_f64           : List[float] = field(default_factory=lambda: [0.0, 1.0])
    end_time_f64_opt            : Optional[float] = None

                 
    completed_tasks_list_task  : List[Task] = field(default_factory=list)
    dropped_tasks_list_task    : List[Task] = field(default_factory=list)
    unfinished_tasks_list_task : List[Task] = field(default_factory=list)

                                                         
    queue_len_samples_list_i32      : List[int]   = field(default_factory=list)
    queue_len_sample_times_list_f64 : List[float] = field(default_factory=list)

                                     
    server_busy_samples_list_i32      : List[int]   = field(default_factory=list)
    server_busy_sample_times_list_f64 : List[float] = field(default_factory=list)

    queue_nonempty_samples_list_i32      : List[int]   = field(default_factory=list)
    queue_nonempty_sample_times_list_f64 : List[float] = field(default_factory=list)

    ewma_sample_times_list_f64 : List[float] = field(default_factory=list)
    ewma_slow_samples_list_f64 : List[float] = field(default_factory=list)
    ewma_fast_samples_list_f64 : List[float] = field(default_factory=list)
    ewma_slow_counts_list_i32  : List[int]   = field(default_factory=list)
    ewma_fast_counts_list_i32  : List[int]   = field(default_factory=list)

    def Set_End_Time(self, now_f64: float) -> None:
        self.end_time_f64_opt = float(now_f64)

    def Record_Server_Busy(self, now_f64: float, busy_bool: bool) -> None:
        if now_f64 < self.warmup_time_f64:
            return
        self.server_busy_sample_times_list_f64.append(float(now_f64))
        self.server_busy_samples_list_i32.append(1 if busy_bool else 0)

    def Record_Queue_Nonempty(self, now_f64: float, nonempty_bool: bool) -> None:
        if now_f64 < self.warmup_time_f64:
            return
        self.queue_nonempty_sample_times_list_f64.append(float(now_f64))
        self.queue_nonempty_samples_list_i32.append(1 if nonempty_bool else 0)

    def Record_Queue_Length(self, now_f64: float, qlen_i32: int) -> None:
        if now_f64 < self.warmup_time_f64:
            return
        self.queue_len_sample_times_list_f64.append(float(now_f64))
        self.queue_len_samples_list_i32.append(int(qlen_i32))

    def Record_EWMA_Estimates(
        self,
        now_f64: float,
        slow_ewma_f64: float,
        fast_ewma_f64: float,
        slow_count_i32: int,
        fast_count_i32: int,
    ) -> None:
        if now_f64 < self.warmup_time_f64:
            return
        self.ewma_sample_times_list_f64.append(float(now_f64))
        self.ewma_slow_samples_list_f64.append(float(slow_ewma_f64))
        self.ewma_fast_samples_list_f64.append(float(fast_ewma_f64))
        self.ewma_slow_counts_list_i32.append(int(slow_count_i32))
        self.ewma_fast_counts_list_i32.append(int(fast_count_i32))

    def Record_Task_Final(self, task: Task) -> None:
        if task.arrival_time < self.warmup_time_f64:
            return

        if task.dropped_in_queue_bool:
            self.dropped_tasks_list_task.append(task)
        elif task.completion_time_f64_opt is not None:
            self.completed_tasks_list_task.append(task)
        else:
            self.unfinished_tasks_list_task.append(task)

    def Finalize_Unfinished(self, tasks_list_task: List[Task]) -> None:
        for task_ in tasks_list_task:
            self.Record_Task_Final(task_)

    def Aggregate(self) -> Dict[str, object]:
        tasks_all_list_task = (
            self.completed_tasks_list_task
            + self.dropped_tasks_list_task
            + self.unfinished_tasks_list_task
        )
        n_i32 = len(tasks_all_list_task)

                   
        utilities_list_f64 = [self.utility_model_utility_model.Utility(task_) for task_ in tasks_all_list_task]
        mean_utility_f64 = float(np.mean(utilities_list_f64)) if utilities_list_f64 else float("nan")

                                 
        miss_flags_list_i32: List[int] = []
        for task_ in tasks_all_list_task:
            missed = 1
            if task_.dropped_in_queue_bool:
                missed = 1
            elif task_.completion_time_f64_opt is None:
                missed = 1
            else:
                missed = 1 if (task_.completion_time_f64_opt > task_.deadline) else 0
            miss_flags_list_i32.append(int(missed))

                              
        on_time_i32 = 0
        missed_i32  = 0

        for task_ in self.completed_tasks_list_task:
            if (
                task_.completion_time_f64_opt is not None
                and task_.completion_time_f64_opt <= task_.deadline
            ):
                on_time_i32 += 1
            else:
                missed_i32 += 1

                                                                
        missed_i32 += len(self.dropped_tasks_list_task) + len(self.unfinished_tasks_list_task)

        miss_rate_f64 = (missed_i32 / n_i32) if n_i32 > 0 else float("nan")

                                                 
        resp_list_f64 = [
            task_.Response_Time
            for task_ in self.completed_tasks_list_task
            if task_.Response_Time is not None
        ]
        wait_list_f64 = [
            task_.Waiting_Time
            for task_ in self.completed_tasks_list_task
            if task_.Waiting_Time is not None
        ]

        resp_stats_summary_stats = SummaryStats.From_Samples([float(x_f64) for x_f64 in resp_list_f64])
        wait_stats_summary_stats = SummaryStats.From_Samples([float(x_f64) for x_f64 in wait_list_f64])

                    
        mode_counts_dict_obj = {Mode.SLOW.value: 0, Mode.FAST.value: 0, "unknown": 0}
        for task_ in self.completed_tasks_list_task:
            if task_.chosen_mode_mode_opt == Mode.SLOW:
                mode_counts_dict_obj[Mode.SLOW.value] += 1
            elif task_.chosen_mode_mode_opt == Mode.FAST:
                mode_counts_dict_obj[Mode.FAST.value] += 1
            else:
                mode_counts_dict_obj["unknown"] += 1

        # Per-tier accounting (J-mode support)
        J = self.num_tiers_i32
        tier_counts_dict: Dict[int, int] = {}
        for task_ in self.completed_tasks_list_task:
            tier_idx = getattr(task_, "chosen_tier_i32", None)
            if tier_idx is None:
                # Fallback from Mode for backward compat.
                # TODO(Prompt 4+): once tier-indexed routing is the sole path,
                # this fallback should emit a logged warning rather than
                # silently recovering.
                if task_.chosen_mode_mode_opt == Mode.FAST:
                    tier_idx = 0
                elif task_.chosen_mode_mode_opt == Mode.SLOW:
                    tier_idx = J - 1
                else:
                    continue
            tier_counts_dict[tier_idx] = tier_counts_dict.get(tier_idx, 0) + 1
        total_tiered = sum(tier_counts_dict.values())
        tier_fractions_list_f64: List[float] = []
        for j in range(J):
            count_j = tier_counts_dict.get(j, 0)
            tier_fractions_list_f64.append(float(count_j / total_tiered) if total_tiered > 0 else 0.0)

                                         
        qlen_mean_f64 = (
            float(np.mean(self.queue_len_samples_list_i32))
            if self.queue_len_samples_list_i32
            else float("nan")
        )
        qlen_p90_f64 = (
            float(np.percentile(self.queue_len_samples_list_i32, 90))
            if self.queue_len_samples_list_i32
            else float("nan")
        )

                                                 
        end_time_f64 = (
            float(self.end_time_f64_opt)
            if self.end_time_f64_opt is not None
            else float("nan")
        )

        if end_time_f64 == end_time_f64:
            horizon_f64 = float(end_time_f64 - float(self.warmup_time_f64))
        else:
            horizon_f64 = float("nan")

        if horizon_f64 <= 0.0:
            horizon_f64 = float("nan")

        paug_dict_obj: Dict[str, float] = {}
        if horizon_f64 == horizon_f64:
            sum_util_f64 = float(np.sum(utilities_list_f64)) if utilities_list_f64 else 0.0
            sum_miss_i32 = int(np.sum(miss_flags_list_i32)) if miss_flags_list_i32 else 0
            for beta in self.paugbeta_list_f64:
                key = f"paug_beta_{str(beta).replace('.', 'p')}"
                paug_dict_obj[key] = (sum_util_f64 - float(beta) * float(sum_miss_i32)) / horizon_f64

        return {
            "n_tasks_total": n_i32,
            "n_completed": len(self.completed_tasks_list_task),
            "n_dropped_in_queue": len(self.dropped_tasks_list_task),
            "n_unfinished": len(self.unfinished_tasks_list_task),
            "miss_rate": miss_rate_f64,
            "mean_utility": mean_utility_f64,
            "response_time": resp_stats_summary_stats,
            "waiting_time": wait_stats_summary_stats,
            "mode_counts_completed": mode_counts_dict_obj,
            "tier_fractions": tier_fractions_list_f64,
            "queue_len_mean": qlen_mean_f64,
            "queue_len_p90": qlen_p90_f64,
            "paug_horizon": horizon_f64,
            **paug_dict_obj,
        }
