"""
This script produces:

- Queue density over simulation time (time series plot of queue length)

- Average tool handling times (mean/percentiles of service time by mode: slow/fast)

- Server idle anomaly detection (time server was idle while queue was non-empty)

- Full summary statistics (your full list, printed consistently)

- It is also generic/extensible: Metrics are implemented as “plugins” that consume
  a shared AnalysisContext. Adding a new metric is adding one new class implementing
  Compute().
"""

from __future__ import annotations

import os.path
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Core.Task import Mode, Task
from Metrics.Collector import SummaryStats


# -----------------------------
# Data contract between sim and Analysis
# -----------------------------

@dataclass(frozen=True)
class Timeline_Series:
    """
    Generic time-series container.
    times: strictly increasing time points
    values: same length as times
    """

    times  : Sequence[float]
    values : Sequence[float]

    def As_Arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        t_arr_f64 = np.asarray(self.times, dtype=float)
        v_arr_f64 = np.asarray(self.values, dtype=float)
        if t_arr_f64.size != v_arr_f64.size:
            raise ValueError("TimelineSeries times/values length mismatch.")
        return t_arr_f64, v_arr_f64


@dataclass(frozen=True)
class Simulation_Artifacts:
    """
    Minimal artifacts needed for Analysis.

    Required:
      - aggregate: headline Metrics dict from MetricsCollector.Aggregate()
      - completed/dropped/unfinished: task lists from MetricsCollector

    Optional (recommended for deeper Analysis):
      - queue_length_timeline: sampled queue length over time
      - server_busy_timeline: sampled server busy flag over time (1=busy, 0=idle)
      - queue_nonempty_timeline: sampled queue non-empty flag over time (1 if qlen>0 else 0)
    """

    aggregate        : Dict[str, object]
    completed_tasks  : Sequence[Task]
    dropped_tasks    : Sequence[Task]
    unfinished_tasks : Sequence[Task]

    queue_length_timeline   : Optional[Timeline_Series] = None
    server_busy_timeline    : Optional[Timeline_Series] = None
    queue_nonempty_timeline : Optional[Timeline_Series] = None
    ewma_slow_timeline      : Optional[Timeline_Series] = None
    ewma_fast_timeline      : Optional[Timeline_Series] = None
    ewma_slow_count_timeline: Optional[Timeline_Series] = None
    ewma_fast_count_timeline: Optional[Timeline_Series] = None
    epsilon_timeline        : Optional[Timeline_Series] = None


@dataclass
class Analysis_Context:
    """
    Shared context passed to all Metrics/plugins.
    """

    artifacts : Simulation_Artifacts

    def All_Tasks(self) -> List[Task]:
        completed_tasks_list_task  = list(self.artifacts.completed_tasks)
        dropped_tasks_list_task    = list(self.artifacts.dropped_tasks)
        unfinished_tasks_list_task = list(self.artifacts.unfinished_tasks)
        return completed_tasks_list_task + dropped_tasks_list_task + unfinished_tasks_list_task


# -----------------------------
# Plugin interface (extensible Metrics)
# -----------------------------

class Metric_Plugin(Protocol):
    """
    Each plugin can compute numbers and/or emit plots.
    """

    name : str

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        ...

    def Plot(self, ctx_analysis_context: Analysis_Context) -> None:
        """
        Optional plotting hook. If not needed, return without plotting.
        """
        return


# -----------------------------
# Helpers
# -----------------------------

def _Summary_Stats(samples_seq_f64: Sequence[float]) -> SummaryStats:
    return SummaryStats.From_Samples([float(x_f64) for x_f64 in samples_seq_f64])


def _Nan_If_Empty(x_seq_f64: Sequence[float]) -> float:
    return float(np.mean(x_seq_f64)) if x_seq_f64 else float("nan")


# -----------------------------
# Built-in plugins
# -----------------------------

class Queue_Density_Plot_Plugin:
    name = "queue_density_plot"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        tl_timeline_series_opt = ctx_analysis_context.artifacts.queue_length_timeline
        if tl_timeline_series_opt is None:
            return {"queue_density_plot_available": False}

        t_arr_f64, q_arr_f64 = tl_timeline_series_opt.As_Arrays()

        return {
            "queue_density_plot_available": True,
            "queue_length_samples": int(q_arr_f64.size),
            "queue_length_mean_sampled": float(np.mean(q_arr_f64)) if q_arr_f64.size else float("nan"),
            "queue_length_p90_sampled": float(np.percentile(q_arr_f64, 90)) if q_arr_f64.size else float("nan"),
        }

    @staticmethod
    def Plot(ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        tl_timeline_series_opt = ctx_analysis_context.artifacts.queue_length_timeline
        if tl_timeline_series_opt is None:
            return

        t_arr_f64, q_arr_f64 = tl_timeline_series_opt.As_Arrays()
        if t_arr_f64.size == 0:
            return

        plt.figure()
        plt.plot(t_arr_f64, q_arr_f64)
        plt.xlabel("Simulation time")
        plt.ylabel("Queue length")
        plt.title("Queue density over time (sampled queue length)")
        plt.tight_layout()
        now = str(time.time()).split(".")[0]
        save_path = os.path.join(results_path, f"queue_density_plot_t{now}.png")
        plt.savefig(save_path)
        plt.close()



class Average_Handling_Times_By_Mode_Plugin:
    """
    Computes average / percentiles of service times by chosen mode, on completed tasks.

    Notes:
      - Uses Task.service_time_f64_opt (which in your simulator is set for service).
      - If switching is enabled, simulator sets service_time_f64_opt to consumed+remaining (total).
    """

    name = "avg_handling_times_by_mode"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        slow_times_list_f64    : List[float] = []
        fast_times_list_f64    : List[float] = []
        unknown_times_list_f64 : List[float] = []

        for task_ in ctx_analysis_context.artifacts.completed_tasks:
            if task_.service_time_f64_opt is None:
                continue

            if task_.chosen_mode_mode_opt == Mode.SLOW:
                slow_times_list_f64.append(float(task_.service_time_f64_opt))
            elif task_.chosen_mode_mode_opt == Mode.FAST:
                fast_times_list_f64.append(float(task_.service_time_f64_opt))
            else:
                unknown_times_list_f64.append(float(task_.service_time_f64_opt))

        return {
            "service_time_slow": _Summary_Stats(slow_times_list_f64),
            "service_time_fast": _Summary_Stats(fast_times_list_f64),
            "service_time_unknown": _Summary_Stats(unknown_times_list_f64),
        }

    def Plot(self, ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        # Optional: histogram per mode (kept minimal; can be extended later)
        slow_list_f64 = [
            float(task_.service_time_f64_opt)
            for task_ in ctx_analysis_context.artifacts.completed_tasks
            if task_.chosen_mode_mode_opt == Mode.SLOW and task_.service_time_f64_opt is not None
        ]
        fast_list_f64 = [
            float(task_.service_time_f64_opt)
            for task_ in ctx_analysis_context.artifacts.completed_tasks
            if task_.chosen_mode_mode_opt == Mode.FAST and task_.service_time_f64_opt is not None
        ]

        if not slow_list_f64 and not fast_list_f64:
            return

        if slow_list_f64:
            plt.figure()
            plt.hist(np.asarray(slow_list_f64, dtype=float), bins=30)
            plt.xlabel("Service time")
            plt.ylabel("Count")
            plt.title("Service time distribution (SLOW, completed)")
            plt.tight_layout()
            now = str(time.time()).split(".")[0]
            save_path = os.path.join(results_path, f"service_time_dist_t{now}.png")
            plt.savefig(save_path)
            plt.close()

        if fast_list_f64:
            plt.figure()
            plt.hist(np.asarray(fast_list_f64, dtype=float), bins=30)
            plt.xlabel("Service time")
            plt.ylabel("Count")
            plt.title("Service time distribution (FAST, completed)")
            plt.tight_layout()
            now = str(time.time()).split(".")[0]
            save_path = os.path.join(results_path, f"service_time_dist_t{now}.png")
            plt.savefig(save_path)
            plt.close()


class EWMA_Estimates_Timeline_Plugin:
    """
    Tracks EWMA service-time estimates over time (slow/fast).
    """

    name = "ewma_estimates_timeline"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        slow_tl_opt = ctx_analysis_context.artifacts.ewma_slow_timeline
        fast_tl_opt = ctx_analysis_context.artifacts.ewma_fast_timeline

        if slow_tl_opt is None or fast_tl_opt is None:
            return {"ewma_timeline_available": False}

        t_slow, v_slow = slow_tl_opt.As_Arrays()
        t_fast, v_fast = fast_tl_opt.As_Arrays()

        return {
            "ewma_timeline_available": True,
            "samples_slow": int(v_slow.size),
            "samples_fast": int(v_fast.size),
            "last_slow_ewma": float(v_slow[-1]) if v_slow.size else float("nan"),
            "last_fast_ewma": float(v_fast[-1]) if v_fast.size else float("nan"),
        }

    def Plot(self, ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        slow_tl_opt = ctx_analysis_context.artifacts.ewma_slow_timeline
        fast_tl_opt = ctx_analysis_context.artifacts.ewma_fast_timeline

        if slow_tl_opt is None or fast_tl_opt is None:
            return

        t_slow, v_slow = slow_tl_opt.As_Arrays()
        t_fast, v_fast = fast_tl_opt.As_Arrays()

        if t_slow.size == 0 and t_fast.size == 0:
            return

        plt.figure()
        if t_slow.size:
            plt.plot(t_slow, v_slow, label="EWMA slow")
        if t_fast.size:
            plt.plot(t_fast, v_fast, label="EWMA fast")
        plt.xlabel("Simulation time")
        plt.ylabel("EWMA service time")
        plt.title("EWMA service-time estimates over time")
        plt.legend()
        plt.tight_layout()
        now = str(time.time()).split(".")[0]
        save_path = os.path.join(results_path, f"ewma_estimates_t{now}.png")
        plt.savefig(save_path)
        plt.close()


class EWMA_Warmup_Progress_Timeline_Plugin:
    """
    Tracks EWMA sample counts over time (warmup progress).
    """

    name = "ewma_warmup_progress_timeline"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        slow_tl_opt = ctx_analysis_context.artifacts.ewma_slow_count_timeline
        fast_tl_opt = ctx_analysis_context.artifacts.ewma_fast_count_timeline

        if slow_tl_opt is None or fast_tl_opt is None:
            return {"warmup_timeline_available": False}

        _, v_slow = slow_tl_opt.As_Arrays()
        _, v_fast = fast_tl_opt.As_Arrays()

        return {
            "warmup_timeline_available": True,
            "slow_count_last": int(v_slow[-1]) if v_slow.size else 0,
            "fast_count_last": int(v_fast[-1]) if v_fast.size else 0,
        }

    def Plot(self, ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        slow_tl_opt = ctx_analysis_context.artifacts.ewma_slow_count_timeline
        fast_tl_opt = ctx_analysis_context.artifacts.ewma_fast_count_timeline

        if slow_tl_opt is None or fast_tl_opt is None:
            return

        t_slow, v_slow = slow_tl_opt.As_Arrays()
        t_fast, v_fast = fast_tl_opt.As_Arrays()

        if t_slow.size == 0 and t_fast.size == 0:
            return

        plt.figure()
        if t_slow.size:
            plt.plot(t_slow, v_slow, label="EWMA slow count")
        if t_fast.size:
            plt.plot(t_fast, v_fast, label="EWMA fast count")
        plt.xlabel("Simulation time")
        plt.ylabel("EWMA sample count")
        plt.title("EWMA warmup progress (sample counts)")
        plt.legend()
        plt.tight_layout()
        now = str(time.time()).split(".")[0]
        save_path = os.path.join(results_path, f"ewma_warmup_progress_t{now}.png")
        plt.savefig(save_path)
        plt.close()


class Epsilon_Adaptation_Timeline_Plugin:
    """
    Tracks adaptive epsilon values over time when enabled.
    """

    name = "epsilon_adaptation_timeline"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        eps_tl_opt = ctx_analysis_context.artifacts.epsilon_timeline
        if eps_tl_opt is None:
            return {"epsilon_timeline_available": False}

        _, v_eps = eps_tl_opt.As_Arrays()
        return {
            "epsilon_timeline_available": True,
            "samples": int(v_eps.size),
            "last_epsilon": float(v_eps[-1]) if v_eps.size else float("nan"),
        }

    def Plot(self, ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        eps_tl_opt = ctx_analysis_context.artifacts.epsilon_timeline
        if eps_tl_opt is None:
            return

        t_eps, v_eps = eps_tl_opt.As_Arrays()
        if t_eps.size == 0:
            return

        plt.figure()
        plt.plot(t_eps, v_eps, label="epsilon")
        plt.xlabel("Simulation time")
        plt.ylabel("epsilon")
        plt.title("Adaptive epsilon over time")
        plt.tight_layout()
        now = str(time.time()).split(".")[0]
        save_path = os.path.join(results_path, f"adaptive_epsilon_t{now}.png")
        plt.xlim([50000, 51000])
        plt.savefig(save_path)
        plt.close()


class Server_Idle_Anomaly_Plugin:
    """
    Detects whether the server was idle while the queue was non-empty.

    Requires server_busy_timeline and queue_nonempty_timeline sampled on the same event times.
    If not available, returns "not available".

    Metric:
      - idle_with_backlog_time: total time duration where (server_busy==0 and queue_nonempty==1)
      - idle_with_backlog_fraction_of_horizon: normalized by total observed horizon in the timelines
    """

    name = "server_idle_anomaly"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        busy_tl_timeline_series_opt     = ctx_analysis_context.artifacts.server_busy_timeline
        nonempty_tl_timeline_series_opt = ctx_analysis_context.artifacts.queue_nonempty_timeline

        if busy_tl_timeline_series_opt is None or nonempty_tl_timeline_series_opt is None:
            return {
                "server_idle_anomaly_available": False,
                "idle_with_backlog_time": float("nan"),
                "idle_with_backlog_fraction_of_horizon": float("nan"),
            }

        t_busy_arr_f64, busy_arr_f64         = busy_tl_timeline_series_opt.As_Arrays()
        t_nonempty_arr_f64, nonempty_arr_f64 = nonempty_tl_timeline_series_opt.As_Arrays()

        if t_busy_arr_f64.size == 0 or t_nonempty_arr_f64.size == 0:
            return {
                "server_idle_anomaly_available": True,
                "idle_with_backlog_time": 0.0,
                "idle_with_backlog_fraction_of_horizon": 0.0,
            }

        # Expect same sampling times; if not, fall back to intersection via last-observation-carried-forward.
        if t_busy_arr_f64.size != t_nonempty_arr_f64.size or np.any(t_busy_arr_f64 != t_nonempty_arr_f64):
            t_arr_f64 = np.unique(np.concatenate([t_busy_arr_f64, t_nonempty_arr_f64]))
            busy_aligned_arr_f64 = _Locf_Resample(t_busy_arr_f64, busy_arr_f64, t_arr_f64)
            nonempty_aligned_arr_f64 = _Locf_Resample(t_nonempty_arr_f64, nonempty_arr_f64, t_arr_f64)
        else:
            t_arr_f64 = t_busy_arr_f64
            busy_aligned_arr_f64 = busy_arr_f64
            nonempty_aligned_arr_f64 = nonempty_arr_f64

        if t_arr_f64.size < 2:
            return {
                "server_idle_anomaly_available": True,
                "idle_with_backlog_time": 0.0,
                "idle_with_backlog_fraction_of_horizon": 0.0,
            }

        durations_arr_f64 = np.diff(t_arr_f64)
        idle_with_backlog_arr_f64 = (
            (busy_aligned_arr_f64[:-1] <= 0.5) & (nonempty_aligned_arr_f64[:-1] >= 0.5)
        ).astype(float)

        idle_time_f64 = float(np.sum(durations_arr_f64 * idle_with_backlog_arr_f64))
        horizon_f64 = float(t_arr_f64[-1] - t_arr_f64[0])
        frac_f64 = (idle_time_f64 / horizon_f64) if horizon_f64 > 0 else 0.0

        return {
            "server_idle_anomaly_available": True,
            "idle_with_backlog_time": idle_time_f64,
            "idle_with_backlog_fraction_of_horizon": frac_f64,
        }

    def Plot(self, ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        return


def _Locf_Resample(src_t_arr_f64: np.ndarray, src_v_arr_f64: np.ndarray, new_t_arr_f64: np.ndarray) -> np.ndarray:
    """
    Last-Observation-Carried-Forward resampling from (src_t, src_v) to new_t.
    Assumes src_t increasing.
    """
    src_t_arr_f64 = np.asarray(src_t_arr_f64, dtype=float)
    src_v_arr_f64 = np.asarray(src_v_arr_f64, dtype=float)
    new_t_arr_f64 = np.asarray(new_t_arr_f64, dtype=float)

    out_arr_f64 = np.zeros_like(new_t_arr_f64, dtype=float)
    j_i32 = 0
    last_f64 = src_v_arr_f64[0]

    for i_i32, ti_f64 in enumerate(new_t_arr_f64):
        while j_i32 + 1 < src_t_arr_f64.size and src_t_arr_f64[j_i32 + 1] <= ti_f64:
            j_i32 += 1
            last_f64 = src_v_arr_f64[j_i32]
        out_arr_f64[i_i32] = last_f64

    return out_arr_f64


class Full_Summary_Stats_Plugin:
    """
    Re-prints (and optionally validates) the Core Metrics your collector aggregates.
    This is intentionally redundant: it ensures Analysis outputs are stable and explicit.
    """

    name = "full_summary_stats"

    def Compute(self, ctx_analysis_context: Analysis_Context) -> Dict[str, object]:
        agg_dict_obj = dict(ctx_analysis_context.artifacts.aggregate)
        agg_dict_obj["n_tasks_observed"] = len(ctx_analysis_context.All_Tasks())
        return agg_dict_obj

    def Plot(self, ctx_analysis_context: Analysis_Context, results_path: str=None) -> None:
        return


# -----------------------------
# Orchestrator
# -----------------------------

class Process_Analyzer:
    """
    Runs a suite of plugins and returns a combined Results dict.
    """

    def __init__(self, plugins_seq_opt: Optional[Sequence[Metric_Plugin]] = None, Results_Path: str = None) -> None:
        self.results_path = Results_Path
        if plugins_seq_opt is None:
            plugins_seq_opt = [
                Queue_Density_Plot_Plugin(),
                Average_Handling_Times_By_Mode_Plugin(),
                EWMA_Estimates_Timeline_Plugin(),
                EWMA_Warmup_Progress_Timeline_Plugin(),
                Epsilon_Adaptation_Timeline_Plugin(),
                Server_Idle_Anomaly_Plugin(),
                Full_Summary_Stats_Plugin(),
            ]
        self.plugins_list_plugin = list(plugins_seq_opt)

    def Analyze(
        self,
        artifacts_simulation_artifacts: Simulation_Artifacts,
        make_plots_bool: bool = True,
    ) -> Dict[str, object]:
        ctx_analysis_context = Analysis_Context(artifacts=artifacts_simulation_artifacts)

        results_dict_obj: Dict[str, object] = {"plugins": [p_plugin.name for p_plugin in self.plugins_list_plugin]}
        for p_plugin in self.plugins_list_plugin:
            out_dict_obj = p_plugin.Compute(ctx_analysis_context)
            results_dict_obj[p_plugin.name] = out_dict_obj
            if make_plots_bool:
                p_plugin.Plot(ctx_analysis_context, self.results_path)

        return results_dict_obj
