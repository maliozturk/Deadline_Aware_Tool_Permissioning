from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from Configurations import Simulation_Config
from Core.Simulator import Simulator
from Core.Task import Mode
from Metrics.Collector import MetricsCollector
from Models.Distributions import (
    EWMA_Service_Time_Estimator,
    Exponential_Interarrival,
    Lognormal_Service_Times,
    Trace_Service_Times,
)
from Models.Policies import (
    Baseline_Heuristic_Policy,
    Drift_Penalty_Myopic_Policy,
    FTCPolicy,
    Fcfs_Always_Fast_Policy,
    Fcfs_Always_Slow_Policy,
    Queue_Length_Bang_Bang_Policy,
    Static_Mix_Policy,
)
from Models.Utility import Firm_Deadline_Quality_Utility


POLICY_LABELS: Dict[str, str] = {
    "Fcfs_Always_Fast": "AlwaysFast (AF)",
    "Fcfs_Always_Slow": "AlwaysSlow (AS)",
    "Static_Mix_Policy": "StaticMix (p=0.5)",
    "Queue_Length_Bang_Bang_Policy": "Q-BB",
    "Drift_Penalty_Myopic_Policy": "DriftPenaltyMyopic (V=1.0)",
    "Baseline_Heuristic_Policy": "TTL-aware heuristic",
    "FTCPolicy": "FTC*",
    "DATPPolicy": "FTC*",
}

COMPARISON_POLICY_NAMES: Sequence[str] = (
    "Fcfs_Always_Fast",
    "Fcfs_Always_Slow",
    "Static_Mix_Policy",
    "Queue_Length_Bang_Bang_Policy",
    "Drift_Penalty_Myopic_Policy",
    "Baseline_Heuristic_Policy",
    "FTCPolicy",
)

TTL_PRESETS: Dict[str, Tuple[float, float]] = {
    "very_tight": (28.0, 9.0),
    "tight": (35.0, 10.0),
    "relaxed": (45.0, 12.0),
    "very_relaxed": (60.0, 15.0),
}

TTL_REGIME_LABELS: Dict[str, str] = {
    "very_tight": "TTL Regime I",
    "tight": "TTL Regime II",
    "relaxed": "TTL Regime III",
    "very_relaxed": "TTL Regime IV",
}


@dataclass
class Run_Config:
    outdir: str = "Results/Journal/premium_table"
    lambda_f64: float = 0.05
    ttl_preset: str = "tight"
    premium_rate_f64: float = 0.10
    priority_first_enabled_bool: bool = True


RUN_CONFIG = Run_Config()


def _Log(msg_str: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[Premium][{stamp}] {msg_str}", flush=True)


def _Make_Dir(path_str: str) -> None:
    os.makedirs(path_str, exist_ok=True)


def _Policy_Label(policy_name: str) -> str:
    return POLICY_LABELS.get(policy_name, policy_name)


def _Ttl_Label(preset_name: str) -> str:
    if preset_name not in TTL_PRESETS:
        return preset_name
    mu, sigma = TTL_PRESETS[preset_name]
    regime = TTL_REGIME_LABELS.get(preset_name, preset_name)
    return f"{regime} (mu={mu}, sigma={sigma})"


def _Apply_Comparison_Policy_Config(sim_cfg: Simulation_Config) -> Simulation_Config:
    return replace(
        sim_cfg,
        policy_config=replace(
            sim_cfg.policy_config,
            static_mix_p=0.5,
            queue_threshold_tau=2,
            drift_V=1.0,
        ),
    )


def _Apply_Ttl_Preset(sim_cfg: Simulation_Config, preset_name: str) -> Simulation_Config:
    if preset_name not in TTL_PRESETS:
        raise ValueError(f"Unknown TTL preset: {preset_name}")
    ttl_seconds, ttl_std = TTL_PRESETS[preset_name]
    return replace(
        sim_cfg,
        ttl_ttl_config=replace(
            sim_cfg.ttl_ttl_config,
            ttl_seconds_f64=float(ttl_seconds),
            ttl_std_f64=float(ttl_std),
        ),
    )


def _Build_Service_Model(sim_cfg: Simulation_Config):
    service_cfg = sim_cfg.service_config
    service_source = str(service_cfg.service_time_source).lower()
    if service_source == "trace":
        base = Trace_Service_Times(
            csv_path_str=str(REPO_ROOT / service_cfg.trace_csv_path_str),
            fast_latency_column_str=service_cfg.trace_fast_column_str,
            slow_latency_column_str=service_cfg.trace_slow_column_str,
            drop_error_rows_bool=service_cfg.trace_drop_error_rows_bool,
            prompt_type_filter_opt=service_cfg.trace_prompt_type_filter_opt,
        )
    elif service_source == "lognormal":
        base = Lognormal_Service_Times(
            slow_mu_f64=service_cfg.slow_logn_mu_f64,
            slow_sigma_f64=service_cfg.slow_logn_sigma_f64,
            fast_mu_f64=service_cfg.fast_logn_mu_f64,
            fast_sigma_f64=service_cfg.fast_logn_sigma_f64,
        )
    else:
        raise ValueError(f"Unknown service_time_source: {service_cfg.service_time_source}")

    if service_cfg.ewma_enabled_bool:
        return EWMA_Service_Time_Estimator(
            base_model=base,
            alpha_f64=service_cfg.ewma_alpha_f64,
            warmup_count_i32=service_cfg.ewma_warmup_count_i32,
        )
    return base


def _Build_Policy(
    policy_name: str,
    sim_cfg: Simulation_Config,
    service_model,
    rng: np.random.Generator,
):
    if policy_name == "Fcfs_Always_Fast":
        return Fcfs_Always_Fast_Policy()
    if policy_name == "Fcfs_Always_Slow":
        return Fcfs_Always_Slow_Policy()
    if policy_name == "Baseline_Heuristic_Policy":
        return Baseline_Heuristic_Policy(sim_cfg.policy_config, service_model)
    if policy_name == "Static_Mix_Policy":
        return Static_Mix_Policy(sim_cfg.policy_config, rng=rng)
    if policy_name == "Queue_Length_Bang_Bang_Policy":
        return Queue_Length_Bang_Bang_Policy(sim_cfg.policy_config)
    if policy_name == "Drift_Penalty_Myopic_Policy":
        return Drift_Penalty_Myopic_Policy(sim_cfg.policy_config, service_model, sim_cfg.utility_config)
    if policy_name in {"FTCPolicy", "DATPPolicy"}:
        return FTCPolicy(sim_cfg.policy_config, service_model)
    raise ValueError(f"Unknown policy: {policy_name}")


def _Run_Single(sim_cfg: Simulation_Config, policy_name: str) -> MetricsCollector:
    interarrival = Exponential_Interarrival(sim_cfg.arrival_config.lambda_rate_f64)
    service_model = _Build_Service_Model(sim_cfg)
    utility_rng = np.random.default_rng(sim_cfg.seed_i32 + 2000)
    utility_model = Firm_Deadline_Quality_Utility(sim_cfg.utility_config, rng_opt=utility_rng)
    metrics = MetricsCollector(utility_model_utility_model=utility_model, warmup_time_f64=sim_cfg.warmup_time_f64)
    rng = np.random.default_rng(sim_cfg.seed_i32 + 1000)
    policy = _Build_Policy(policy_name, sim_cfg, service_model, rng)
    sim = Simulator(sim_cfg, interarrival, service_model, policy, metrics)
    _ = sim.Run()
    return metrics


def _Task_Missed(task_obj) -> int:
    if task_obj.dropped_in_queue_bool:
        return 1
    if task_obj.completion_time_f64_opt is None:
        return 1
    return 1 if float(task_obj.completion_time_f64_opt) > float(task_obj.deadline) else 0


def _Class_Summary(tasks_list, utility_model) -> Dict[str, float]:
    n_i32 = len(tasks_list)
    if n_i32 == 0:
        return {
            "n_tasks_total": 0,
            "miss_rate": float("nan"),
            "mean_utility": float("nan"),
            "response_time_p95": float("nan"),
            "slow_fraction": float("nan"),
        }

    utilities_list_f64 = [utility_model.Utility(t) for t in tasks_list]
    miss_flags = [_Task_Missed(t) for t in tasks_list]
    resp_list = [float(t.Response_Time) for t in tasks_list if t.Response_Time is not None]

    slow_count = 0
    fast_count = 0
    for t in tasks_list:
        if t.chosen_mode_mode_opt == Mode.SLOW:
            slow_count += 1
        elif t.chosen_mode_mode_opt == Mode.FAST:
            fast_count += 1

    denom = slow_count + fast_count
    slow_frac = float(slow_count / denom) if denom > 0 else float("nan")
    p95 = float(np.percentile(resp_list, 95)) if resp_list else float("nan")

    return {
        "n_tasks_total": int(n_i32),
        "miss_rate": float(np.mean(miss_flags)) if miss_flags else float("nan"),
        "mean_utility": float(np.mean(utilities_list_f64)) if utilities_list_f64 else float("nan"),
        "response_time_p95": p95,
        "slow_fraction": slow_frac,
    }


def _Priority_Class_Summaries(metrics: MetricsCollector) -> Dict[str, Dict[str, float]]:
    tasks_all = (
        list(metrics.completed_tasks_list_task)
        + list(metrics.dropped_tasks_list_task)
        + list(metrics.unfinished_tasks_list_task)
    )
    premium = [t for t in tasks_all if bool(getattr(t, "high_priority_bool", False))]
    standard = [t for t in tasks_all if not bool(getattr(t, "high_priority_bool", False))]

    util_model = metrics.utility_model_utility_model
    return {
        "premium": _Class_Summary(premium, util_model),
        "normal": _Class_Summary(standard, util_model),
        "overall": _Class_Summary(tasks_all, util_model),
    }


def Run_Premium_Table(run_cfg: Run_Config) -> None:
    out_dir = run_cfg.outdir
    _Make_Dir(out_dir)

    sim_cfg = _Apply_Comparison_Policy_Config(Simulation_Config())
    cfg_ttl = _Apply_Ttl_Preset(sim_cfg, run_cfg.ttl_preset)
    cfg = replace(
        cfg_ttl,
        arrival_config=replace(cfg_ttl.arrival_config, lambda_rate_f64=float(run_cfg.lambda_f64)),
        priority_task_rate_f64=float(run_cfg.premium_rate_f64),
        priority_first_enabled_bool=bool(run_cfg.priority_first_enabled_bool),
    )

    rows: List[Dict[str, object]] = []
    class_order = [("premium", "Premium"), ("normal", "Normal"), ("overall", "Overall")]

    for policy_name in COMPARISON_POLICY_NAMES:
        _Log(f"Running policy: {_Policy_Label(policy_name)}")
        metrics = _Run_Single(cfg, policy_name)
        class_summaries = _Priority_Class_Summaries(metrics)
        for class_key, class_label in class_order:
            stats = class_summaries.get(class_key, {})
            rows.append(
                {
                    "Class": class_label,
                    "Policy": _Policy_Label(policy_name),
                    "lambda": float(run_cfg.lambda_f64),
                    "ttl_preset": _Ttl_Label(run_cfg.ttl_preset),
                    "premium_rate": float(run_cfg.premium_rate_f64),
                    "n_tasks_total": int(stats.get("n_tasks_total", 0)),
                    "DMR": float(stats.get("miss_rate", float("nan"))),
                    "Mean_Utility": float(stats.get("mean_utility", float("nan"))),
                    "p95_Response": float(stats.get("response_time_p95", float("nan"))),
                    "Slow_Fraction": float(stats.get("slow_fraction", float("nan"))),
                }
            )

    out_path = os.path.join(out_dir, "premium_policy_table.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    Run_Premium_Table(RUN_CONFIG)
