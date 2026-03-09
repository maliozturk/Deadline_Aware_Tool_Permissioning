from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from Configurations import Simulation_Config
from Core.Task import Mode
from Core.Simulator import Simulator
from Metrics.Collector import MetricsCollector, SummaryStats
from Models.Distributions import (
    EWMA_Service_Time_Estimator,
    Exponential_Interarrival,
    Lognormal_Service_Times,
    Trace_Service_Times,
)
from Models.Policies import (
    Baseline_Heuristic_Policy,
    DATPPolicy,
    Drift_Penalty_Myopic_Policy,
    Fcfs_Always_Fast_Policy,
    Fcfs_Always_Slow_Policy,
    Queue_Length_Bang_Bang_Policy,
    Static_Mix_Policy,
    TTL_Feasibility_Bang_Bang_Policy,
)
from Models.Utility import Firm_Deadline_Quality_Utility


POLICY_LABELS: Dict[str, str] = {
    "Fcfs_Always_Fast": "AlwaysFast (AF)",
    "Fcfs_Always_Slow": "AlwaysSlow (AS)",
    "Static_Mix_Policy": "StaticMix (p=0.5)",
    "Queue_Length_Bang_Bang_Policy": "Q-BB",
    "TTL_Feasibility_Bang_Bang_Policy": "F-BB",
    "Drift_Penalty_Myopic_Policy": "DriftPenaltyMyopic (V=1.0)",
    "Baseline_Heuristic_Policy": "TTL-aware heuristic",
    "DATPPolicy": "DATP*",
}

COMPARISON_POLICY_NAMES: Sequence[str] = (
    "Fcfs_Always_Fast",
    "Fcfs_Always_Slow",
    "Static_Mix_Policy",
    "Queue_Length_Bang_Bang_Policy",
    "TTL_Feasibility_Bang_Bang_Policy",
    "Drift_Penalty_Myopic_Policy",
    "Baseline_Heuristic_Policy",
    "DATPPolicy",
)

REPORT_METRICS: Sequence[str] = (
    "mean_utility",
    "miss_rate",
    "response_time_p95",
    "slow_fraction",
    "queue_len_mean",
)

RESULTS_TABLE_FIELDS: Sequence[str] = ("lambda", "policy_name", *REPORT_METRICS)

# TTL_PRESETS: Dict[str, Tuple[float, float]] = {
#     "very_tight": (2.0, 0.3),
#     "tight": (2.5, 0.5),
#     "relaxed": (2.8, 0.5),
#     "very_relaxed": (3.0, 0.5),
# }

TTL_REGIME_LABELS: Dict[str, str] = {
    "very_tight": "TTL Regime I",
    "tight": "TTL Regime II",
    "relaxed": "TTL Regime III",
    "very_relaxed": "TTL Regime IV",
}

TTL_PRESETS: Dict[str, Tuple[float, float]] = {
    "very_tight": (28.0, 9.0),
    "tight": (35.0, 10.0),
    "relaxed": (45.0, 12.0),
    "very_relaxed": (60.0, 15),
}


@dataclass
class Run_Config:
    outdir: str = "Results/Paper"
    tool_trials_csv: str = "Tool_Caller_Agent/results_tool_calls/tool_call_trials.csv"

    lambda_rate_list_f64: Sequence[float] = (0.03, 0.05, 0.07, 0.09)
    ttl_seconds_list_f64: Sequence[float] = (28.0, 35.0, 45.0, 60.0)
    ttl_presets_list_str: Sequence[str] = ("very_tight", "tight", "relaxed", "very_relaxed")
    datp_slack_factor_list_f64: Sequence[float] = (0.8, 0.9, 1.0, 1.1)
    datp_epsilon_list_f64: Sequence[float] = (0.0, 0.05, 0.1, 0.15)
    epsilon_sweep_lambda_f64: float = 0.05
    datp_adaptive_epsilon_enabled_bool: bool = False
    epsilon_pareto_policy_names_list_str: Sequence[str] = COMPARISON_POLICY_NAMES

    priority_task_rate_f64: float = 0.0
    priority_first_enabled_bool: bool = False

    queue_dist_lambda_f64: float = 0.9
    trace_lambda_f64: float = 0.9
    slack_lambda_f64: float = 1.0
    trace_window_i32: int = 25

    run_all: bool = False
    run_tool_latency: bool = False
    run_main_sweep: bool = True
    run_queue_dist: bool = False
    run_trace: bool = False
    run_slack_sweep: bool = False
    run_epsilon_sweep: bool = False
    run_epsilon_pareto_compare: bool = False
    run_epsilon_table: bool = False
    run_ttl_sweep: bool = False
    run_priority: bool = False
    run_table: bool = True


# Edit RUN_CONFIG to change what runs.
RUN_CONFIG = Run_Config()


def _Log(msg_str: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[DATP][{stamp}] {msg_str}", flush=True)


def _Make_Dir(path_str: str) -> None:
    os.makedirs(path_str, exist_ok=True)


def _Apply_Plot_Style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _Parse_Float_List(s_list_str: str) -> List[float]:
    if not s_list_str:
        return []
    return [float(x.strip()) for x in s_list_str.split(",") if x.strip()]


def _Parse_Name_List(s_list_str: str) -> List[str]:
    if not s_list_str:
        return []
    return [x.strip() for x in s_list_str.split(",") if x.strip()]


def _Coerce_Float_List(values) -> List[float]:
    if values is None:
        return []
    if isinstance(values, str):
        return _Parse_Float_List(values)
    return [float(v) for v in values]


def _Coerce_Name_List(values) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return _Parse_Name_List(values)
    return [str(v) for v in values]


def _Safe_Label(val_f64: float) -> str:
    return str(val_f64).replace(".", "p")

def _Ttl_Label(preset_name: str) -> str:
    if preset_name not in TTL_PRESETS:
        return preset_name
    mu, sigma = TTL_PRESETS[preset_name]
    regime = TTL_REGIME_LABELS.get(preset_name, preset_name)
    return f"{regime} (mu={mu}, sigma={sigma})"



def _Policy_Label(policy_name: str) -> str:
    return POLICY_LABELS.get(policy_name, policy_name)


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


def _Summary(samples_list_f64: Sequence[float]) -> Dict[str, float]:
    if not samples_list_f64:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    arr = np.asarray(samples_list_f64, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _Fit_Lognormal(samples_list_f64: Sequence[float]) -> Tuple[float, float]:
    xs = [x for x in samples_list_f64 if x > 0]
    if len(xs) < 2:
        return float("nan"), float("nan")
    logs = np.log(np.asarray(xs, dtype=float))
    return float(np.mean(logs)), float(np.std(logs, ddof=1))


def _Lognormal_Pdf(x_arr_f64: np.ndarray, mu_f64: float, sigma_f64: float) -> np.ndarray:
    x_arr_f64 = np.asarray(x_arr_f64, dtype=float)
    if sigma_f64 <= 0:
        return np.zeros_like(x_arr_f64)
    coeff = 1.0 / (x_arr_f64 * sigma_f64 * math.sqrt(2.0 * math.pi))
    expo = np.exp(-((np.log(x_arr_f64) - mu_f64) ** 2) / (2.0 * sigma_f64 ** 2))
    return coeff * expo


def _Plot_Tool_Latency_Fit(samples_dict: Dict[str, List[float]], out_dir: str) -> Dict[str, Dict[str, float]]:
    stats_dict: Dict[str, Dict[str, float]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, mode in enumerate(["fast", "slow"]):
        samples = samples_dict.get(mode, [])
        mu_f64, sigma_f64 = _Fit_Lognormal(samples)
        stats = _Summary(samples)
        stats["mu"] = mu_f64
        stats["sigma"] = sigma_f64
        stats_dict[mode] = stats

        ax = axes[idx]
        if samples:
            ax.hist(samples, bins=30, density=True, alpha=0.6, label=f"{mode} hist")
            x_min = max(min(samples), 1e-6)
            x_max = max(samples)
            x_grid = np.linspace(x_min, x_max, 300)
            pdf = _Lognormal_Pdf(x_grid, mu_f64, sigma_f64)
            ax.plot(x_grid, pdf, color="black", linewidth=1.5, label="lognormal pdf")
        ax.set_title(f"{mode} tool latency")
        ax.set_xlabel("latency (s)")
        ax.set_ylabel("density")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "tool_latency_hist_fit.png")
    plt.savefig(out_path)
    plt.close(fig)
    return stats_dict


def _Plot_Tool_Latency_QQ(samples_dict: Dict[str, List[float]], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, mode in enumerate(["fast", "slow"]):
        samples = [x for x in samples_dict.get(mode, []) if x > 0]
        ax = axes[idx]
        if not samples:
            ax.set_title(f"{mode} QQ (no samples)")
            continue

        logs = np.log(np.sort(np.asarray(samples, dtype=float)))
        n = logs.size
        probs = (np.arange(1, n + 1) - 0.5) / n
        mu_f64, sigma_f64 = _Fit_Lognormal(samples)
        z = np.asarray([NormalDist().inv_cdf(float(p)) for p in probs], dtype=float)
        theo = mu_f64 + sigma_f64 * z

        ax.plot(theo, logs, "o", markersize=3, alpha=0.7, label="empirical")
        min_v = float(min(theo.min(), logs.min()))
        max_v = float(max(theo.max(), logs.max()))
        ax.plot([min_v, max_v], [min_v, max_v], "-", color="black", linewidth=1.0, label="y=x")
        ax.set_title(f"{mode} log-latency QQ")
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("sample quantiles")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "tool_latency_qq.png")
    plt.savefig(out_path)
    plt.close(fig)


def _Load_Tool_Call_Trials(csv_path: str) -> Dict[str, List[float]]:
    fast_list: List[float] = []
    slow_list: List[float] = []

    if not os.path.exists(csv_path):
        return {"fast": fast_list, "slow": slow_list}

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            scenario = row.get("scenario", "")
            chosen_tool = row.get("chosen_tool", "")
            try:
                t = float(row.get("tool_runtime_s", "nan"))
            except ValueError:
                continue

            if scenario == "forced_fast" and chosen_tool == "fast_tool":
                fast_list.append(t)
            elif scenario == "forced_slow" and chosen_tool == "slow_tool":
                slow_list.append(t)

    return {"fast": fast_list, "slow": slow_list}


def Tool_Latency_Analysis(trials_csv_path: str, out_dir: str) -> None:
    _Make_Dir(out_dir)
    _Log(f"Tool latency analysis: {trials_csv_path} -> {out_dir}")
    samples_dict = _Load_Tool_Call_Trials(trials_csv_path)
    stats = _Plot_Tool_Latency_Fit(samples_dict, out_dir)
    _Plot_Tool_Latency_QQ(samples_dict, out_dir)

    stats_path = os.path.join(out_dir, "tool_latency_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def Utility_Calibration_Report(sim_cfg: Simulation_Config, out_dir: str) -> None:
    _Make_Dir(out_dir)
    _Log(f"Utility calibration report -> {out_dir}")
    ucfg = sim_cfg.utility_config
    payload = {
        "firm_deadline": bool(ucfg.firm_deadline_bool),
        "slow_success_mean": float(ucfg.slow_success_utility_f64),
        "slow_success_std": float(ucfg.slow_success_std_f64),
        "slow_success_min": float(ucfg.slow_success_min_f64),
        "slow_success_max": float(ucfg.slow_success_max_f64) if ucfg.slow_success_max_f64 is not None else None,
        "fast_success_mean": float(ucfg.fast_success_utility_f64),
        "fast_success_std": float(ucfg.fast_success_std_f64),
        "fast_success_min": float(ucfg.fast_success_min_f64),
        "fast_success_max": float(ucfg.fast_success_max_f64) if ucfg.fast_success_max_f64 is not None else None,
        "missed_deadline_utility": float(ucfg.missed_deadline_utility_f64),
        "notes": "Utilities are sampled from clipped normals when on-time; configured means/stds shown here.",
    }
    out_path = os.path.join(out_dir, "utility_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
    if policy_name == "TTL_Feasibility_Bang_Bang_Policy":
        return TTL_Feasibility_Bang_Bang_Policy(sim_cfg.policy_config, service_model)
    if policy_name == "Drift_Penalty_Myopic_Policy":
        return Drift_Penalty_Myopic_Policy(sim_cfg.policy_config, service_model, sim_cfg.utility_config)
    if policy_name == "DATPPolicy":
        return DATPPolicy(sim_cfg.policy_config, service_model)
    raise ValueError(f"Unknown policy: {policy_name}")


def _Run_Single(
    sim_cfg: Simulation_Config,
    policy_name: str,
    trace_enabled_bool: bool = False,
):
    if trace_enabled_bool:
        sim_cfg = replace(
            sim_cfg,
            policy_config=replace(sim_cfg.policy_config, datp_trace_enabled_bool=True),
        )

    interarrival = Exponential_Interarrival(sim_cfg.arrival_config.lambda_rate_f64)
    service_model = _Build_Service_Model(sim_cfg)
    utility_rng = np.random.default_rng(sim_cfg.seed_i32 + 2000)
    utility_model = Firm_Deadline_Quality_Utility(sim_cfg.utility_config, rng_opt=utility_rng)
    metrics = MetricsCollector(utility_model_utility_model=utility_model, warmup_time_f64=sim_cfg.warmup_time_f64)
    rng = np.random.default_rng(sim_cfg.seed_i32 + 1000)
    policy = _Build_Policy(policy_name, sim_cfg, service_model, rng)
    sim = Simulator(sim_cfg, interarrival, service_model, policy, metrics)
    agg = sim.Run()

    return agg, metrics, policy


def _Extract_Summary(agg_dict_obj: Dict[str, object]) -> Dict[str, float]:
    resp_stats: SummaryStats = agg_dict_obj.get("response_time")  # type: ignore[assignment]
    wait_stats: SummaryStats = agg_dict_obj.get("waiting_time")  # type: ignore[assignment]
    mode_counts: Dict[str, int] = agg_dict_obj.get("mode_counts_completed")  # type: ignore[assignment]

    slow_count = int(mode_counts.get("slow", 0))
    fast_count = int(mode_counts.get("fast", 0))
    denom = slow_count + fast_count
    slow_frac = float(slow_count / denom) if denom > 0 else float("nan")

    summary = {
        "n_tasks_total": int(agg_dict_obj.get("n_tasks_total", 0)),
        "n_completed": int(agg_dict_obj.get("n_completed", 0)),
        "n_dropped_in_queue": int(agg_dict_obj.get("n_dropped_in_queue", 0)),
        "n_unfinished": int(agg_dict_obj.get("n_unfinished", 0)),
        "miss_rate": float(agg_dict_obj.get("miss_rate", float("nan"))),
        "mean_utility": float(agg_dict_obj.get("mean_utility", float("nan"))),
        "response_time_mean": float(resp_stats.mean_f64),
        "response_time_p90": float(resp_stats.p90_f64),
        "response_time_p95": float(resp_stats.p95_f64),
        "response_time_p99": float(resp_stats.p99_f64),
        "waiting_time_mean": float(wait_stats.mean_f64),
        "queue_len_mean": float(agg_dict_obj.get("queue_len_mean", float("nan"))),
        "slow_fraction": slow_frac,
    }

    for key in agg_dict_obj.keys():
        if str(key).startswith("paug_beta_"):
            summary[str(key)] = float(agg_dict_obj.get(key, float("nan")))

    return summary


def Run_Lambda_Sweep(
    sim_cfg: Simulation_Config,
    lambda_rate_list_f64: Sequence[float],
    policy_names: Sequence[str],
    out_dir: str,
    tag_str: str,
) -> List[Dict[str, object]]:
    _Make_Dir(out_dir)
    results_list: List[Dict[str, object]] = []

    for lambda_f64 in lambda_rate_list_f64:
        _Log(f"Lambda sweep: lambda={lambda_f64}")
        cfg_lambda = replace(
            sim_cfg,
            arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
        )
        for policy_name in policy_names:
            _Log(f"  policy={_Policy_Label(policy_name)}")
            agg, metrics, _ = _Run_Single(cfg_lambda, policy_name, trace_enabled_bool=False)
            _ = metrics
            summary = _Extract_Summary(agg)
            summary["lambda"] = float(lambda_f64)
            summary["policy_name"] = _Policy_Label(policy_name)
            results_list.append(summary)

    out_path = os.path.join(out_dir, f"sweep_lambda_{tag_str}.csv")
    if results_list:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results_list[0].keys()))
            w.writeheader()
            for row in results_list:
                w.writerow(row)

    return results_list


def _Plot_Curve(
    results_list: Sequence[Dict[str, object]],
    metric_key: str,
    out_path: str,
    ylabel: str,
    title: str,
    ttl_label_opt: Optional[str] = None,
) -> None:
    plt.figure(figsize=(6.5, 4))
    policies = sorted({r["policy_name"] for r in results_list})
    for policy in policies:
        xs = [float(r["lambda"]) for r in results_list if r["policy_name"] == policy]
        ys = [float(r[metric_key]) for r in results_list if r["policy_name"] == policy]
        label = POLICY_LABELS.get(str(policy), str(policy))
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("arrival rate (lambda)")
    plt.ylabel(ylabel)
    plot_title = f"{title} ({ttl_label_opt})" if ttl_label_opt else title
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def Plot_Main_Curves(
    results_list: Sequence[Dict[str, object]],
    out_dir: str,
    tag_str: str,
    ttl_label_opt: Optional[str] = None,
) -> None:
    _Make_Dir(out_dir)
    _Plot_Curve(
        results_list,
        metric_key="mean_utility",
        out_path=os.path.join(out_dir, f"utility_vs_lambda_{tag_str}.png"),
        ylabel="avg deadline-gated utility",
        title="Utility vs arrival rate",
        ttl_label_opt=ttl_label_opt,
    )
    _Plot_Curve(
        results_list,
        metric_key="miss_rate",
        out_path=os.path.join(out_dir, f"miss_rate_vs_lambda_{tag_str}.png"),
        ylabel="deadline miss rate",
        title="Miss rate vs arrival rate",
        ttl_label_opt=ttl_label_opt,
    )
    _Plot_Curve(
        results_list,
        metric_key="slow_fraction",
        out_path=os.path.join(out_dir, f"slow_fraction_vs_lambda_{tag_str}.png"),
        ylabel="slow-mode fraction",
        title="Slow-mode fraction vs arrival rate",
        ttl_label_opt=ttl_label_opt,
    )
    _Plot_Curve(
        results_list,
        metric_key="response_time_p90",
        out_path=os.path.join(out_dir, f"p90_response_vs_lambda_{tag_str}.png"),
        ylabel="p90 response time",
        title="Tail latency (p90) vs arrival rate",
        ttl_label_opt=ttl_label_opt,
    )
    _Plot_Curve(
        results_list,
        metric_key="response_time_p99",
        out_path=os.path.join(out_dir, f"p99_response_vs_lambda_{tag_str}.png"),
        ylabel="p99 response time",
        title="Tail latency (p99) vs arrival rate",
        ttl_label_opt=ttl_label_opt,
    )

    paug_keys = sorted({k for r in results_list for k in r.keys() if str(k).startswith("paug_beta_")})
    for key in paug_keys:
        beta_label = str(key).replace("paug_beta_", "").replace("p", ".")
        _Plot_Curve(
            results_list,
            metric_key=str(key),
            out_path=os.path.join(out_dir, f"{key}_vs_lambda_{tag_str}.png"),
            ylabel=f"PAUG (beta={beta_label})",
            title=f"PAUG vs arrival rate (beta={beta_label})",
            ttl_label_opt=ttl_label_opt,
        )


def Plot_Queue_Length_CDF(samples_dict: Dict[str, Sequence[int]], out_path: str) -> None:
    _Apply_Plot_Style()
    plt.figure(figsize=(6.5, 4))
    for policy_name, samples in samples_dict.items():
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            continue
        arr.sort()
        cdf = np.arange(1, arr.size + 1) / float(arr.size)
        label = POLICY_LABELS.get(policy_name, policy_name)
        plt.plot(arr, cdf, label=label)
    plt.xlabel("queue length")
    plt.ylabel("CDF")
    plt.title("Queue length distribution")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def Run_Queue_Length_Distribution(
    sim_cfg: Simulation_Config,
    lambda_f64: float,
    policy_names: Sequence[str],
    out_dir: str,
    tag_str: str,
) -> None:
    _Make_Dir(out_dir)
    _Log(f"Queue length CDF: lambda={lambda_f64}")
    samples_dict: Dict[str, Sequence[int]] = {}
    cfg_lambda = replace(
        sim_cfg,
        arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
    )
    for policy_name in policy_names:
        _Log(f"  policy={_Policy_Label(policy_name)}")
        agg, metrics, _ = _Run_Single(cfg_lambda, policy_name, trace_enabled_bool=False)
        _ = agg
        samples_dict[policy_name] = list(metrics.queue_len_samples_list_i32)

    out_path = os.path.join(out_dir, f"queue_len_cdf_lambda_{tag_str}.png")
    Plot_Queue_Length_CDF(samples_dict, out_path)


def _Rolling_Mean(values_list_f64: Sequence[float], window_i32: int) -> np.ndarray:
    arr = np.asarray(values_list_f64, dtype=float)
    if arr.size == 0:
        return arr
    if window_i32 <= 1:
        return arr
    window_i32 = min(window_i32, int(arr.size))
    cumsum = np.cumsum(arr, dtype=float)
    cumsum[window_i32:] = cumsum[window_i32:] - cumsum[:-window_i32]
    out = cumsum[window_i32 - 1:] / float(window_i32)
    pad = np.full(window_i32 - 1, out[0] if out.size else float("nan"))
    return np.concatenate([pad, out])


def Plot_Time_Series_Trace(
    metrics: MetricsCollector,
    out_path: str,
    window_i32: int = 200,
    slack_trace_opt: Optional[Sequence[float]] = None,
    slack_time_opt: Optional[Sequence[float]] = None,
) -> None:
    tasks_all = (
        list(metrics.completed_tasks_list_task)
        + list(metrics.dropped_tasks_list_task)
        + list(metrics.unfinished_tasks_list_task)
    )
    tasks_all.sort(key=lambda t: t.arrival_time)

    if not tasks_all:
        return

    arrivals = [float(t.arrival_time) for t in tasks_all]
    slow_flags = [1.0 if t.chosen_mode_mode_opt and t.chosen_mode_mode_opt.value == "slow" else 0.0 for t in tasks_all]
    miss_flags = []
    for t in tasks_all:
        miss = 1.0
        if t.dropped_in_queue_bool:
            miss = 1.0
        elif t.completion_time_f64_opt is None:
            miss = 1.0
        else:
            miss = 1.0 if float(t.completion_time_f64_opt) > float(t.deadline) else 0.0
        miss_flags.append(miss)

    slow_frac = _Rolling_Mean(slow_flags, window_i32)
    dmr = _Rolling_Mean(miss_flags, window_i32)

    q_times = metrics.queue_len_sample_times_list_f64
    q_vals = metrics.queue_len_samples_list_i32

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False)
    axes[0].plot(q_times, q_vals, color="tab:blue")
    axes[0].set_ylabel("queue length")
    axes[0].set_title("Time-series trace")

    axes[1].plot(arrivals, slow_frac, color="tab:green")
    axes[1].set_ylabel(f"slow fraction (w={window_i32})")

    axes[2].plot(arrivals, dmr, color="tab:red")
    axes[2].set_ylabel(f"miss rate (w={window_i32})")
    axes[2].set_xlabel("time (arrival order)")

    if slack_trace_opt is not None and slack_time_opt is not None and len(slack_trace_opt) == len(slack_time_opt):
        ax2 = axes[1].twinx()
        ax2.plot(slack_time_opt, slack_trace_opt, color="tab:purple", alpha=0.6)
        ax2.set_ylabel("predicted slack")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _Wilson_CI(k_i32: int, n_i32: int, z_f64: float = 1.96) -> Tuple[float, float]:
    if n_i32 <= 0:
        return float("nan"), float("nan")

    p_hat = float(k_i32) / float(n_i32)
    denom = 1.0 + (z_f64 ** 2) / float(n_i32)
    center = (p_hat + (z_f64 ** 2) / (2.0 * float(n_i32))) / denom
    half = (
        z_f64
        * math.sqrt((p_hat * (1.0 - p_hat) / float(n_i32)) + (z_f64 ** 2) / (4.0 * (float(n_i32) ** 2)))
        / denom
    )
    return float(max(0.0, center - half)), float(min(1.0, center + half))

def Run_Time_Series_Trace(
    sim_cfg: Simulation_Config,
    lambda_f64: float,
    out_dir: str,
    tag_str: str,
    window_i32: int = 200,
) -> None:
    _Make_Dir(out_dir)
    _Log(f"Time-series trace: lambda={lambda_f64}")
    cfg_lambda = replace(
        sim_cfg,
        arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
    )
    agg, metrics, policy = _Run_Single(cfg_lambda, "DATPPolicy", trace_enabled_bool=True)
    _ = agg

    slack_vals = []
    slack_times = []
    if hasattr(policy, "decision_trace_list"):
        for tr in policy.decision_trace_list:
            slack = float(tr.delta_k_f64 - (tr.w_hat_f64 + tr.s_slow_f64))
            slack_vals.append(slack)
            slack_times.append(float(tr.arrival_time_f64))

    out_path = os.path.join(out_dir, f"time_series_trace_lambda_{tag_str}.png")
    Plot_Time_Series_Trace(metrics, out_path, window_i32=window_i32, slack_trace_opt=slack_vals, slack_time_opt=slack_times)



def Run_Slack_Sweep(
    sim_cfg: Simulation_Config,
    lambda_f64: float,
    tfg_slack_factor_list_f64: Sequence[float],
    out_dir: str,
    tag_str: str,
) -> None:
    _Make_Dir(out_dir)
    _Log(f"Slack sweep: lambda={lambda_f64}")
    points: List[Tuple[float, float, float]] = []

    cfg_lambda = replace(
        sim_cfg,
        arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
    )
    for slack in tfg_slack_factor_list_f64:
        _Log(f"  slack_factor={slack}")
        cfg_slack = replace(
            cfg_lambda,
            policy_config=replace(cfg_lambda.policy_config, datp_slack_factor=float(slack)),
        )
        agg, _, _ = _Run_Single(cfg_slack, "DATPPolicy", trace_enabled_bool=False)
        summary = _Extract_Summary(agg)
        points.append((float(slack), float(summary["miss_rate"]), float(summary["mean_utility"])))

    plt.figure(figsize=(5.5, 4))
    for slack, miss, util in points:
        plt.scatter([miss], [util], label=f"slack={slack:.2f}")
    plt.xlabel("miss rate")
    plt.ylabel("avg utility")
    plt.title("Slack factor sensitivity (Pareto)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tfg_slack_pareto_{tag_str}.png"))
    plt.close()

    out_path = os.path.join(out_dir, f"tfg_slack_pareto_{tag_str}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slack_factor", "miss_rate", "mean_utility"])
        for slack, miss, util in points:
            w.writerow([slack, miss, util])


def Run_Epsilon_Sweep(
    sim_cfg: Simulation_Config,
    lambda_f64: float,
    tfg_epsilon_list_f64: Sequence[float],
    out_dir: str,
    tag_str: str,
    tfg_adaptive_epsilon_enabled_bool: bool = False,
) -> None:
    if not tfg_epsilon_list_f64:
        return
    _Make_Dir(out_dir)
    _Log(f"Epsilon sweep: lambda={lambda_f64} adaptive={tfg_adaptive_epsilon_enabled_bool}")
    points: List[Tuple[float, float, float]] = []

    cfg_lambda = replace(
        sim_cfg,
        arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
    )
    for eps in tfg_epsilon_list_f64:
        _Log(f"  epsilon={eps}")
        cfg_eps = replace(
            cfg_lambda,
            policy_config=replace(
                cfg_lambda.policy_config,
                datp_epsilon_f64=float(eps),
                datp_adaptive_epsilon_enabled_bool=bool(tfg_adaptive_epsilon_enabled_bool),
            ),
        )
        agg, _, _ = _Run_Single(cfg_eps, "DATPPolicy", trace_enabled_bool=False)
        summary = _Extract_Summary(agg)
        points.append((float(eps), float(summary["miss_rate"]), float(summary["mean_utility"])))

    plt.figure(figsize=(5.5, 4))
    miss_vals = [p[1] for p in points]
    util_vals = [p[2] for p in points]
    plt.plot(miss_vals, util_vals, marker="o")
    plt.xlabel("miss rate")
    plt.ylabel("avg utility")
    title = (
        "Epsilon sweep Pareto (adaptive)"
        if tfg_adaptive_epsilon_enabled_bool
        else "Epsilon sweep Pareto (fixed)"
    )
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tfg_epsilon_pareto_{tag_str}.png"))
    plt.close()

    out_path = os.path.join(out_dir, f"tfg_epsilon_pareto_{tag_str}.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epsilon_target", "miss_rate", "mean_utility"])
        for eps, miss, util in points:
            w.writerow([eps, miss, util])


def Run_Epsilon_Pareto_Comparison(
    sim_cfg: Simulation_Config,
    lambda_f64: float,
    policy_names: Sequence[str],
    tfg_epsilon_list_f64: Sequence[float],
    out_dir: str,
    tag_str: str,
    tfg_adaptive_epsilon_enabled_bool: bool = False,
) -> None:
    if not policy_names:
        return

    _Make_Dir(out_dir)
    _Log(f"Epsilon pareto compare: lambda={lambda_f64}")

    cfg_lambda = replace(
        sim_cfg,
        arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
    )

    rows: List[Dict[str, object]] = []
    plt.figure(figsize=(6.0, 4))

    for policy_name in policy_names:
        if policy_name == "DATPPolicy":
            if not tfg_epsilon_list_f64:
                _Log("  skip DATPPolicy: datp_epsilon_list_f64 is empty")
                continue
            miss_vals: List[float] = []
            util_vals: List[float] = []
            for eps in tfg_epsilon_list_f64:
                cfg_eps = replace(
                    cfg_lambda,
                    policy_config=replace(
                        cfg_lambda.policy_config,
                        datp_epsilon_f64=float(eps),
                        datp_adaptive_epsilon_enabled_bool=bool(tfg_adaptive_epsilon_enabled_bool),
                    ),
                )
                agg, _, _ = _Run_Single(cfg_eps, policy_name, trace_enabled_bool=False)
                summary = _Extract_Summary(agg)
                miss = float(summary["miss_rate"])
                util = float(summary["mean_utility"])
                miss_vals.append(miss)
                util_vals.append(util)
                rows.append(
                    {
                        "policy_name": _Policy_Label(str(policy_name)),
                        "epsilon_target": float(eps),
                        "miss_rate": miss,
                        "mean_utility": util,
                    }
                )
            label = _Policy_Label(str(policy_name))
            plt.plot(miss_vals, util_vals, marker="o", label=label)
        else:
            agg, _, _ = _Run_Single(cfg_lambda, policy_name, trace_enabled_bool=False)
            summary = _Extract_Summary(agg)
            miss = float(summary["miss_rate"])
            util = float(summary["mean_utility"])
            label = _Policy_Label(str(policy_name))
            plt.scatter([miss], [util], label=label)
            rows.append(
                {
                    "policy_name": _Policy_Label(str(policy_name)),
                    "epsilon_target": float("nan"),
                    "miss_rate": miss,
                    "mean_utility": util,
                }
            )

    if rows:
        plt.xlabel("miss rate")
        plt.ylabel("avg utility")
        plt.title("Utility vs miss rate (epsilon sweep comparison)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"epsilon_pareto_compare_{tag_str}.png"))
        plt.close()

        out_path = os.path.join(out_dir, f"epsilon_pareto_compare_{tag_str}.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)


def Write_Epsilon_Control_Table(
    sim_cfg: Simulation_Config,
    lambda_f64: float,
    tfg_epsilon_list_f64: Sequence[float],
    out_path: str,
) -> None:
    if not tfg_epsilon_list_f64:
        return

    rows: List[Dict[str, object]] = []
    cfg_lambda = replace(
        sim_cfg,
        arrival_config=replace(sim_cfg.arrival_config, lambda_rate_f64=float(lambda_f64)),
    )

    for eps in tfg_epsilon_list_f64:
        for adaptive in [False, True]:
            cfg_eps = replace(
                cfg_lambda,
                policy_config=replace(
                    cfg_lambda.policy_config,
                    datp_epsilon_f64=float(eps),
                    datp_adaptive_epsilon_enabled_bool=bool(adaptive),
                ),
            )
            agg, _, _ = _Run_Single(cfg_eps, "DATPPolicy", trace_enabled_bool=False)
            summary = _Extract_Summary(agg)
            row = {
                "lambda": float(lambda_f64),
                "epsilon_target": float(eps),
                "adaptive_enabled": bool(adaptive),
                "mean_utility": float(summary["mean_utility"]),
                "miss_rate": float(summary["miss_rate"]),
                "response_time_p95": float(summary["response_time_p95"]),
                "slow_fraction": float(summary["slow_fraction"]),
            }
            rows.append(row)

    if not rows:
        return

    _Make_Dir(os.path.dirname(out_path))
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


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
        "standard": _Class_Summary(standard, util_model),
    }


def _Plot_Class_Metric(
    results_list: Sequence[Dict[str, object]],
    metric_key: str,
    out_path: str,
    ylabel: str,
    title: str,
) -> None:
    _Apply_Plot_Style()
    lambdas = sorted({float(r["lambda"]) for r in results_list})
    classes = sorted({str(r["class"]) for r in results_list})

    plt.figure(figsize=(6.5, 4))
    if len(lambdas) <= 1:
        vals = []
        labels = []
        for cls in classes:
            row = next((r for r in results_list if str(r["class"]) == cls), None)
            if row is None:
                continue
            vals.append(float(row[metric_key]))
            labels.append(cls)
        plt.bar(labels, vals, color=["tab:blue", "tab:orange"][: len(labels)])
    else:
        for cls in classes:
            xs = [float(r["lambda"]) for r in results_list if str(r["class"]) == cls]
            ys = [float(r[metric_key]) for r in results_list if str(r["class"]) == cls]
            plt.plot(xs, ys, marker="o", label=cls)
        plt.legend(fontsize=8)

    plt.xlabel("arrival rate (lambda)")
    plt.ylabel(ylabel)
    plot_title = f"{title} ({ttl_label_opt})" if ttl_label_opt else title
    plt.title(plot_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def Run_Priority_Sweep(
    sim_cfg: Simulation_Config,
    lambda_rate_list_f64: Sequence[float],
    policy_name: str,
    priority_task_rate_f64: float,
    priority_first_enabled_bool: bool,
    out_dir: str,
    tag_str: str,
) -> List[Dict[str, object]]:
    _Make_Dir(out_dir)
    results_list: List[Dict[str, object]] = []

    cfg_priority = replace(
        sim_cfg,
        priority_task_rate_f64=float(priority_task_rate_f64),
        priority_first_enabled_bool=bool(priority_first_enabled_bool),
    )

    for lambda_f64 in lambda_rate_list_f64:
        _Log(f"Priority sweep: lambda={lambda_f64}")
        cfg_lambda = replace(
            cfg_priority,
            arrival_config=replace(cfg_priority.arrival_config, lambda_rate_f64=float(lambda_f64)),
        )
        agg, metrics, _ = _Run_Single(cfg_lambda, policy_name, trace_enabled_bool=False)
        _ = agg
        class_summaries = _Priority_Class_Summaries(metrics)
        for cls, stats in class_summaries.items():
            row = {
                "lambda": float(lambda_f64),
                "class": cls,
                **stats,
            }
            results_list.append(row)

    out_path = os.path.join(out_dir, f"priority_class_summary_{tag_str}.csv")
    if results_list:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results_list[0].keys()))
            w.writeheader()
            for row in results_list:
                w.writerow(row)

    _Plot_Class_Metric(
        results_list,
        metric_key="miss_rate",
        out_path=os.path.join(out_dir, f"premium_vs_standard_miss_rate_{tag_str}.png"),
        ylabel="miss rate",
        title="Premium vs standard miss rate",
    )
    _Plot_Class_Metric(
        results_list,
        metric_key="mean_utility",
        out_path=os.path.join(out_dir, f"utility_by_class_{tag_str}.png"),
        ylabel="avg utility",
        title="Utility by class",
    )

    return results_list


def Write_Priority_Table(
    results_list: Sequence[Dict[str, object]],
    load_points: Sequence[float],
    out_path: str,
) -> None:
    rows: List[Dict[str, object]] = []
    for lam in load_points:
        for r in results_list:
            if float(r["lambda"]) != float(lam):
                continue
            rows.append(
                {
                    "lambda": float(r["lambda"]),
                    "class": r["class"],
                    "n_tasks_total": int(r["n_tasks_total"]),
                    "miss_rate": float(r["miss_rate"]),
                    "mean_utility": float(r["mean_utility"]),
                    "response_time_p95": float(r["response_time_p95"]),
                    "slow_fraction": float(r["slow_fraction"]),
                }
            )

    if not rows:
        return

    _Make_Dir(os.path.dirname(out_path))
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def Run_Ttl_Sweep(
    sim_cfg: Simulation_Config,
    lambda_rate_list_f64: Sequence[float],
    ttl_seconds_list_f64: Sequence[float],
    out_dir: str,
    tag_str: str,
) -> None:
    _Make_Dir(out_dir)
    _Log("TTL sweep")
    policies = list(COMPARISON_POLICY_NAMES)
    for ttl in ttl_seconds_list_f64:
        _Log(f"  ttl_seconds={ttl}")
        cfg_ttl = replace(
            sim_cfg,
            ttl_ttl_config=replace(sim_cfg.ttl_ttl_config, ttl_seconds_f64=float(ttl)),
        )
        results = Run_Lambda_Sweep(
            cfg_ttl,
            lambda_rate_list_f64,
            policies,
            out_dir,
            f"{tag_str}_ttl_{_Safe_Label(ttl)}",
        )
        Plot_Main_Curves(results, out_dir, f"{tag_str}_ttl_{_Safe_Label(ttl)}")


def Write_Results_Table(
    results_list: Sequence[Dict[str, object]],
    load_points: Sequence[float],
    out_path: str,
) -> None:
    rows: List[Dict[str, object]] = []
    for lam in load_points:
        for r in results_list:
            if float(r["lambda"]) != float(lam):
                continue
            row = {
                "lambda": float(r["lambda"]),
                "policy_name": r["policy_name"],
                "mean_utility": float(r["mean_utility"]),
                "miss_rate": float(r["miss_rate"]),
                "response_time_p95": float(r["response_time_p95"]),
                "slow_fraction": float(r["slow_fraction"]),
                "queue_len_mean": float(r["queue_len_mean"]),
            }
            rows.append(row)

    if not rows:
        return

    _Make_Dir(os.path.dirname(out_path))
    fieldnames = list(RESULTS_TABLE_FIELDS)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(run_cfg: Run_Config) -> None:
    sim_cfg = _Apply_Comparison_Policy_Config(Simulation_Config())
    outdir = run_cfg.outdir

    _Log(f"Output dir: {outdir}")

    lambda_list = _Coerce_Float_List(run_cfg.lambda_rate_list_f64)
    ttl_list = _Coerce_Float_List(run_cfg.ttl_seconds_list_f64)
    ttl_presets = _Coerce_Name_List(run_cfg.ttl_presets_list_str)
    slack_list = _Coerce_Float_List(run_cfg.datp_slack_factor_list_f64)
    epsilon_list = _Coerce_Float_List(run_cfg.datp_epsilon_list_f64)
    epsilon_pareto_policy_names = _Coerce_Name_List(run_cfg.epsilon_pareto_policy_names_list_str)

    run_all = bool(run_cfg.run_all)

    if run_all or run_cfg.run_tool_latency:
        Tool_Latency_Analysis(run_cfg.tool_trials_csv, os.path.join(outdir, "tool_latency"))
        Utility_Calibration_Report(sim_cfg, os.path.join(outdir, "tool_latency"))

    policy_names = list(COMPARISON_POLICY_NAMES)
    run_specs: List[Tuple[str, Simulation_Config, str]] = []
    if ttl_presets:
        for preset_name in ttl_presets:
            cfg_preset = _Apply_Ttl_Preset(sim_cfg, preset_name)
            outdir_preset = os.path.join(outdir, f"ttl_{preset_name}")
            run_specs.append((preset_name, cfg_preset, outdir_preset))
    else:
        run_specs.append(("default", sim_cfg, outdir))

    for preset_name, cfg_run, outdir_run in run_specs:
        _Log(f"Run config: {preset_name} -> {outdir_run}")
        ttl_label = _Ttl_Label(preset_name)

        sweep_results: List[Dict[str, object]] = []
        if run_all or run_cfg.run_main_sweep:
            _Log("Main sweep")
            sweep_results = Run_Lambda_Sweep(
                sim_cfg=cfg_run,
                lambda_rate_list_f64=lambda_list,
                policy_names=policy_names,
                out_dir=os.path.join(outdir_run, "main_sweep"),
                tag_str="main",
            )
            Plot_Main_Curves(
                sweep_results,
                os.path.join(outdir_run, "main_sweep"),
                "main",
                ttl_label_opt=ttl_label,
            )

        if run_all or run_cfg.run_queue_dist:
            _Log("Queue distribution")
            Run_Queue_Length_Distribution(
                sim_cfg=cfg_run,
                lambda_f64=float(run_cfg.queue_dist_lambda_f64),
                policy_names=policy_names,
                out_dir=os.path.join(outdir_run, "queue_dist"),
                tag_str=_Safe_Label(float(run_cfg.queue_dist_lambda_f64)),
            )

        if run_all or run_cfg.run_trace:
            _Log("Time-series trace")
            Run_Time_Series_Trace(
                sim_cfg=cfg_run,
                lambda_f64=float(run_cfg.trace_lambda_f64),
                out_dir=os.path.join(outdir_run, "traces"),
                tag_str=_Safe_Label(float(run_cfg.trace_lambda_f64)),
                window_i32=int(run_cfg.trace_window_i32),
            )

        if run_all or run_cfg.run_slack_sweep:
            _Log("Slack sweep")
            Run_Slack_Sweep(
                sim_cfg=cfg_run,
                lambda_f64=float(run_cfg.slack_lambda_f64),
                tfg_slack_factor_list_f64=slack_list,
                out_dir=os.path.join(outdir_run, "ablations"),
                tag_str=_Safe_Label(float(run_cfg.slack_lambda_f64)),
            )

        if (run_all or run_cfg.run_epsilon_sweep) and epsilon_list:
            _Log("Epsilon sweep")
            Run_Epsilon_Sweep(
                sim_cfg=cfg_run,
                lambda_f64=float(run_cfg.epsilon_sweep_lambda_f64),
                tfg_epsilon_list_f64=epsilon_list,
                out_dir=os.path.join(outdir_run, "ablations"),
                tag_str=_Safe_Label(float(run_cfg.epsilon_sweep_lambda_f64)),
                tfg_adaptive_epsilon_enabled_bool=bool(run_cfg.datp_adaptive_epsilon_enabled_bool),
            )

        if run_all or run_cfg.run_epsilon_pareto_compare:
            _Log("Epsilon pareto comparison")
            Run_Epsilon_Pareto_Comparison(
                sim_cfg=cfg_run,
                lambda_f64=float(run_cfg.epsilon_sweep_lambda_f64),
                policy_names=epsilon_pareto_policy_names,
                tfg_epsilon_list_f64=epsilon_list,
                out_dir=os.path.join(outdir_run, "ablations"),
                tag_str=_Safe_Label(float(run_cfg.epsilon_sweep_lambda_f64)),
                tfg_adaptive_epsilon_enabled_bool=bool(run_cfg.datp_adaptive_epsilon_enabled_bool),
            )

        if (run_all or run_cfg.run_epsilon_table) and epsilon_list:
            _Log("Epsilon control table")
            Write_Epsilon_Control_Table(
                sim_cfg=cfg_run,
                lambda_f64=float(run_cfg.epsilon_sweep_lambda_f64),
                tfg_epsilon_list_f64=epsilon_list,
                out_path=os.path.join(outdir_run, "tables", "epsilon_control_table.csv"),
            )

        if run_all or run_cfg.run_ttl_sweep:
            if not ttl_presets:
                _Log("TTL sweep")
                Run_Ttl_Sweep(
                    sim_cfg=cfg_run,
                    lambda_rate_list_f64=lambda_list,
                    ttl_seconds_list_f64=ttl_list,
                    out_dir=os.path.join(outdir_run, "ttl_sweep"),
                    tag_str="ttl",
                )

        if run_all or run_cfg.run_table:
            _Log("Results table")
            if not sweep_results:
                sweep_results = Run_Lambda_Sweep(
                    sim_cfg=cfg_run,
                    lambda_rate_list_f64=lambda_list,
                    policy_names=policy_names,
                    out_dir=os.path.join(outdir_run, "main_sweep"),
                    tag_str="main",
                )
            if lambda_list:
                load_points = [min(lambda_list), lambda_list[len(lambda_list) // 2], max(lambda_list)]
                Write_Results_Table(
                    sweep_results,
                    load_points=load_points,
                    out_path=os.path.join(outdir_run, "tables", "summary_table.csv"),
                )

        if (run_all or run_cfg.run_priority) and float(run_cfg.priority_task_rate_f64) > 0.0:
            _Log("Priority sweep")
            priority_results = Run_Priority_Sweep(
                sim_cfg=cfg_run,
                lambda_rate_list_f64=lambda_list,
                policy_name="DATPPolicy",
                priority_task_rate_f64=float(run_cfg.priority_task_rate_f64),
                priority_first_enabled_bool=bool(run_cfg.priority_first_enabled_bool),
                out_dir=os.path.join(outdir_run, "priority"),
                tag_str="priority",
            )
            if lambda_list:
                load_points = [min(lambda_list), lambda_list[len(lambda_list) // 2], max(lambda_list)]
                Write_Priority_Table(
                    priority_results,
                    load_points=load_points,
                    out_path=os.path.join(outdir_run, "tables", "priority_table.csv"),
                )


if __name__ == "__main__":
    main(RUN_CONFIG)
