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

import matplotlib.pyplot as plt
import numpy as np

from Configurations import Simulation_Config
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
    "miss_rate",
    "mean_utility",
    "response_time_p95",
    "slow_fraction",
    "waiting_time_p95",
    "queue_len_p95",
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
    outdir: str = "Results/Journal"
    num_reps_i32: int = 5
    seed_base_i32: int = 1453

    main_table_lambda_f64: float = 0.05
    main_table_ttl_presets: Sequence[str] = (
        "very_tight",
        "tight",
        "relaxed",
        "very_relaxed",
    )

    load_sweep_lambdas: Sequence[float] = (0.03, 0.05, 0.07, 0.09)
    load_sweep_ttl_preset: str = "tight"

    epsilon_sweep_lambda_f64: float = 0.05
    epsilon_list_f64: Sequence[float] = (-0.10, -0.05, 0.0, 0.05, 0.10)
    datp_adaptive_epsilon_enabled_bool: bool = False

    run_main_table: bool = True
    run_load_sweep: bool = True
    run_epsilon_tradeoff: bool = True


RUN_CONFIG = Run_Config()


def _Log(msg_str: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[Journal][{stamp}] {msg_str}", flush=True)


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
    if policy_name == "TTL_Feasibility_Bang_Bang_Policy":
        return TTL_Feasibility_Bang_Bang_Policy(sim_cfg.policy_config, service_model)
    if policy_name == "Drift_Penalty_Myopic_Policy":
        return Drift_Penalty_Myopic_Policy(sim_cfg.policy_config, service_model, sim_cfg.utility_config)
    if policy_name == "DATPPolicy":
        return DATPPolicy(sim_cfg.policy_config, service_model)
    raise ValueError(f"Unknown policy: {policy_name}")


def _Run_Single(sim_cfg: Simulation_Config, policy_name: str):
    interarrival = Exponential_Interarrival(sim_cfg.arrival_config.lambda_rate_f64)
    service_model = _Build_Service_Model(sim_cfg)
    utility_rng = np.random.default_rng(sim_cfg.seed_i32 + 2000)
    utility_model = Firm_Deadline_Quality_Utility(sim_cfg.utility_config, rng_opt=utility_rng)
    metrics = MetricsCollector(utility_model_utility_model=utility_model, warmup_time_f64=sim_cfg.warmup_time_f64)
    rng = np.random.default_rng(sim_cfg.seed_i32 + 1000)
    policy = _Build_Policy(policy_name, sim_cfg, service_model, rng)
    sim = Simulator(sim_cfg, interarrival, service_model, policy, metrics)
    agg = sim.Run()

    return agg, metrics


def _Extract_Metrics(agg: Dict[str, object], metrics: MetricsCollector) -> Dict[str, float]:
    resp_stats: SummaryStats = agg.get("response_time")  # type: ignore[assignment]
    wait_stats: SummaryStats = agg.get("waiting_time")  # type: ignore[assignment]
    mode_counts: Dict[str, int] = agg.get("mode_counts_completed")  # type: ignore[assignment]

    slow_count = int(mode_counts.get("slow", 0))
    fast_count = int(mode_counts.get("fast", 0))
    denom = slow_count + fast_count
    slow_fraction = float(slow_count / denom) if denom > 0 else float("nan")

    if metrics.queue_len_samples_list_i32:
        queue_len_p95 = float(np.percentile(metrics.queue_len_samples_list_i32, 95))
    else:
        queue_len_p95 = float("nan")

    return {
        "miss_rate": float(agg.get("miss_rate", float("nan"))),
        "mean_utility": float(agg.get("mean_utility", float("nan"))),
        "response_time_p95": float(resp_stats.p95_f64),
        "slow_fraction": slow_fraction,
        "waiting_time_p95": float(wait_stats.p95_f64),
        "queue_len_p95": queue_len_p95,
    }


def _Summarize_Runs(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key in REPORT_METRICS:
        vals = np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        if vals.size > 1:
            summary[f"{key}_std"] = float(np.nanstd(vals, ddof=1))
        else:
            summary[f"{key}_std"] = 0.0
    return summary


def _Make_Seeds(base_seed: int, count: int) -> List[int]:
    return [int(base_seed + i * 1000) for i in range(count)]


def _Write_Csv(rows: Sequence[Dict[str, object]], out_path: str) -> None:
    if not rows:
        return
    _Make_Dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def Run_Main_Table(run_cfg: Run_Config, sim_cfg: Simulation_Config) -> None:
    out_dir = os.path.join(run_cfg.outdir, "main_table")
    _Make_Dir(out_dir)

    rows: List[Dict[str, object]] = []
    seeds = _Make_Seeds(run_cfg.seed_base_i32, run_cfg.num_reps_i32)

    for preset_name in run_cfg.main_table_ttl_presets:
        cfg_ttl = _Apply_Ttl_Preset(sim_cfg, preset_name)
        cfg_lambda = replace(
            cfg_ttl,
            arrival_config=replace(cfg_ttl.arrival_config, lambda_rate_f64=float(run_cfg.main_table_lambda_f64)),
        )
        for policy_name in COMPARISON_POLICY_NAMES:
            run_rows: List[Dict[str, float]] = []
            for seed in seeds:
                cfg_seed = replace(
                    cfg_lambda,
                    seed_i32=int(seed),
                    policy_config=replace(
                        cfg_lambda.policy_config,
                        datp_adaptive_epsilon_enabled_bool=bool(run_cfg.datp_adaptive_epsilon_enabled_bool),
                    ),
                )
                agg, metrics = _Run_Single(cfg_seed, policy_name)
                run_rows.append(_Extract_Metrics(agg, metrics))

            stats = _Summarize_Runs(run_rows)
            row = {
                "ttl_preset": _Ttl_Label(preset_name),
                "lambda": float(run_cfg.main_table_lambda_f64),
                "policy_name": _Policy_Label(policy_name),
            }
            row.update(stats)
            rows.append(row)

    _Write_Csv(rows, os.path.join(out_dir, "main_table.csv"))


def _Plot_Load_Curve(rows: Sequence[Dict[str, object]], metric_key: str, out_path: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(6.5, 4))
    policies = sorted({str(r["policy_name"]) for r in rows})
    for policy in policies:
        xs = [float(r["lambda"]) for r in rows if str(r["policy_name"]) == policy]
        ys = [float(r[f"{metric_key}_mean"]) for r in rows if str(r["policy_name"]) == policy]
        plt.plot(xs, ys, marker="o", label=policy)
    plt.xlabel("arrival rate (lambda)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def Run_Load_Sweep(run_cfg: Run_Config, sim_cfg: Simulation_Config) -> None:
    out_dir = os.path.join(run_cfg.outdir, "load_sweep")
    _Make_Dir(out_dir)
    _Apply_Plot_Style()

    rows: List[Dict[str, object]] = []
    seeds = _Make_Seeds(run_cfg.seed_base_i32, run_cfg.num_reps_i32)

    cfg_ttl = _Apply_Ttl_Preset(sim_cfg, run_cfg.load_sweep_ttl_preset)
    for lambda_f64 in run_cfg.load_sweep_lambdas:
        cfg_lambda = replace(
            cfg_ttl,
            arrival_config=replace(cfg_ttl.arrival_config, lambda_rate_f64=float(lambda_f64)),
        )
        for policy_name in COMPARISON_POLICY_NAMES:
            run_rows: List[Dict[str, float]] = []
            for seed in seeds:
                cfg_seed = replace(
                    cfg_lambda,
                    seed_i32=int(seed),
                    policy_config=replace(
                        cfg_lambda.policy_config,
                        datp_adaptive_epsilon_enabled_bool=bool(run_cfg.datp_adaptive_epsilon_enabled_bool),
                    ),
                )
                agg, metrics = _Run_Single(cfg_seed, policy_name)
                run_rows.append(_Extract_Metrics(agg, metrics))
            stats = _Summarize_Runs(run_rows)
            row = {
                "ttl_preset": _Ttl_Label(run_cfg.load_sweep_ttl_preset),
                "lambda": float(lambda_f64),
                "policy_name": _Policy_Label(policy_name),
            }
            row.update(stats)
            rows.append(row)

    _Write_Csv(rows, os.path.join(out_dir, "load_sweep.csv"))

    _Plot_Load_Curve(
        rows,
        metric_key="mean_utility",
        out_path=os.path.join(out_dir, "utility_vs_lambda.png"),
        title=f"Utility vs lambda ({_Ttl_Label(run_cfg.load_sweep_ttl_preset)})",
        ylabel="mean realized utility",
    )
    _Plot_Load_Curve(
        rows,
        metric_key="miss_rate",
        out_path=os.path.join(out_dir, "dmr_vs_lambda.png"),
        title=f"DMR vs lambda ({_Ttl_Label(run_cfg.load_sweep_ttl_preset)})",
        ylabel="deadline miss rate",
    )


def Run_Epsilon_Tradeoff(run_cfg: Run_Config, sim_cfg: Simulation_Config) -> None:
    _Apply_Plot_Style()
    out_dir = os.path.join(run_cfg.outdir, "epsilon_tradeoff")
    _Make_Dir(out_dir)

    rows: List[Dict[str, object]] = []
    seeds = _Make_Seeds(run_cfg.seed_base_i32, run_cfg.num_reps_i32)

    cfg_ttl = _Apply_Ttl_Preset(sim_cfg, "tight")
    cfg_lambda = replace(
        cfg_ttl,
        arrival_config=replace(cfg_ttl.arrival_config, lambda_rate_f64=float(run_cfg.epsilon_sweep_lambda_f64)),
    )

    for eps in run_cfg.epsilon_list_f64:
        run_rows: List[Dict[str, float]] = []
        for seed in seeds:
            cfg_eps = replace(
                cfg_lambda,
                seed_i32=int(seed),
                policy_config=replace(
                    cfg_lambda.policy_config,
                    datp_epsilon_f64=float(eps),
                    datp_adaptive_epsilon_enabled_bool=bool(run_cfg.datp_adaptive_epsilon_enabled_bool),
                ),
            )
            agg, metrics = _Run_Single(cfg_eps, "DATPPolicy")
            run_rows.append(_Extract_Metrics(agg, metrics))

        stats = _Summarize_Runs(run_rows)
        row = {
            "epsilon": float(eps),
            "lambda": float(run_cfg.epsilon_sweep_lambda_f64),
            "policy_name": _Policy_Label("DATPPolicy"),
        }
        row.update(stats)
        rows.append(row)

    _Write_Csv(rows, os.path.join(out_dir, "epsilon_tradeoff.csv"))

    plt.figure(figsize=(6.0, 4))
    xs = [float(r["miss_rate_mean"]) for r in rows]
    ys = [float(r["mean_utility_mean"]) for r in rows]
    plt.plot(xs, ys, marker="o")
    for row in rows:
        plt.annotate(
            f'{row["epsilon"]:+.2f}',
            (float(row["miss_rate_mean"]), float(row["mean_utility_mean"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("deadline miss rate (DMR)")
    plt.ylabel("mean realized utility")
    plt.title(f"DATP epsilon trade-off ({_Ttl_Label('tight')}, lambda=0.05)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "datp_epsilon_pareto.png"))
    plt.close()


def main(run_cfg: Run_Config) -> None:
    sim_cfg = _Apply_Comparison_Policy_Config(Simulation_Config())

    if run_cfg.run_main_table:
        _Log("Running main table")
        Run_Main_Table(run_cfg, sim_cfg)
    if run_cfg.run_load_sweep:
        _Log("Running load sweep")
        Run_Load_Sweep(run_cfg, sim_cfg)
    if run_cfg.run_epsilon_tradeoff:
        _Log("Running epsilon trade-off")
        Run_Epsilon_Tradeoff(run_cfg, sim_cfg)


if __name__ == "__main__":
    main(RUN_CONFIG)
