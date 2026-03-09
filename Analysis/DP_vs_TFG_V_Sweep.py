from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

from Configurations import Simulation_Config
from Core.Simulator import Simulator
from Metrics.Collector import MetricsCollector
from Models.Distributions import (
    EWMA_Service_Time_Estimator,
    Exponential_Interarrival,
    Lognormal_Service_Times,
    Trace_Service_Times,
)
from Models.Policies import DATPPolicy, Drift_Penalty_Myopic_Policy
from Models.Utility import Firm_Deadline_Quality_Utility


DEFAULT_V_LIST: Sequence[float] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


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


def _Run_Policy(sim_cfg: Simulation_Config, policy_name: str) -> Dict[str, float]:
    interarrival = Exponential_Interarrival(sim_cfg.arrival_config.lambda_rate_f64)
    service_model = _Build_Service_Model(sim_cfg)
    utility_rng = np.random.default_rng(sim_cfg.seed_i32 + 2000)
    utility_model = Firm_Deadline_Quality_Utility(sim_cfg.utility_config, rng_opt=utility_rng)
    metrics = MetricsCollector(utility_model_utility_model=utility_model, warmup_time_f64=sim_cfg.warmup_time_f64)

    if policy_name == "Drift_Penalty_Myopic_Policy":
        policy = Drift_Penalty_Myopic_Policy(sim_cfg.policy_config, service_model, sim_cfg.utility_config)
    elif policy_name == "DATPPolicy":
        policy = DATPPolicy(sim_cfg.policy_config, service_model)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    sim = Simulator(sim_cfg, interarrival, service_model, policy, metrics)
    agg = sim.Run()

    return {
        "miss_rate": float(agg.get("miss_rate", float("nan"))),
        "mean_utility": float(agg.get("mean_utility", float("nan"))),
    }


def _Write_Table(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _Plot(results: List[Dict[str, object]], out_path: str) -> None:
    _Apply_Plot_Style()
    dp_rows = [r for r in results if r["policy_name"] == "Drift_Penalty_Myopic_Policy"]
    tfg_row = next((r for r in results if r["policy_name"] == "DATPPolicy"), None)

    v_vals = [float(r["drift_V"]) for r in dp_rows]
    dp_miss = [float(r["miss_rate"]) for r in dp_rows]
    dp_util = [float(r["mean_utility"]) for r in dp_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    axes[0].plot(v_vals, dp_miss, marker="o", label="Drift-Penalty")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Drift-Penalty V")
    axes[0].set_ylabel("DMR (miss rate)")
    axes[0].grid(True, alpha=0.3)
    if tfg_row:
        axes[0].axhline(float(tfg_row["miss_rate"]), color="tab:orange", linestyle="--", label="DATP*")
    axes[0].legend(fontsize=9)

    axes[1].plot(v_vals, dp_util, marker="o", label="Drift-Penalty")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Drift-Penalty V")
    axes[1].set_ylabel("Avg utility")
    axes[1].grid(True, alpha=0.3)
    if tfg_row:
        axes[1].axhline(float(tfg_row["mean_utility"]), color="tab:orange", linestyle="--", label="DATP*")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def Run_DP_vs_TFG_V_Sweep(v_list: Sequence[float], out_dir: str) -> List[Dict[str, object]]:
    os.makedirs(out_dir, exist_ok=True)
    sim_cfg = Simulation_Config()

    results: List[Dict[str, object]] = []
    tfg_metrics = _Run_Policy(sim_cfg, "DATPPolicy")
    results.append(
        {
            "policy_name": "DATPPolicy",
            "drift_V": float("nan"),
            "miss_rate": float(tfg_metrics["miss_rate"]),
            "mean_utility": float(tfg_metrics["mean_utility"]),
        }
    )

    for v in tqdm(v_list):
        cfg_dp = replace(
            sim_cfg,
            policy_config=replace(sim_cfg.policy_config, drift_V=float(v)),
        )
        dp_metrics = _Run_Policy(cfg_dp, "Drift_Penalty_Myopic_Policy")
        results.append(
            {
                "policy_name": "Drift_Penalty_Myopic_Policy",
                "drift_V": float(v),
                "miss_rate": float(dp_metrics["miss_rate"]),
                "mean_utility": float(dp_metrics["mean_utility"]),
            }
        )

    table_path = os.path.join(out_dir, "dp_vs_tfg_v_sweep_table.csv")
    _Write_Table(results, table_path)

    plot_path = os.path.join(out_dir, "dp_vs_tfg_v_sweep.png")
    _Plot(results, plot_path)

    return results


if __name__ == "__main__":
    Run_DP_vs_TFG_V_Sweep(DEFAULT_V_LIST, out_dir="Results/dp_vs_tfg_v_sweep")
