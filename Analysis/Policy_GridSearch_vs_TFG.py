from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from Models.Policies import (
    Baseline_Heuristic_Policy,
    DATPPolicy,
    Queue_Length_Bang_Bang_Policy,
    TTL_Feasibility_Bang_Bang_Policy,
)
from Models.Utility import Firm_Deadline_Quality_Utility


DEFAULT_L_THRESHOLD_LIST: Sequence[int] = (2, 4, 6, 8, 10, 12)
DEFAULT_ALPHA_LIST: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_QT_TAU_LIST: Sequence[int] = (2, 4, 6, 8, 10, 12, 15, 20)


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

    if policy_name == "Baseline_Heuristic_Policy":
        policy = Baseline_Heuristic_Policy(sim_cfg.policy_config, service_model)
    elif policy_name == "Queue_Length_Bang_Bang_Policy":
        policy = Queue_Length_Bang_Bang_Policy(sim_cfg.policy_config)
    elif policy_name == "TTL_Feasibility_Bang_Bang_Policy":
        policy = TTL_Feasibility_Bang_Bang_Policy(sim_cfg.policy_config, service_model)
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


def _Plot_Grid_Comparison(
    heuristic_rows: Sequence[Dict[str, object]],
    qt_rows: Sequence[Dict[str, object]],
    tfg_row: Dict[str, object],
    out_path: str,
) -> None:
    _Apply_Plot_Style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    def _Scatter(ax, rows, title, marker):
        xs = [float(r["miss_rate"]) for r in rows]
        ys = [float(r["mean_utility"]) for r in rows]
        ax.scatter(xs, ys, marker=marker, alpha=0.8, label=title)
        ax.scatter(
            [float(tfg_row["miss_rate"])],
            [float(tfg_row["mean_utility"])],
            marker="*",
            s=140,
            color="tab:orange",
            label="DATP*",
            zorder=3,
        )
        ax.set_xlabel("DMR (miss rate)")
        ax.set_ylabel("Avg utility")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _Scatter(axes[0], heuristic_rows, "Heuristic grid", "o")
    _Scatter(axes[1], qt_rows, "Queue-Threshold grid", "s")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def Run_GridSearch_vs_TFG(
    heuristic_grid: Sequence[Tuple[int, float]],
    qt_tau_list: Sequence[int],
    out_dir: str,
) -> List[Dict[str, object]]:
    os.makedirs(out_dir, exist_ok=True)
    sim_cfg = Simulation_Config()

    results: List[Dict[str, object]] = []

    datp_metrics = _Run_Policy(sim_cfg, "DATPPolicy")
    tfg_row = {
        "policy_name": "DATPPolicy",
        "L_threshold_i32": float("nan"),
        "alpha_f64": float("nan"),
        "queue_threshold_tau": float("nan"),
        "miss_rate": float(datp_metrics["miss_rate"]),
        "mean_utility": float(datp_metrics["mean_utility"]),
    }
    results.append(tfg_row)

    heuristic_rows: List[Dict[str, object]] = []
    for L_threshold, alpha in tqdm(heuristic_grid):
        cfg_h = replace(
            sim_cfg,
            policy_config=replace(
                sim_cfg.policy_config,
                L_threshold_i32=int(L_threshold),
                alpha_f64=float(alpha),
            ),
        )
        metrics = _Run_Policy(cfg_h, "Baseline_Heuristic_Policy")
        row = {
            "policy_name": "Baseline_Heuristic_Policy",
            "L_threshold_i32": int(L_threshold),
            "alpha_f64": float(alpha),
            "queue_threshold_tau": float("nan"),
            "miss_rate": float(metrics["miss_rate"]),
            "mean_utility": float(metrics["mean_utility"]),
        }
        heuristic_rows.append(row)
        results.append(row)

    qt_rows: List[Dict[str, object]] = []
    for tau in qt_tau_list:
        cfg_qt = replace(
            sim_cfg,
            policy_config=replace(
                sim_cfg.policy_config,
                queue_threshold_tau=int(tau),
            ),
        )
        metrics = _Run_Policy(cfg_qt, "Queue_Length_Bang_Bang_Policy")
        row = {
            "policy_name": "Queue_Length_Bang_Bang_Policy",
            "L_threshold_i32": float("nan"),
            "alpha_f64": float("nan"),
            "queue_threshold_tau": int(tau),
            "miss_rate": float(metrics["miss_rate"]),
            "mean_utility": float(metrics["mean_utility"]),
        }
        qt_rows.append(row)
        results.append(row)

    fbb_metrics = _Run_Policy(sim_cfg, "TTL_Feasibility_Bang_Bang_Policy")
    results.append(
        {
            "policy_name": "TTL_Feasibility_Bang_Bang_Policy",
            "L_threshold_i32": float("nan"),
            "alpha_f64": float("nan"),
            "queue_threshold_tau": float("nan"),
            "miss_rate": float(fbb_metrics["miss_rate"]),
            "mean_utility": float(fbb_metrics["mean_utility"]),
        }
    )

    table_path = os.path.join(out_dir, "policy_grid_vs_tfg_table.csv")
    _Write_Table(results, table_path)

    plot_path = os.path.join(out_dir, "policy_grid_vs_tfg.png")
    _Plot_Grid_Comparison(heuristic_rows, qt_rows, tfg_row, plot_path)

    return results


def _Build_Heuristic_Grid(
    L_threshold_list: Sequence[int],
    alpha_list: Sequence[float],
) -> List[Tuple[int, float]]:
    return [(int(L), float(a)) for L in L_threshold_list for a in alpha_list]


if __name__ == "__main__":
    heuristic_grid = _Build_Heuristic_Grid(DEFAULT_L_THRESHOLD_LIST, DEFAULT_ALPHA_LIST)
    Run_GridSearch_vs_TFG(
        heuristic_grid=heuristic_grid,
        qt_tau_list=DEFAULT_QT_TAU_LIST,
        out_dir="Results/policy_grid_vs_tfg",
    )
