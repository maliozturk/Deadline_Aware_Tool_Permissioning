# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
# ------------------------------------------------------------------------------
#  File: reproduce/run_sweep.py
#  Purpose: Arrival-rate (lambda) sweep that produces the paper's main results:
#           the critical-task "hero" curves and the representative table.
#           Runs NUM_REPLICATIONS independent seeds per (lambda, policy),
#           reporting mean, SD, 95% CI, and paired t-tests at lambda = 0.11.
#
#  TTL regime: deterministic (static) deadline budget of 35 s. This is the
#  regime that produced the reported results; the static-TTL model is described
#  in the paper's TTL-regime / static-sensitivity section.
#
#  Outputs (under Results/lambda_sweep/):
#     - sweep_raw.csv     : every replication
#     - sweep_stats.csv   : per-(lambda, policy) mean / SD / CI95  (figure source)
#     - main_table.csv    : tidy table at lambda in {0.03, 0.11, 0.19}
#
#  Usage:
#     python reproduce/run_sweep.py            # parallel (default)
#     python reproduce/run_sweep.py --serial   # no multiprocessing
#     python reproduce/run_sweep.py --reps 30  # override replications
# =============================================================================

import argparse
import csv
import multiprocessing
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Configurations import Arrival_Config, Simulation_Config, Ttl_Config
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
    CADTR_Policy,
    Expected_Utility_Oracle_Policy,
    FTCPolicy,
    Mode_Aware_Baseline_Policy,
    TTL_Feasibility_Bang_Bang_Policy,
)
from Models.Utility import CADTR_Mission_Utility

# ── Sweep configuration ──────────────────────────────────────────────────
LAMBDA_VALUES = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
PRIORITY_RATE = 0.25          # 25% critical tasks ("threat-burst" ratio)
CADTR_SAFETY_BUFFER = -0.20   # critical-event epsilon shift
NUM_REPLICATIONS = 30
SEED_BASE = 42

# TTL regime that reproduces the reported results: deterministic 35 s.
# (To explore the clipped-Normal sensitivity, set ttl_std_f64=10.0,
#  ttl_min_f64=15.0, ttl_max_f64=65.0 -- note this changes all results.)
TTL_CONFIG = Ttl_Config(
    ttl_seconds_f64=35.0,
    ttl_std_f64=0.0,
    ttl_min_f64=0.05,
)

POLICY_DEFS = {
    "CADTR": "cadtr",
    "Myopic CDF Oracle": "cdf_oracle",
    "Mode-Aware Oracle": "mode_aware",
    "F-BB": "f_bb",
    "FTC*": "ftc_star",
}

REPRESENTATIVE_LAMBDAS = [0.03, 0.11, 0.19]


def _task_missed(task) -> int:
    if task.dropped_in_queue_bool:
        return 1
    if task.completion_time_f64_opt is None:
        return 1
    return 1 if float(task.completion_time_f64_opt) > float(task.deadline) else 0


def _priority_class_stats(metrics: MetricsCollector) -> Dict[str, Any]:
    all_tasks = (
        list(metrics.completed_tasks_list_task)
        + list(metrics.dropped_tasks_list_task)
        + list(metrics.unfinished_tasks_list_task)
    )
    premium = [t for t in all_tasks if bool(getattr(t, "high_priority_bool", False))]
    standard = [t for t in all_tasks if not bool(getattr(t, "high_priority_bool", False))]

    def _miss_rate(tasks):
        return float(np.mean([_task_missed(t) for t in tasks])) if tasks else 0.0

    def _mean_utility(tasks, util_model):
        return float(np.mean([util_model.Utility(t) for t in tasks])) if tasks else 0.0

    def _slow_frac(tasks):
        slow = sum(1 for t in tasks if t.chosen_mode_mode_opt == Mode.SLOW)
        total = sum(1 for t in tasks if t.chosen_mode_mode_opt in (Mode.SLOW, Mode.FAST))
        return float(slow / total) if total > 0 else 0.0

    util_model = metrics.utility_model_utility_model
    return {
        "premium_miss_rate": _miss_rate(premium),
        "standard_miss_rate": _miss_rate(standard),
        "overall_miss_rate": _miss_rate(all_tasks),
        "premium_utility": _mean_utility(premium, util_model),
        "standard_utility": _mean_utility(standard, util_model),
        "overall_utility": _mean_utility(all_tasks, util_model),
        "premium_n": len(premium),
        "standard_n": len(standard),
        "premium_slow_frac": _slow_frac(premium),
        "standard_slow_frac": _slow_frac(standard),
    }


def _build_service_model(sim_config: Simulation_Config):
    service_cfg = sim_config.service_config
    source = str(service_cfg.service_time_source).lower()
    if source == "trace":
        base = Trace_Service_Times(
            csv_path_str=str(REPO_ROOT / service_cfg.trace_csv_path_str),
            fast_latency_column_str=service_cfg.trace_fast_column_str,
            slow_latency_column_str=service_cfg.trace_slow_column_str,
            drop_error_rows_bool=service_cfg.trace_drop_error_rows_bool,
            prompt_type_filter_opt=service_cfg.trace_prompt_type_filter_opt,
        )
    elif source == "lognormal":
        base = Lognormal_Service_Times(
            slow_mu_f64=service_cfg.slow_logn_mu_f64,
            slow_sigma_f64=service_cfg.slow_logn_sigma_f64,
            fast_mu_f64=service_cfg.fast_logn_mu_f64,
            fast_sigma_f64=service_cfg.fast_logn_sigma_f64,
        )
    else:
        raise ValueError(f"Unknown service_time_source: {source}")

    if service_cfg.ewma_enabled_bool:
        return EWMA_Service_Time_Estimator(
            base_model=base,
            alpha_f64=service_cfg.ewma_alpha_f64,
            warmup_count_i32=service_cfg.ewma_warmup_count_i32,
        )
    return base


def _build_policy(key: str, policy_cfg, service_model):
    if key == "cadtr":
        return CADTR_Policy(cfg=policy_cfg, service_model=service_model,
                            priority_safety_buffer_f64=CADTR_SAFETY_BUFFER)
    if key == "cdf_oracle":
        return Expected_Utility_Oracle_Policy(cfg=policy_cfg, service_model=service_model)
    if key == "mode_aware":
        return Mode_Aware_Baseline_Policy(cfg=policy_cfg, service_model=service_model)
    if key == "f_bb":
        return TTL_Feasibility_Bang_Bang_Policy(cfg=policy_cfg, service_model=service_model)
    if key == "ftc_star":
        return FTCPolicy(cfg=policy_cfg, service_model=service_model)
    raise ValueError(f"Unknown policy key: {key}")


def run_single_simulation(args) -> Dict[str, Any]:
    lam, policy_label, policy_key, rep_idx = args
    sim_config = Simulation_Config(
        seed_i32=SEED_BASE + rep_idx,
        priority_task_rate_f64=PRIORITY_RATE,
        arrival_config=Arrival_Config(lambda_rate_f64=lam),
        ttl_ttl_config=TTL_CONFIG,
    )
    service_model = _build_service_model(sim_config)
    interarrival = Exponential_Interarrival(lam)
    utility_model = CADTR_Mission_Utility(
        cfg_utility_config=sim_config.utility_config,
        rng_opt=np.random.default_rng(sim_config.seed_i32 + 2000),
    )
    metrics = MetricsCollector(
        utility_model_utility_model=utility_model,
        warmup_time_f64=sim_config.warmup_time_f64,
    )
    policy = _build_policy(policy_key, sim_config.policy_config, service_model)
    sim = Simulator(sim_config, interarrival, service_model, policy, metrics)
    sim.Initialize()
    sim.Run()

    stats = _priority_class_stats(metrics)
    stats.update({"lambda": lam, "policy": policy_label, "replication": rep_idx})
    return stats


def main() -> None:
    global NUM_REPLICATIONS
    parser = argparse.ArgumentParser(description="Lambda sweep (paper main results).")
    parser.add_argument("--serial", action="store_true",
                        help="Run without multiprocessing (slower, but works "
                             "where process spawning is restricted).")
    parser.add_argument("--reps", type=int, default=NUM_REPLICATIONS,
                        help=f"Replications per cell (default {NUM_REPLICATIONS}).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = all CPUs). Use a small value "
                             "to limit CPU load.")
    args = parser.parse_args()
    NUM_REPLICATIONS = args.reps

    tasks = [(lam, label, key, rep)
             for lam in LAMBDA_VALUES
             for label, key in POLICY_DEFS.items()
             for rep in range(NUM_REPLICATIONS)]
    print(f"Total runs: {len(tasks)} | TTL=N(35,10) clip[15,65] | reps={NUM_REPLICATIONS}")

    if args.serial:
        print("Running serially...")
        results = [run_single_simulation(t) for t in tasks]
    else:
        n = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        print(f"Running on {n} workers...")
        with multiprocessing.Pool(n) as pool:
            results = pool.map(run_single_simulation, tasks)

    out_dir = REPO_ROOT / "Results" / "lambda_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "sweep_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault((r["lambda"], r["policy"]), []).append(r)

    metrics_keys = ["premium_miss_rate", "standard_miss_rate", "overall_miss_rate",
                    "premium_utility", "standard_utility", "overall_utility",
                    "premium_slow_frac", "standard_slow_frac"]
    stats_headers = ["lambda", "policy"]
    for m in metrics_keys:
        stats_headers += [f"{m}_mean", f"{m}_std", f"{m}_ci95"]

    stats_rows = []
    for (lam, policy), reps in grouped.items():
        row = {"lambda": lam, "policy": policy}
        for m in metrics_keys:
            vals = [r[m] for r in reps]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_ci95"] = 1.96 * std / np.sqrt(len(vals))
        stats_rows.append(row)

    stats_path = out_dir / "sweep_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=stats_headers)
        w.writeheader()
        w.writerows(stats_rows)
    print(f"Wrote {raw_path}\nWrote {stats_path}")

    # Representative table (CSV).
    table_path = out_dir / "main_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "policy", "crit_mr_pct", "crit_mr_ci95_pct",
                    "overall_mr_pct", "overall_util", "crit_strat_pct"])
        for lam in REPRESENTATIVE_LAMBDAS:
            for policy in POLICY_DEFS:
                r = next(s for s in stats_rows if s["lambda"] == lam and s["policy"] == policy)
                w.writerow([
                    lam, policy,
                    round(r["premium_miss_rate_mean"] * 100, 2),
                    round(r["premium_miss_rate_ci95"] * 100, 2),
                    round(r["overall_miss_rate_mean"] * 100, 2),
                    round(r["overall_utility_mean"], 3),
                    round(r["premium_slow_frac_mean"] * 100, 1),
                ])
    print(f"Wrote {table_path}")

    # Paired t-tests at lambda = 0.11.
    def col(policy):
        return [r["premium_miss_rate"] for r in results
                if r["lambda"] == 0.11 and r["policy"] == policy]
    cadtr = col("CADTR")
    print("\nPaired t-tests at lambda=0.11 (critical miss rate):")
    for other in ["Myopic CDF Oracle", "Mode-Aware Oracle", "F-BB"]:
        t, p = sp_stats.ttest_rel(cadtr, col(other))
        print(f"  CADTR vs {other:18s}: t={t:.3f}, p={p:.3e}")

    print("\nRepresentative table:")
    for lam in REPRESENTATIVE_LAMBDAS:
        print(f"\nlambda = {lam}:")
        for policy in POLICY_DEFS:
            r = next(s for s in stats_rows if s["lambda"] == lam and s["policy"] == policy)
            print(f"  {policy:18s} CritMR {r['premium_miss_rate_mean']*100:5.2f}% "
                  f"| OvrMR {r['overall_miss_rate_mean']*100:5.2f}% "
                  f"| Util {r['overall_utility_mean']:.3f} "
                  f"| CritStrat {r['premium_slow_frac_mean']*100:4.1f}%")


if __name__ == "__main__":
    main()
