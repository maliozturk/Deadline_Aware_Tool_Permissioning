# =============================================================================
#  CADTR Lambda Sweep -- "Hero Graph" Generator
#  Purpose: High-Priority Miss Rate vs. Arrival Rate (lambda)
#  Step 4 of CADTR migration
# =============================================================================

import csv
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from Configurations import (
    Arrival_Config,
    Simulation_Config,
)
from Core.Simulator import Simulator
from Metrics.Collector import MetricsCollector
from Models.Distributions import (
    Exponential_Interarrival,
    Lognormal_Service_Times,
    Trace_Service_Times,
    EWMA_Service_Time_Estimator,
)
from Models.Policies import (
    CADTR_Policy,
    Expected_Utility_Oracle_Policy,
    FTCPolicy,
    Mode_Aware_Baseline_Policy,
    TTL_Feasibility_Bang_Bang_Policy,
)
from Models.Utility import CADTR_Mission_Utility
from Core.Task import Mode


# ── Sweep configuration ─────────────────────────────────────────────────
LAMBDA_VALUES = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19]
PRIORITY_RATE = 0.25          # 25% critical tasks ("threat-burst" ratio)
CADTR_SAFETY_BUFFER = -0.20   # Tighter buffer for stress test

POLICY_DEFS = {
    "CADTR": "cadtr",
    "Myopic CDF Oracle": "cdf_oracle",
    "Mode-Aware Oracle": "mode_aware",
    "F-BB": "f_bb",
    "FTC*": "ftc_star",
}


# ── Helper: classify tasks by priority ───────────────────────────────────
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
        if not tasks:
            return float("nan")
        return float(np.mean([_task_missed(t) for t in tasks]))

    def _mean_utility(tasks, util_model):
        if not tasks:
            return float("nan")
        return float(np.mean([util_model.Utility(t) for t in tasks]))

    def _slow_frac(tasks):
        slow = sum(1 for t in tasks if t.chosen_mode_mode_opt == Mode.SLOW)
        total = sum(1 for t in tasks if t.chosen_mode_mode_opt in (Mode.SLOW, Mode.FAST))
        return float(slow / total) if total > 0 else float("nan")

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


# ── Build service model from config ─────────────────────────────────────
def _build_service_model(sim_config: Simulation_Config):
    service_cfg = sim_config.service_config
    source = str(service_cfg.service_time_source).lower()
    if source == "trace":
        base = Trace_Service_Times(
            csv_path_str=service_cfg.trace_csv_path_str,
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


# ── Build policy by key ──────────────────────────────────────────────────
def _build_policy(key: str, policy_cfg, service_model):
    if key == "cadtr":
        return CADTR_Policy(
            cfg=policy_cfg,
            service_model=service_model,
            priority_safety_buffer_f64=CADTR_SAFETY_BUFFER,
        )
    elif key == "cdf_oracle":
        return Expected_Utility_Oracle_Policy(
            cfg=policy_cfg,
            service_model=service_model,
        )
    elif key == "mode_aware":
        return Mode_Aware_Baseline_Policy(
            cfg=policy_cfg,
            service_model=service_model,
        )
    elif key == "f_bb":
        return TTL_Feasibility_Bang_Bang_Policy(
            cfg=policy_cfg,
            service_model=service_model,
        )
    elif key == "ftc_star":
        return FTCPolicy(
            cfg=policy_cfg,
            service_model=service_model,
        )
    else:
        raise ValueError(f"Unknown policy key: {key}")


# ── Main sweep ───────────────────────────────────────────────────────────
def run_sweep() -> None:
    out_dir = Path("Results") / "lambda_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for lam in LAMBDA_VALUES:
        print(f"\n{'='*60}")
        print(f"  lambda = {lam:.2f}  (inter-arrival = {1/lam:.1f}s)")
        print(f"{'='*60}")

        for policy_label, policy_key in POLICY_DEFS.items():
            # Build config for this lambda
            sim_config = Simulation_Config(
                priority_task_rate_f64=PRIORITY_RATE,
                arrival_config=Arrival_Config(lambda_rate_f64=lam),
            )

            service_model = _build_service_model(sim_config)
            interarrival = Exponential_Interarrival(lam)

            utility_rng = np.random.default_rng(sim_config.seed_i32 + 2000)
            utility_model = CADTR_Mission_Utility(
                cfg_utility_config=sim_config.utility_config,
                rng_opt=utility_rng,
            )

            metrics = MetricsCollector(
                utility_model_utility_model=utility_model,
                warmup_time_f64=sim_config.warmup_time_f64,
            )

            policy = _build_policy(policy_key, sim_config.policy_config, service_model)

            sim = Simulator(
                sim_config,
                interarrival,
                service_model,
                policy,
                metrics,
            )
            sim.Initialize()
            sim.Run()

            stats = _priority_class_stats(metrics)

            print(
                f"  {policy_label:25s} | "
                f"Premium MR: {stats['premium_miss_rate']:.4f}  "
                f"Std MR: {stats['standard_miss_rate']:.4f}  "
                f"Utility: {stats['overall_utility']:.4f}  "
                f"Prem Slow%: {stats['premium_slow_frac']:.2f}"
            )

            row = {
                "lambda": lam,
                "policy": policy_label,
                "premium_miss_rate": stats["premium_miss_rate"],
                "standard_miss_rate": stats["standard_miss_rate"],
                "overall_miss_rate": stats["overall_miss_rate"],
                "premium_utility": stats["premium_utility"],
                "standard_utility": stats["standard_utility"],
                "overall_utility": stats["overall_utility"],
                "premium_n": stats["premium_n"],
                "standard_n": stats["standard_n"],
                "premium_slow_frac": stats["premium_slow_frac"],
                "standard_slow_frac": stats["standard_slow_frac"],
            }
            all_rows.append(row)

    # ── Write CSV ────────────────────────────────────────────────────────
    csv_path = out_dir / "lambda_sweep_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[OK] Results written to {csv_path}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  HERO GRAPH DATA: High-Priority Miss Rate vs. lambda")
    print(f"{'='*80}")
    header = f"{'lambda':>6s}"
    for label in POLICY_DEFS:
        header += f" | {label:>22s}"
    print(header)
    print("-" * len(header))

    for lam in LAMBDA_VALUES:
        line = f"{lam:6.2f}"
        for label in POLICY_DEFS:
            row = next(r for r in all_rows if r["lambda"] == lam and r["policy"] == label)
            line += f" | {row['premium_miss_rate']:22.4f}"
        print(line)

    print(f"\n{'='*80}")
    print("  CADTR Shield Effect: Premium Slow-Mode Fraction vs. lambda")
    print(f"{'='*80}")
    header2 = f"{'lambda':>6s}"
    for label in POLICY_DEFS:
        header2 += f" | {label:>22s}"
    print(header2)
    print("-" * len(header2))

    for lam in LAMBDA_VALUES:
        line = f"{lam:6.2f}"
        for label in POLICY_DEFS:
            row = next(r for r in all_rows if r["lambda"] == lam and r["policy"] == label)
            line += f" | {row['premium_slow_frac']:22.2f}"
        print(line)


if __name__ == "__main__":
    run_sweep()
