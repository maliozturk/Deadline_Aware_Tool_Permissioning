"""
Regression test: J=2 mode must reproduce existing Table 2 numbers
within absolute tolerance 1e-6.

Seeds: seed_base=1453, 5 seeds → [1453, 2453, 3453, 4453, 5453]
TTL regimes: very_tight, tight, relaxed, very_relaxed
Policies: AF, AS, StaticMix, Q-BB, DP, H, FTC*
"""
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

import numpy as np
import pytest

from Configurations import Simulation_Config
from Core.Simulator import Simulator
from Metrics.Collector import MetricsCollector, SummaryStats
from Models.Distributions import (
    EWMA_Service_Time_Estimator,
    Exponential_Interarrival,
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
from Core.Task import Mode

# ── Reference data from existing Table 2 (main_table.csv) ───────────────────

REFERENCE_TABLE_PATH = REPO_ROOT / "Results" / "Journal" / "main_table" / "main_table.csv"

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

POLICY_LABELS: Dict[str, str] = {
    "Fcfs_Always_Fast": "AlwaysFast (AF)",
    "Fcfs_Always_Slow": "AlwaysSlow (AS)",
    "Static_Mix_Policy": "StaticMix (p=0.5)",
    "Queue_Length_Bang_Bang_Policy": "Q-BB",
    "Drift_Penalty_Myopic_Policy": "DriftPenaltyMyopic (V=1.0)",
    "Baseline_Heuristic_Policy": "TTL-aware heuristic",
    "FTCPolicy": "FTC*",
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

REPORT_METRICS = ("miss_rate", "mean_utility", "response_time_p95", "slow_fraction")

ABS_TOL = 1e-6


def _make_seeds(base: int, count: int) -> List[int]:
    return [base + i * 1000 for i in range(count)]


def _load_reference_table() -> Dict[Tuple[str, str], Dict[str, float]]:
    """Load reference Table 2 from CSV into a dict keyed by (regime_label, policy_label)."""
    ref = {}
    with open(REFERENCE_TABLE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ttl_preset = row["ttl_preset"].strip().strip('"')
            policy = row["policy_name"].strip()
            ref[(ttl_preset, policy)] = {
                "miss_rate": float(row["miss_rate_mean"]),
                "mean_utility": float(row["mean_utility_mean"]),
                "response_time_p95": float(row["response_time_p95_mean"]),
                "slow_fraction": float(row["slow_fraction_mean"]),
            }
    return ref


def _build_service_model(sim_cfg: Simulation_Config):
    service_cfg = sim_cfg.service_config
    base = Trace_Service_Times(
        csv_path_str=str(REPO_ROOT / service_cfg.trace_csv_path_str),
        fast_latency_column_str=service_cfg.trace_fast_column_str,
        slow_latency_column_str=service_cfg.trace_slow_column_str,
        drop_error_rows_bool=service_cfg.trace_drop_error_rows_bool,
        prompt_type_filter_opt=service_cfg.trace_prompt_type_filter_opt,
    )
    if service_cfg.ewma_enabled_bool:
        return EWMA_Service_Time_Estimator(
            base_model=base,
            alpha_f64=service_cfg.ewma_alpha_f64,
            warmup_count_i32=service_cfg.ewma_warmup_count_i32,
        )
    return base


def _build_policy(policy_name: str, sim_cfg: Simulation_Config, service_model, rng):
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


def _run_single(sim_cfg: Simulation_Config, policy_name: str):
    interarrival = Exponential_Interarrival(sim_cfg.arrival_config.lambda_rate_f64)
    service_model = _build_service_model(sim_cfg)
    utility_rng = np.random.default_rng(sim_cfg.seed_i32 + 2000)
    utility_model = Firm_Deadline_Quality_Utility(sim_cfg.utility_config, rng_opt=utility_rng)
    metrics = MetricsCollector(utility_model_utility_model=utility_model, warmup_time_f64=sim_cfg.warmup_time_f64)
    rng = np.random.default_rng(sim_cfg.seed_i32 + 1000)
    policy = _build_policy(policy_name, sim_cfg, service_model, rng)
    sim = Simulator(sim_cfg, interarrival, service_model, policy, metrics)
    agg = sim.Run()
    return agg, metrics


def _extract_metrics(agg: dict, metrics: MetricsCollector) -> Dict[str, float]:
    resp_stats: SummaryStats = agg.get("response_time")
    mode_counts: Dict[str, int] = agg.get("mode_counts_completed")
    slow_count = int(mode_counts.get("slow", 0))
    fast_count = int(mode_counts.get("fast", 0))
    denom = slow_count + fast_count
    slow_fraction = float(slow_count / denom) if denom > 0 else float("nan")
    return {
        "miss_rate": float(agg.get("miss_rate", float("nan"))),
        "mean_utility": float(agg.get("mean_utility", float("nan"))),
        "response_time_p95": float(resp_stats.p95_f64),
        "slow_fraction": slow_fraction,
    }


def _summarize_runs(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key in REPORT_METRICS:
        vals = np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
    return summary


def _run_table2_new() -> Dict[Tuple[str, str], Dict[str, float]]:
    """Run the full Table 2 with the new J=2 code and return results."""
    seeds = _make_seeds(1453, 5)
    base_cfg = Simulation_Config()
    # Apply comparison policy config (same as Journal_Experiments.py)
    base_cfg = replace(
        base_cfg,
        policy_config=replace(
            base_cfg.policy_config,
            static_mix_p=0.5,
            queue_threshold_tau=2,
            drift_V=1.0,
        ),
    )

    results = {}
    for preset_name in ("very_tight", "tight", "relaxed", "very_relaxed"):
        ttl_seconds, ttl_std = TTL_PRESETS[preset_name]
        cfg_ttl = replace(
            base_cfg,
            ttl_ttl_config=replace(
                base_cfg.ttl_ttl_config,
                ttl_seconds_f64=float(ttl_seconds),
                ttl_std_f64=float(ttl_std),
            ),
        )
        cfg_lambda = replace(
            cfg_ttl,
            arrival_config=replace(cfg_ttl.arrival_config, lambda_rate_f64=0.05),
        )
        for policy_name in COMPARISON_POLICY_NAMES:
            run_rows: List[Dict[str, float]] = []
            for seed in seeds:
                cfg_seed = replace(cfg_lambda, seed_i32=int(seed))
                agg, metrics = _run_single(cfg_seed, policy_name)
                run_rows.append(_extract_metrics(agg, metrics))
            stats = _summarize_runs(run_rows)
            regime_label = f"{TTL_REGIME_LABELS[preset_name]} (mu={ttl_seconds}, sigma={ttl_std})"
            policy_label = POLICY_LABELS[policy_name]
            results[(regime_label, policy_label)] = {
                "miss_rate": stats["miss_rate_mean"],
                "mean_utility": stats["mean_utility_mean"],
                "response_time_p95": stats["response_time_p95_mean"],
                "slow_fraction": stats["slow_fraction_mean"],
            }
    return results


class TestJModeRegression:
    """Regression: J=2 new simulator must exactly reproduce Table 2."""

    @pytest.fixture(scope="class")
    def reference(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        return _load_reference_table()

    @pytest.fixture(scope="class")
    def new_results(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        return _run_table2_new()

    def test_all_cells_match(self, reference, new_results):
        failures = []
        max_dev = 0.0
        for key in reference:
            if key not in new_results:
                failures.append(f"MISSING key {key} in new results")
                continue
            ref_vals = reference[key]
            new_vals = new_results[key]
            for metric in REPORT_METRICS:
                ref_v = ref_vals[metric]
                new_v = new_vals[metric]
                dev = abs(ref_v - new_v)
                max_dev = max(max_dev, dev)
                if dev > ABS_TOL:
                    failures.append(
                        f"{key[0]} | {key[1]} | {metric}: "
                        f"ref={ref_v:.10f} new={new_v:.10f} dev={dev:.2e}"
                    )
        if failures:
            msg = "\n".join(failures)
            pytest.fail(f"REGRESSION GATE FAILED — {len(failures)} cells differ:\n{msg}")


# ── Standalone runner for console output ─────────────────────────────────────

def main():
    print("=" * 100)
    print("  REGRESSION GATE: J=2 vs. existing Table 2")
    print("=" * 100)
    print()

    ref = _load_reference_table()
    new = _run_table2_new()

    # Print comparison table
    header = f"{'TTL Regime':<50} {'Policy':<30} {'Metric':<20} {'Ref':>14} {'New':>14} {'Dev':>12}"
    print(header)
    print("-" * len(header))

    failures = []
    max_dev_overall = 0.0

    for key in sorted(ref.keys()):
        regime, policy = key
        if key not in new:
            print(f"  MISSING: {regime} | {policy}")
            failures.append(f"MISSING: {regime} | {policy}")
            continue

        ref_vals = ref[key]
        new_vals = new[key]
        for metric in REPORT_METRICS:
            ref_v = ref_vals[metric]
            new_v = new_vals[metric]
            dev = abs(ref_v - new_v)
            max_dev_overall = max(max_dev_overall, dev)
            flag = " ***FAIL***" if dev > ABS_TOL else ""
            print(
                f"{regime:<50} {policy:<30} {metric:<20} "
                f"{ref_v:>14.10f} {new_v:>14.10f} {dev:>12.2e}{flag}"
            )
            if dev > ABS_TOL:
                failures.append(
                    f"{regime} | {policy} | {metric}: "
                    f"ref={ref_v:.10f} new={new_v:.10f} dev={dev:.2e}"
                )

    print()
    print(f"Max absolute deviation overall: {max_dev_overall:.2e}")
    print()

    if failures:
        print("REGRESSION GATE FAILED")
        print(f"  {len(failures)} cell(s) exceed tolerance {ABS_TOL}:")
        for f in failures:
            print(f"    {f}")
    else:
        print("REGRESSION GATE PASSED")
        print(f"  All cells match within tolerance {ABS_TOL}.")


if __name__ == "__main__":
    main()
