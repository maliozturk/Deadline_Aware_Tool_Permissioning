# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Run_Simulation.py
#  Purpose: Run simulations, compute policy summaries, and write outputs.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Metrics.Collector import SummaryStats

from Analysis.Process_Analysis import (
    Process_Analyzer,
    Simulation_Artifacts,
    Timeline_Series,
)
from Configurations import Simulation_Config
from Core.Simulator import Simulator
from Metrics.Collector import MetricsCollector
from Metrics.Reports import Format_Summary
from Models.Distributions import (
    Exponential_Interarrival,
    Lognormal_Service_Times,
    EWMA_Service_Time_Estimator,
    Trace_Service_Times,
)
from Models.Policies import (
    Baseline_Heuristic_Policy,
    Fcfs_Always_Fast_Policy,
    Fcfs_Always_Slow_Policy,
    Static_Mix_Policy,
    Queue_Threshold_Policy,
    Drift_Penalty_Myopic_Policy,
    DATPPolicy,
)
from Core.Task import Mode
from Models.Utility import Firm_Deadline_Quality_Utility


POLICY_LABELS = {
    "Fcfs_Always_Fast": "AlwaysFast (AF)",
    "Fcfs_Always_Slow": "AlwaysSlow (AS)",
    "Static_Mix_Policy": "StaticMix (p=0.5)",
    "Queue_Threshold_Policy": "QueueThreshold (Q0=5)",
    "Drift_Penalty_Myopic_Policy": "DriftPenaltyMyopic (V=1.0)",
    "Baseline_Heuristic_Policy": "TTL-aware heuristic",
    "DATPPolicy": "DATP*",
}


def _Task_Missed(task_obj) -> int:
    if task_obj.dropped_in_queue_bool:
        return 1
    if task_obj.completion_time_f64_opt is None:
        return 1
    return 1 if float(task_obj.completion_time_f64_opt) > float(task_obj.deadline) else 0


def _Class_Miss_Rate(tasks_list) -> float:
    if not tasks_list:
        return float("nan")
    miss_flags = [_Task_Missed(t) for t in tasks_list]
    return float(np.mean(miss_flags)) if miss_flags else float("nan")


def _Class_Summary(tasks_list, utility_model) -> dict:
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


def _Priority_Class_Summaries(metrics: MetricsCollector) -> dict:
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
        "overall": _Class_Summary(tasks_all, util_model),
    }


def _Priority_Class_Miss_Rates(metrics: MetricsCollector) -> dict:
    summaries = _Priority_Class_Summaries(metrics)
    return {
        "premium": summaries["premium"]["miss_rate"],
        "standard": summaries["standard"]["miss_rate"],
    }


def _Write_Priority_Policy_Table(priority_policy_summaries: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_order = [
        "Fcfs_Always_Fast",
        "Baseline_Heuristic_Policy",
        "Drift_Penalty_Myopic_Policy",
        "DATPPolicy",
    ]
    policy_short = {
        "Fcfs_Always_Fast": "AF",
        "Baseline_Heuristic_Policy": "H",
        "Drift_Penalty_Myopic_Policy": "DP",
        "DATPPolicy": "DATP*",
    }
    class_order = [("premium", "Premium"), ("standard", "Normal"), ("overall", "Overall")]

    rows = []
    for class_key, class_label in class_order:
        for policy_name in policy_order:
            if policy_name not in priority_policy_summaries:
                continue
            stats = priority_policy_summaries[policy_name].get(class_key)
            if not stats:
                continue
            rows.append(
                {
                    "Class": class_label,
                    "Policy": policy_short.get(policy_name, policy_name),
                    "Utility": float(stats["mean_utility"]),
                    "DMR": float(stats["miss_rate"]),
                    "p95_Response": float(stats["response_time_p95"]),
                    "Slow_fraction": float(stats["slow_fraction"]),
                }
            )

    if not rows:
        return

    out_path = out_dir / "priority_policy_table.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _Plot_Priority_Miss_Rate_Comparison(
    policy_miss_rates: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = list(policy_miss_rates.keys())
    labels = [POLICY_LABELS.get(p, p) for p in policies]
    premium_vals = [float(policy_miss_rates[p]["premium"]) for p in policies]
    standard_vals = [float(policy_miss_rates[p]["standard"]) for p in policies]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, vals, title in zip(
        axes,
        [premium_vals, standard_vals],
        ["Premium miss rate", "Standard miss rate"],
    ):
        ax.bar(range(len(labels)), vals, color="tab:blue", alpha=0.75)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel("miss rate")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "priority_miss_rate_comparison.png"
    plt.savefig(out_path)
    plt.close(fig)

def Run_Baseline() -> None:
    simulation_config = Simulation_Config()
    policy_names = [
        "Fcfs_Always_Fast",
        "Fcfs_Always_Slow",
        "Baseline_Heuristic_Policy",
        "Static_Mix_Policy",
        "Queue_Threshold_Policy",
        "Drift_Penalty_Myopic_Policy",
        "DATPPolicy",
    ]

    Path("Results").mkdir(parents=True, exist_ok=True)
    base = Path("Results")
    results_path = base / "results.csv"

    header = ("Policy_ID,Total_Tasks,CompletedTaskCount,AbandonTaskCount,"
              "UnfinishedTaskCount,MissRate,AvgUtility,AvgResponseTime,"
              "ResponseTimeP50,ResponseTimeP90,AvgWT,WTP50,WTP90,"
              "SLOW_MODE_COUNT,FAST_MODE_COUNT,Avg_QLength")
    if not results_path.exists():
        with open(results_path, "a", encoding="utf-8", newline="") as results_file:
            print(header, file=results_file)

    priority_miss_rates: dict = {}
    priority_policy_summaries: dict = {}

    for policy_name in policy_names:
        simulation_config = Simulation_Config()

        interarrival_exponential = Exponential_Interarrival(simulation_config.arrival_config.lambda_rate_f64)
        service_cfg = simulation_config.service_config
        service_source = str(service_cfg.service_time_source).lower()
        if service_source == "trace":
            base_service_model = Trace_Service_Times(
                csv_path_str=service_cfg.trace_csv_path_str,
                fast_latency_column_str=service_cfg.trace_fast_column_str,
                slow_latency_column_str=service_cfg.trace_slow_column_str,
                drop_error_rows_bool=service_cfg.trace_drop_error_rows_bool,
                prompt_type_filter_opt=service_cfg.trace_prompt_type_filter_opt,
            )
        elif service_source == "lognormal":
            base_service_model = Lognormal_Service_Times(
                slow_mu_f64     =   service_cfg.slow_logn_mu_f64,
                slow_sigma_f64  =   service_cfg.slow_logn_sigma_f64,
                fast_mu_f64     =   service_cfg.fast_logn_mu_f64,
                fast_sigma_f64  =   service_cfg.fast_logn_sigma_f64,
            )
        else:
            raise ValueError(f"Unknown service_time_source: {service_cfg.service_time_source}")
        if simulation_config.service_config.ewma_enabled_bool:
            service_lognormal_service_times = EWMA_Service_Time_Estimator(
                base_model=base_service_model,
                alpha_f64=simulation_config.service_config.ewma_alpha_f64,
                warmup_count_i32=simulation_config.service_config.ewma_warmup_count_i32,
            )
        else:
            service_lognormal_service_times = base_service_model

        utility_rng = np.random.default_rng(simulation_config.seed_i32 + 2000)
        utility_model_firm_deadline_quality_utility = Firm_Deadline_Quality_Utility(
            simulation_config.utility_config,
            rng_opt=utility_rng,
        )

        policy_label = POLICY_LABELS.get(policy_name, policy_name)
        print(f"Running Policy: {policy_label}")
        metrics_collector = MetricsCollector(
            utility_model_utility_model=utility_model_firm_deadline_quality_utility,
            warmup_time_f64=simulation_config.warmup_time_f64,
        )

        if policy_name == "Fcfs_Always_Fast":
            policy_scheduling_policy = Fcfs_Always_Fast_Policy()

        elif policy_name == "Fcfs_Always_Slow":
            policy_scheduling_policy = Fcfs_Always_Slow_Policy()

        elif policy_name == "Baseline_Heuristic_Policy":
            policy_scheduling_policy = Baseline_Heuristic_Policy(simulation_config.policy_config,
                                                                 service_lognormal_service_times)

        elif policy_name == "Static_Mix_Policy":
            policy_scheduling_policy  = Static_Mix_Policy(simulation_config.policy_config,
                                                          rng=np.random.default_rng(simulation_config.seed_i32 + 1000))

        elif policy_name == "Queue_Threshold_Policy":
            policy_scheduling_policy  = Queue_Threshold_Policy(simulation_config.policy_config)

        elif policy_name == "Drift_Penalty_Myopic_Policy":
            policy_scheduling_policy  = Drift_Penalty_Myopic_Policy(simulation_config.policy_config,
                                                                    service_model=service_lognormal_service_times,
                                                                    utility_cfg=simulation_config.utility_config)

        elif policy_name == "DATPPolicy":
            policy_scheduling_policy  =  DATPPolicy(simulation_config.policy_config, service_model=service_lognormal_service_times)

        else:
            raise ValueError(f"Policy name {policy_name} not recognized.")

        sim_simulator = Simulator(
            simulation_config,
            interarrival_exponential,
            service_lognormal_service_times,
            policy_scheduling_policy,
            metrics_collector,
        )
        agg_dict_obj = sim_simulator.Run()

        epsilon_timeline_opt = None
        if hasattr(policy_scheduling_policy, "epsilon_trace_times_list_f64") and hasattr(
            policy_scheduling_policy, "epsilon_trace_values_list_f64"
        ):
            eps_times = list(getattr(policy_scheduling_policy, "epsilon_trace_times_list_f64"))
            eps_vals = list(getattr(policy_scheduling_policy, "epsilon_trace_values_list_f64"))
            if eps_times and eps_vals and len(eps_times) == len(eps_vals):
                epsilon_timeline_opt = Timeline_Series(eps_times, eps_vals)

        artifacts_simulation_artifacts = Simulation_Artifacts(
            aggregate=agg_dict_obj,
            completed_tasks=metrics_collector.completed_tasks_list_task,
            dropped_tasks=metrics_collector.dropped_tasks_list_task,
            unfinished_tasks=metrics_collector.unfinished_tasks_list_task,
            queue_length_timeline=Timeline_Series(
                metrics_collector.queue_len_sample_times_list_f64,
                metrics_collector.queue_len_samples_list_i32,
            ),
            server_busy_timeline=Timeline_Series(
                metrics_collector.server_busy_sample_times_list_f64,
                metrics_collector.server_busy_samples_list_i32,
            ),
            queue_nonempty_timeline=Timeline_Series(
                metrics_collector.queue_nonempty_sample_times_list_f64,
                metrics_collector.queue_nonempty_samples_list_i32,
            ),
            ewma_slow_timeline=Timeline_Series(
                metrics_collector.ewma_sample_times_list_f64,
                metrics_collector.ewma_slow_samples_list_f64,
            ),
            ewma_fast_timeline=Timeline_Series(
                metrics_collector.ewma_sample_times_list_f64,
                metrics_collector.ewma_fast_samples_list_f64,
            ),
            ewma_slow_count_timeline=Timeline_Series(
                metrics_collector.ewma_sample_times_list_f64,
                metrics_collector.ewma_slow_counts_list_i32,
            ),
            ewma_fast_count_timeline=Timeline_Series(
                metrics_collector.ewma_sample_times_list_f64,
                metrics_collector.ewma_fast_counts_list_i32,
            ),
            epsilon_timeline=epsilon_timeline_opt,
        )

        policy_dir = base / policy_name
        policy_dir.mkdir(parents=True, exist_ok=True)
        analyzer_process_analyzer = Process_Analyzer(Results_Path=str(policy_dir))

        analysis_results_dict_obj = analyzer_process_analyzer.Analyze(
            artifacts_simulation_artifacts,
            make_plots_bool=True,
        )
        _ = analysis_results_dict_obj

        print(Format_Summary(agg_dict_obj))
        print()

        """
        Extending with additional Metrics (generic requirement)
    
        To add a new metric:
        Create a new class implementing MetricPlugin:
    
        name: str
        compute(ctx) -> Dict[str, object]
        optional plot(ctx)
    
        Add it to the analyzer’s plugin list:
        analyzer = ProcessAnalyzer(plugins=[..., MyNewMetricPlugin()])
    
        No changes to simulator are needed unless the new metric
        requires new traces. In that case, add another record_*
        sampler in MetricsCollector and call it at the same
        event sampling points.
    
        policy = BaselineHeuristicPolicy(cfg.policy, service)
        policy2 = FCFSAlwaysSlowPolicy()
        policy3 = FCFSAlwaysFastPolicy()
        """

        resp_summary_stats: SummaryStats = agg_dict_obj.get("response_time")                            
        wait_summary_stats: SummaryStats = agg_dict_obj.get("waiting_time")                            

        n_tasks_total = agg_dict_obj.get('n_tasks_total')
        n_completed = agg_dict_obj.get('n_completed')

        abandon_n = agg_dict_obj.get('n_dropped_in_queue')
        unfinished_n = agg_dict_obj.get('n_unfinished')
        miss_rate = agg_dict_obj.get('miss_rate')
        avg_utility = agg_dict_obj.get('mean_utility')

        response_time_avg = resp_summary_stats.mean_f64
        response_time_p50 = resp_summary_stats.p50_f64
        response_time_p90 = resp_summary_stats.p90_f64

        wt_avg = wait_summary_stats.mean_f64
        wt_p50 = wait_summary_stats.p50_f64
        wt_p90 = wait_summary_stats.p90_f64

        SLOW_COUNT = agg_dict_obj.get('mode_counts_completed')["slow"]
        FAST_COUNT = agg_dict_obj.get('mode_counts_completed')["fast"]

        avg_q_length = agg_dict_obj.get('queue_len_mean')

        with open(results_path, "a", encoding="utf-8", newline="") as results_file:
            print(
                f"{policy_label},{n_tasks_total},{n_completed},{abandon_n},{unfinished_n},{miss_rate},{avg_utility},{response_time_avg},{response_time_p50},{response_time_p90},{wt_avg},{wt_p50},{wt_p90},{SLOW_COUNT},{FAST_COUNT},{avg_q_length}",
                file=results_file)

        if float(simulation_config.priority_task_rate_f64) > 0.0:
            priority_miss_rates[policy_name] = _Priority_Class_Miss_Rates(metrics_collector)
            priority_policy_summaries[policy_name] = _Priority_Class_Summaries(metrics_collector)

    if priority_miss_rates:
        datp_tables_dir = base / "DATPPolicy" / "tables"
        _Plot_Priority_Miss_Rate_Comparison(priority_miss_rates, datp_tables_dir)
        _Write_Priority_Policy_Table(priority_policy_summaries, datp_tables_dir)

if __name__ == "__main__":
    Run_Baseline()
