# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Metrics/Reports.py
#  Purpose: Format aggregated metrics into a report string.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from typing import Any, Dict

from Metrics.Collector import SummaryStats


def _Fmt_Float(x_any: Any, nd_i32: int = 4) -> str:
    try:
        if x_any is None:
            return "NA"

        if isinstance(x_any, float):
            if x_any != x_any:       
                return "NA"
            return f"{x_any:.{nd_i32}f}"

        return str(x_any)
    except Exception:
        return str(x_any)


def Format_Summary(agg_dict_obj: Dict[str, object]) -> str:
    resp_summary_stats: SummaryStats = agg_dict_obj.get("response_time")                            
    wait_summary_stats: SummaryStats = agg_dict_obj.get("waiting_time")                             

    lines_list_str = []
    lines_list_str.append("=== Simulation Summary ===")
    lines_list_str.append(f"Total tasks:         {agg_dict_obj.get('n_tasks_total')}")
    lines_list_str.append(f"Completed:           {agg_dict_obj.get('n_completed')}")
    lines_list_str.append(f"Dropped in queue:    {agg_dict_obj.get('n_dropped_in_queue')}")
    lines_list_str.append(f"Unfinished:          {agg_dict_obj.get('n_unfinished')}")
    lines_list_str.append("")

    lines_list_str.append(f"Miss rate:           {_Fmt_Float(agg_dict_obj.get('miss_rate'))}")
    lines_list_str.append(f"Mean utility:        {_Fmt_Float(agg_dict_obj.get('mean_utility'))}")
    lines_list_str.append("")

    lines_list_str.append("Response time (completed only):")
    lines_list_str.append(
        f"  n={resp_summary_stats.count_i32} "
        f"mean={_Fmt_Float(resp_summary_stats.mean_f64)} "
        f"p50={_Fmt_Float(resp_summary_stats.p50_f64)} "
        f"p90={_Fmt_Float(resp_summary_stats.p90_f64)} "
        f"p99={_Fmt_Float(resp_summary_stats.p99_f64)}"
    )

    lines_list_str.append("Waiting time (completed only):")
    lines_list_str.append(
        f"  n={wait_summary_stats.count_i32} "
        f"mean={_Fmt_Float(wait_summary_stats.mean_f64)} "
        f"p50={_Fmt_Float(wait_summary_stats.p50_f64)} "
        f"p90={_Fmt_Float(wait_summary_stats.p90_f64)} "
        f"p99={_Fmt_Float(wait_summary_stats.p99_f64)}"
    )
    lines_list_str.append("")

    lines_list_str.append(f"Mode counts (completed): {agg_dict_obj.get('mode_counts_completed')}")
    lines_list_str.append(f"Queue length mean:      {_Fmt_Float(agg_dict_obj.get('queue_len_mean'))}")
    lines_list_str.append(f"Queue length p90:       {_Fmt_Float(agg_dict_obj.get('queue_len_p90'))}")

                               
    paug_keys = sorted([k for k in agg_dict_obj.keys() if str(k).startswith("paug_beta_")])
    if paug_keys:
        lines_list_str.append("")
        lines_list_str.append("PAUG (penalty-adjusted utility goodput):")
        for key in paug_keys:
            lines_list_str.append(f"  {key}: {_Fmt_Float(agg_dict_obj.get(key))}")

    return "\n".join(lines_list_str)
