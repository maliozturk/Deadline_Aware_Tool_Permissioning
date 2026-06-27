# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
# ------------------------------------------------------------------------------
#  File: reproduce/make_multimodel_ecdf.py
#  Purpose: Build the multi-model service-time ECDF figure and summary table
#           used in the paper (Figure: ecdf_fast_vs_slow_multimodel) from the
#           per-model counterfactual traces.
#
#  Inputs (gen-only FAST/SLOW seconds, error rows dropped):
#     - Tool_Caller_Agent/trace_results_counterfactual.csv               (Llama 3.1)
#     - Tool_Caller_Agent/trace_results_counterfactual_qwen2.5-7b.csv     (Qwen 2.5 7B)
#     - Tool_Caller_Agent/trace_results_counterfactual_mistral-7b-instruct.csv (Mistral 7B)
#
#  Outputs:
#     - Results/Trace_ECDF/ecdf_fast_vs_slow_multimodel.png (+ .pdf)
#     - Results/Trace_ECDF/ecdf_multimodel_summary.csv
#     (also copied into the paper figure dir if it exists)
#
#  Usage:
#     python reproduce/make_multimodel_ecdf.py
# =============================================================================

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = REPO_ROOT / "Tool_Caller_Agent"
OUT_DIR = REPO_ROOT / "Results" / "Trace_ECDF"
# Optional: also place the figure where the manuscript includes it from.
PAPER_FIG_DIR = REPO_ROOT / "Tex" / "PeerJ" / "Results" / "Trace_ECDF"

# (model_name, display label, trace csv path)
SOURCES = [
    ("llama3.1", "Llama 3.1",
     TRACE_DIR / "trace_results_counterfactual.csv"),
    ("qwen2.5:7b", "Qwen 2.5 7B",
     TRACE_DIR / "trace_results_counterfactual_qwen2.5-7b.csv"),
    ("mistral:7b-instruct", "Mistral 7B Instruct",
     TRACE_DIR / "trace_results_counterfactual_mistral-7b-instruct.csv"),
]

FAST_COL = "fast_generation_only_sec"
SLOW_COL = "slow_generation_only_sec"


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _load(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    fast_vals: list[float] = []
    slow_vals: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("fast_error") or row.get("slow_error"):
                continue
            try:
                fv = float(row.get(FAST_COL, ""))
                sv = float(row.get(SLOW_COL, ""))
            except (TypeError, ValueError):
                continue
            if fv <= 0.0 or sv <= 0.0:
                continue
            fast_vals.append(fv)
            slow_vals.append(sv)
    return np.sort(np.asarray(fast_vals)), np.sort(np.asarray(slow_vals))


def _ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, ys


def _summary_rows(model_name, label, fast, slow):
    rows = []
    for mode, arr in (("fast", fast), ("slow", slow)):
        rows.append({
            "model_name": model_name,
            "model_label": label,
            "mode": mode,
            "n": int(arr.size),
            "mean_sec": float(np.mean(arr)),
            "median_sec": float(np.median(arr)),
            "p90_sec": float(np.percentile(arr, 90)),
            "max_sec": float(np.max(arr)),
        })
    return rows


def main() -> int:
    _apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    available = [(m, lbl, p) for (m, lbl, p) in SOURCES if p.exists()]
    missing = [str(p.name) for (_, _, p) in SOURCES if not p.exists()]
    if missing:
        print("[warn] missing trace files (run collect_multimodel_traces.py):")
        for m in missing:
            print("       -", m)
    if not available:
        print("[error] no trace files found; nothing to plot.")
        return 1

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.3 * n_panels, 4.0),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    summary: list[dict] = []
    for ax, (model_name, label, path) in zip(axes, available):
        fast, slow = _load(path)
        if fast.size == 0 or slow.size == 0:
            ax.set_title(f"{label}\n(no valid samples)")
            continue
        fx, fy = _ecdf(fast)
        sx, sy = _ecdf(slow)
        ax.step(fx, fy, where="post", color="#1f77b4", label="Fast (Tactical)")
        ax.step(sx, sy, where="post", color="#d62728", label="Slow (Strategic)")
        ax.set_title(f"{label}  (n={fast.size})")
        ax.set_xlabel("generation-only service time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        summary.extend(_summary_rows(model_name, label, fast, slow))

    axes[0].set_ylabel("ECDF")
    fig.tight_layout()

    png = OUT_DIR / "ecdf_fast_vs_slow_multimodel.png"
    pdf = OUT_DIR / "ecdf_fast_vs_slow_multimodel.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[ok] wrote {png}")
    print(f"[ok] wrote {pdf}")

    summary_csv = OUT_DIR / "ecdf_multimodel_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model_name", "model_label", "mode", "n",
            "mean_sec", "median_sec", "p90_sec", "max_sec"])
        w.writeheader()
        w.writerows(summary)
    print(f"[ok] wrote {summary_csv}")

    # Mirror the figure into the manuscript build dir if present.
    if PAPER_FIG_DIR.exists():
        for src in (png, pdf):
            shutil.copy2(src, PAPER_FIG_DIR / src.name)
        print(f"[ok] copied figure into {PAPER_FIG_DIR}")

    print("\nSummary:")
    for r in summary:
        print(f"  {r['model_label']:>20s} {r['mode']:>4s}: "
              f"n={r['n']:4d} mean={r['mean_sec']:6.2f} "
              f"median={r['median_sec']:6.2f} p90={r['p90_sec']:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
