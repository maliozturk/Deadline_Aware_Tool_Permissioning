from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np


TRACE_CSV = REPO_ROOT / "Tool_Caller_Agent" / "trace_results_counterfactual.csv"
OUT_DIR = REPO_ROOT / "Results" / "Trace_ECDF"
OUT_PATH = OUT_DIR / "ecdf_fast_vs_slow.png"


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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _Load_Latencies(csv_path: Path) -> tuple[List[float], List[float]]:
    fast_vals: List[float] = []
    slow_vals: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("fast_error") or row.get("slow_error"):
                continue
            fast_raw = row.get("fast_generation_only_sec")
            slow_raw = row.get("slow_generation_only_sec")
            if not fast_raw or not slow_raw:
                continue
            try:
                fast_val = float(fast_raw)
                slow_val = float(slow_raw)
            except ValueError:
                continue
            if fast_val <= 0.0 or slow_val <= 0.0:
                continue
            fast_vals.append(fast_val)
            slow_vals.append(slow_val)
    if not fast_vals or not slow_vals:
        raise RuntimeError(f"No valid fast/slow samples found in {csv_path}")
    return fast_vals, slow_vals


def _Ecdf(vals: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(vals, dtype=float))
    ys = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, ys


def main() -> int:
    fast_vals, slow_vals = _Load_Latencies(TRACE_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _Apply_Plot_Style()

    fast_xs, fast_ys = _Ecdf(fast_vals)
    slow_xs, slow_ys = _Ecdf(slow_vals)

    plt.figure(figsize=(6.5, 4))
    plt.step(fast_xs, fast_ys, where="post", color="#1f77b4", label="Fast")
    plt.step(slow_xs, slow_ys, where="post", color="#d62728", label="Slow")
    plt.xlabel("generation-only service time (s)")
    plt.ylabel("ECDF")
    plt.title("Empirical ECDF of fast vs slow service times")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.close()

    print(f"[Trace_ECDF] wrote {OUT_PATH}", flush=True)
    print(f"[Trace_ECDF] fast_n={len(fast_vals)} slow_n={len(slow_vals)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
