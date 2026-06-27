# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
# ------------------------------------------------------------------------------
#  File: reproduce/make_figures.py
#  Purpose: Publication-quality figures for the paper, generated from the SAME
#           30-replication statistics as the results table
#           (Results/lambda_sweep/sweep_stats.csv produced by run_sweep.py).
#
#  Figures (PNG + PDF) written to Results/figures/ (and mirrored into the
#  manuscript figure dir Tex/PeerJ/Results/CADTR/ if it exists):
#     - hero_graph_premium_mr     : critical-task miss rate vs lambda
#     - shield_mechanism_slow_frac: Strategic usage on critical tasks vs lambda
#     - overall_utility           : mean realized utility vs lambda
#     - overall_miss_rate         : overall deadline miss rate vs lambda
#
#  Usage:
#     python reproduce/make_figures.py
# =============================================================================

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "Results" / "lambda_sweep" / "sweep_stats.csv"
OUT_DIR = REPO_ROOT / "Results" / "figures"
PAPER_FIG_DIR = REPO_ROOT / "Tex" / "PeerJ" / "Results" / "CADTR"

POLICY_STYLES = {
    "F-BB":              {"label": "F-BB",              "color": "#27AE60", "marker": "^", "ls": ":"},
    "FTC*":              {"label": "FTC*",              "color": "#8E44AD", "marker": "v", "ls": "--"},
    "Mode-Aware Oracle": {"label": "Mode-Aware Oracle", "color": "#E67E22", "marker": "D", "ls": "-."},
    "Myopic CDF Oracle": {"label": "CDF Oracle",        "color": "#C0392B", "marker": "s", "ls": "--"},
    "CADTR":             {"label": "CADTR",             "color": "#1B4F72", "marker": "o", "ls": "-"},
}
DRAW_ORDER = ["F-BB", "FTC*", "Mode-Aware Oracle", "Myopic CDF Oracle", "CADTR"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "lines.linewidth": 2.0, "lines.markersize": 7,
    "grid.linewidth": 0.8, "grid.linestyle": ":", "grid.color": "#cccccc",
    "axes.spines.top": False, "axes.spines.right": False,
})
FIGSIZE = (7, 4.5)
DPI = 300


def _plot(df, mean_col, y_label, fname, scale=1.0, ci_col=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for policy in DRAW_ORDER:
        s = POLICY_STYLES[policy]
        sub = df[df["policy"] == policy].sort_values("lambda")
        if sub.empty:
            continue
        y = sub[mean_col] * scale
        ax.plot(sub["lambda"], y, label=s["label"], color=s["color"],
                marker=s["marker"], linestyle=s["ls"],
                markeredgecolor="white", markeredgewidth=0.6,
                zorder=3 if policy == "CADTR" else 2)
        if ci_col and ci_col in sub.columns:
            err = sub[ci_col] * scale
            ax.fill_between(sub["lambda"], y - err, y + err,
                            color=s["color"], alpha=0.12, zorder=1)
    ax.set_xlabel(r"Arrival Rate $\lambda$ (requests/s)")
    ax.set_ylabel(y_label)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    if "CADTR" in labels:
        i = labels.index("CADTR")
        handles = [handles[i]] + [h for j, h in enumerate(handles) if j != i]
        labels = [labels[i]] + [l for j, l in enumerate(labels) if j != i]
    ax.legend(handles, labels, frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fancybox=False)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        written.append(out)
        print(f"  [ok] {out}")
    plt.close(fig)
    if PAPER_FIG_DIR.exists():
        for out in written:
            shutil.copy2(out, PAPER_FIG_DIR / out.name)


def main() -> int:
    if not CSV_PATH.exists():
        print(f"[error] {CSV_PATH} not found. Run: python reproduce/run_sweep.py")
        return 1
    df = pd.read_csv(CSV_PATH)
    print("[1/4] critical miss rate")
    _plot(df, "premium_miss_rate_mean", "Critical Task Miss Rate (%)",
          "hero_graph_premium_mr", scale=100, ci_col="premium_miss_rate_ci95")
    print("[2/4] strategic usage on critical tasks")
    _plot(df, "premium_slow_frac_mean", "Strategic Tool Usage on Critical Tasks (%)",
          "shield_mechanism_slow_frac", scale=100)
    print("[3/4] overall utility")
    _plot(df, "overall_utility_mean", "Mean Realized Utility",
          "overall_utility", scale=1.0)
    print("[4/4] overall miss rate")
    _plot(df, "overall_miss_rate_mean", "Overall Deadline Miss Rate (%)",
          "overall_miss_rate", scale=100, ci_col="overall_miss_rate_ci95")
    if PAPER_FIG_DIR.exists():
        print(f"[ok] mirrored figures into {PAPER_FIG_DIR}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
