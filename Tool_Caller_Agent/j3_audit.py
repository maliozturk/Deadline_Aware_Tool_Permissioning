"""J=3 Trace Dataset Audit — comprehensive validation before downstream use."""
import pandas as pd
import numpy as np
from pathlib import Path
import json

TRACE_PATH = Path(__file__).resolve().parent / "trace_results_counterfactual_j3.parquet"
REPORT_PATH = TRACE_PATH.parent / "j3_audit_report.md"

df = pd.read_parquet(TRACE_PATH)
lines = []  # collect markdown report lines

def pr(s=""):
    print(s)
    lines.append(s)

# ═══════════════════════════════════════════════════════════════════════════
# 1. ROW COUNTS
# ═══════════════════════════════════════════════════════════════════════════
pr("# J=3 Trace Dataset Audit Report")
pr()
pr("## 1. Row Counts")
pr()
pr(f"- **Total rows**: {len(df)}")
pr(f"- **Unique prompts**: {df.prompt_id.nunique()}")
pr(f"- **Backends**: {df.backend.unique().tolist()}")
pr(f"- **Tiers**: {df.tier.unique().tolist()}")

# Errors
err_rows = df[df["error_flag"] == True]
pr(f"- **Rows with error_flag=True**: {len(err_rows)}")
if len(err_rows) > 0:
    pr()
    pr("Error details (up to 20):")
    for i, (_, r) in enumerate(err_rows.head(20).iterrows()):
        pr(f"  {i+1}. {r.prompt_id} / {r.tier}: {str(r.error_message)[:120]}")

# Empty responses
empty = df[(df["response_text"].fillna("").str.strip() == "") | (df["response_length_chars"] == 0)]
pr(f"- **Rows with empty/zero-length response**: {len(empty)}")
if len(empty) > 0:
    pr()
    pr("Empty response breakdown by tier:")
    for t in ["C", "B", "A"]:
        n_empty = len(empty[empty["tier"] == t])
        pr(f"  Tier {t}: {n_empty}")

# Timeouts
timeouts = df[df["latency_sec"] >= 119.0]
pr(f"- **Rows hitting ~120s timeout**: {len(timeouts)}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. PER-TIER STATS
# ═══════════════════════════════════════════════════════════════════════════
pr()
pr("## 2. Per-Tier Statistics")
pr()
pr("| Tier | n | Lat Mean | Lat Med | Lat P10 | Lat P90 | Lat Min | Lat Max | Len Mean | Len Med | Len P10 | Len P90 | TC Mean | TC Max | TC>0 |")
pr("|------|---|----------|---------|---------|---------|---------|---------|----------|---------|---------|---------|---------|--------|------|")

tier_stats = {}
for t in ["C", "B", "A"]:
    s = df[df["tier"] == t]
    row = {
        "n": len(s),
        "lat_mean": s.latency_sec.mean(),
        "lat_med": s.latency_sec.median(),
        "lat_p10": s.latency_sec.quantile(0.1),
        "lat_p90": s.latency_sec.quantile(0.9),
        "lat_min": s.latency_sec.min(),
        "lat_max": s.latency_sec.max(),
        "len_mean": s.response_length_chars.mean(),
        "len_med": s.response_length_chars.median(),
        "len_p10": s.response_length_chars.quantile(0.1),
        "len_p90": s.response_length_chars.quantile(0.9),
        "tc_mean": s.tool_calls_count.mean(),
        "tc_max": int(s.tool_calls_count.max()),
        "tc_gt0": int((s.tool_calls_count > 0).sum()),
    }
    tier_stats[t] = row
    pr(f"| {t} | {row['n']} | {row['lat_mean']:.3f} | {row['lat_med']:.3f} | "
       f"{row['lat_p10']:.3f} | {row['lat_p90']:.3f} | {row['lat_min']:.3f} | {row['lat_max']:.3f} | "
       f"{row['len_mean']:.0f} | {row['len_med']:.0f} | {row['len_p10']:.0f} | {row['len_p90']:.0f} | "
       f"{row['tc_mean']:.2f} | {row['tc_max']} | {row['tc_gt0']} |")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PER-TIER x PER-COMPLEXITY GRID
# ═══════════════════════════════════════════════════════════════════════════
pr()
pr("## 3. Tier x Complexity Grid")
pr()
pr("| Complexity | Tier | n | Mean Lat (s) | Mean Resp Len |")
pr("|------------|------|---|-------------|---------------|")

for cplx in sorted(df["complexity"].unique()):
    for t in ["C", "B", "A"]:
        s = df[(df["tier"] == t) & (df["complexity"] == cplx)]
        if len(s) > 0:
            pr(f"| {cplx} | {t} | {len(s)} | {s.latency_sec.mean():.3f} | {s.response_length_chars.mean():.0f} |")

# ═══════════════════════════════════════════════════════════════════════════
# 4. ROW-LEVEL TIER ORDERING
# ═══════════════════════════════════════════════════════════════════════════
pr()
pr("## 4. Row-Level Tier Ordering")
pr()

# Build pivot: prompt_id -> {C, B, A} latency and length
strict_lat = 0
partial_lat = 0
inversion_lat = 0
strict_len = 0
partial_len = 0
inversion_len = 0
total_triples = 0

for pid in df.prompt_id.unique():
    sub = df[df["prompt_id"] == pid]
    c_rows = sub[sub["tier"] == "C"]
    b_rows = sub[sub["tier"] == "B"]
    a_rows = sub[sub["tier"] == "A"]
    if len(c_rows) == 0 or len(b_rows) == 0 or len(a_rows) == 0:
        continue
    total_triples += 1
    lc = c_rows.latency_sec.values[0]
    lb = b_rows.latency_sec.values[0]
    la = a_rows.latency_sec.values[0]
    # Latency ordering
    if la > lb > lc:
        strict_lat += 1
    elif la > lc and lb > lc:  # at least C is fastest
        partial_lat += 1
    else:
        inversion_lat += 1
    # Length ordering
    lenc = c_rows.response_length_chars.values[0]
    lenb = b_rows.response_length_chars.values[0]
    lena = a_rows.response_length_chars.values[0]
    if lena > lenb > lenc:
        strict_len += 1
    elif lena > lenc and lenb > lenc:
        partial_len += 1
    else:
        inversion_len += 1

inv_rate_lat = 100.0 * inversion_lat / total_triples if total_triples > 0 else 0
inv_rate_len = 100.0 * inversion_len / total_triples if total_triples > 0 else 0

pr(f"Total triples (prompt_ids with all 3 tiers): {total_triples}")
pr()
pr("### Latency Ordering")
pr(f"- Strict (A>B>C): {strict_lat} ({100*strict_lat/total_triples:.1f}%)")
pr(f"- Partial: {partial_lat} ({100*partial_lat/total_triples:.1f}%)")
pr(f"- Inversions: {inversion_lat} ({inv_rate_lat:.1f}%)")
pr()
pr("### Response Length Ordering")
pr(f"- Strict (A>B>C): {strict_len} ({100*strict_len/total_triples:.1f}%)")
pr(f"- Partial: {partial_len} ({100*partial_len/total_triples:.1f}%)")
pr(f"- Inversions: {inversion_len} ({inv_rate_len:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════
pr()
pr("## 5. Sanity Checks")
pr()

checks_passed = 0
checks_total = 6

# (a)
a_ok = tier_stats["A"]["lat_mean"] > tier_stats["B"]["lat_mean"] > tier_stats["C"]["lat_mean"]
tag = "PASS" if a_ok else "FAIL"
if a_ok: checks_passed += 1
pr(f"- **(a) Aggregate latency A>B>C**: **{tag}** — "
   f"A={tier_stats['A']['lat_mean']:.3f}, B={tier_stats['B']['lat_mean']:.3f}, C={tier_stats['C']['lat_mean']:.3f}")

# (b)
b_ok = tier_stats["A"]["len_mean"] > tier_stats["B"]["len_mean"] > tier_stats["C"]["len_mean"]
tag = "PASS" if b_ok else "FAIL"
if b_ok: checks_passed += 1
pr(f"- **(b) Aggregate length A>B>C**: **{tag}** — "
   f"A={tier_stats['A']['len_mean']:.0f}, B={tier_stats['B']['len_mean']:.0f}, C={tier_stats['C']['len_mean']:.0f}")

# (c)
c_ok = inv_rate_lat < 20.0
tag = "PASS" if c_ok else "FAIL"
if c_ok: checks_passed += 1
pr(f"- **(c) Latency inversion rate <20%**: **{tag}** — {inv_rate_lat:.1f}%")

# (d)
d_ok = tier_stats["C"]["tc_gt0"] == 0
tag = "PASS" if d_ok else "FAIL"
if d_ok: checks_passed += 1
pr(f"- **(d) Tier C tool_calls=0**: **{tag}** — {tier_stats['C']['tc_gt0']} rows with >0")

# (e)
err_rate = 100.0 * len(err_rows) / len(df)
e_ok = err_rate < 2.0
tag = "PASS" if e_ok else "FAIL"
if e_ok: checks_passed += 1
pr(f"- **(e) Error rate <2%**: **{tag}** — {err_rate:.2f}% ({len(err_rows)}/{len(df)})")

# (f)
timeout_rate = 100.0 * len(timeouts) / len(df)
f_ok = timeout_rate < 5.0
tag = "PASS" if f_ok else "FAIL"
if f_ok: checks_passed += 1
pr(f"- **(f) Timeout rate <5%**: **{tag}** — {timeout_rate:.2f}% ({len(timeouts)}/{len(df)})")

pr()
pr(f"**Result: {checks_passed}/{checks_total} checks passed.**")

# ═══════════════════════════════════════════════════════════════════════════
# 6. VERDICT
# ═══════════════════════════════════════════════════════════════════════════
pr()
pr("## 6. Verdict")
pr()

caveats = []
if len(empty) > 0:
    pct = 100.0 * len(empty) / len(df)
    caveats.append(f"{len(empty)} rows ({pct:.1f}%) have empty responses")
if inv_rate_lat >= 10.0:
    caveats.append(f"Latency inversion rate is {inv_rate_lat:.1f}% (elevated but below 20%)")
if not a_ok or not b_ok:
    caveats.append("Aggregate tier ordering failed")

if checks_passed == checks_total and len(caveats) == 0:
    pr("**PROCEED** — Dataset is clean and ready for downstream experiments.")
elif checks_passed >= checks_total - 1:
    pr("**PROCEED WITH CAVEATS**:")
    for c in caveats:
        pr(f"- {c}")
else:
    pr("**HOLD** — Multiple checks failed. Investigate before proceeding.")
    for c in caveats:
        pr(f"- {c}")

# Save markdown report
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nReport saved to {REPORT_PATH}")
