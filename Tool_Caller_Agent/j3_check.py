"""Quick check on J=3 trace collection progress.
Usage: python Tool_Caller_Agent/j3_check.py
"""
import pandas as pd
from pathlib import Path

p = Path("Tool_Caller_Agent/trace_results_counterfactual_j3.parquet")
if not p.exists():
    print("No parquet file yet — waiting for first batch write.")
    raise SystemExit

df = pd.read_parquet(p)
print(f"Rows: {len(df)}  |  Errors: {int(df.error_flag.sum())}  |  Done: {len(df)}/1686")
print()
for t in ["C", "B", "A"]:
    s = df[df["tier"] == t]
    if len(s):
        errs = int(s.error_flag.sum())
        print(f"  Tier {t}: n={len(s)}, lat={s.latency_sec.mean():.2f}s, "
              f"len={s.response_length_chars.mean():.0f}, "
              f"tools={int(s.tool_calls_count.sum())}, errs={errs}")

errs = df[df["error_flag"] == True]
if len(errs):
    print(f"\nError details ({len(errs)} rows):")
    for _, r in errs.head(5).iterrows():
        msg = str(r.error_message)[:100]
        print(f"  {r.prompt_id} | {r.tier} | {msg}")
