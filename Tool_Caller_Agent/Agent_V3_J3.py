# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC)
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Tool_Caller_Agent/Agent_V3_J3.py
#  Purpose: Three-tier trace collection harness for J=3 experiments.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-05-17
#  Environment: Python 3.11
# =============================================================================

"""Three-tier (J=3) trace collection harness.

Runs each prompt under three system-prompt-induced depth tiers
(C=quick, B=structured, A=deep) for each Ollama backend.

Usage:
    python Tool_Caller_Agent/Agent_V3_J3.py               # full sweep
    python Tool_Caller_Agent/Agent_V3_J3.py --smoke        # smoke test (30 rows)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ollama
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# ── Paths ────────────────────────────────────────────────────────────────────

PROMPTS_DIR = SCRIPT_DIR / "j3_prompts"
CORPUS_DIR = SCRIPT_DIR / "j3_corpus"
COMBINED_PROMPTS_PATH = SCRIPT_DIR / "j3_prompt_set_combined.jsonl"
CYBER_IR_PROMPTS_PATH = SCRIPT_DIR / "j3_prompt_set_cyber_ir.jsonl"

MAIN_TRACE_PATH = SCRIPT_DIR / "trace_results_counterfactual_j3.parquet"
SMOKE_TRACE_PATH = SCRIPT_DIR / "trace_results_counterfactual_j3_smoke.parquet"
PROVENANCE_PATH = SCRIPT_DIR / "j3_provenance.json"

SYSTEM_PROMPTS = {
    "C": PROMPTS_DIR / "tool_C_quick_lookup.txt",
    "B": PROMPTS_DIR / "tool_B_structured_analysis.txt",
    "A": PROMPTS_DIR / "tool_A_deep_investigation.txt",
}

TIER_NAMES = {"C": "tier_C_quick", "B": "tier_B_structured", "A": "tier_A_deep"}
TIER_MAX_TOKENS = {"C": 200, "B": 600, "A": 1500}

BACKENDS = ["llama3.1", "qwen2.5:7b-instruct", "mistral:7b-instruct"]
SMOKE_BACKEND = "qwen2.5:7b-instruct"

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
}

CALL_TIMEOUT_SEC = 120.0
BATCH_SIZE = 50
PROGRESS_INTERVAL = 10

# ── Retriever ────────────────────────────────────────────────────────────────

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        sys.path.insert(0, str(SCRIPT_DIR))
        from j3_retriever import Retriever
        _retriever = Retriever()
    return _retriever


def _do_retrieve(query: str, k: int = 3) -> List[Dict]:
    """Execute a retrieval call against the SOC corpus."""
    r = _get_retriever()
    return r.retrieve(query, k=k)


# ── Ollama tool schema ──────────────────────────────────────────────────────

RETRIEVE_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "Search the local SOC document store for relevant documents. Returns top-k results ranked by relevance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant SOC documents",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# ── Core helpers (match Agent_V3.py style) ───────────────────────────────────

def _load_system_prompt(tier: str) -> str:
    path = SYSTEM_PROMPTS[tier]
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_prompts(path: Path) -> List[Dict]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def _timed_chat_with_tools(
    model: str,
    messages: List[Dict],
    tools: Optional[List] = None,
    max_tokens: int = 1500,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Run an Ollama chat, handling tool calls iteratively.

    Returns dict with: response_text, latency_sec, tool_calls_count,
    retrieved_doc_ids, error_flag, error_message.
    """
    t0 = time.time()
    total_tool_calls = 0
    all_retrieved_doc_ids = []
    accumulated_text = ""

    opts = dict(SAMPLING_PARAMS)
    opts["num_predict"] = max_tokens

    try:
        for round_i in range(max_rounds + 1):
            elapsed_so_far = time.time() - t0
            if elapsed_so_far >= CALL_TIMEOUT_SEC:
                return {
                    "response_text": accumulated_text,
                    "latency_sec": CALL_TIMEOUT_SEC,
                    "tool_calls_count": total_tool_calls,
                    "retrieved_doc_ids": all_retrieved_doc_ids,
                    "error_flag": True,
                    "error_message": "timeout",
                }

            if tools is not None:
                resp = ollama.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    options=opts,
                )
            else:
                resp = ollama.chat(
                    model=model,
                    messages=messages,
                    options=opts,
                )

            msg = resp.get("message", {})
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", []) or []

            if not tool_calls:
                # No tool calls — final response
                accumulated_text += content
                elapsed = time.time() - t0
                return {
                    "response_text": accumulated_text,
                    "latency_sec": round(elapsed, 4),
                    "tool_calls_count": total_tool_calls,
                    "retrieved_doc_ids": all_retrieved_doc_ids,
                    "error_flag": False,
                    "error_message": "",
                }

            # Handle tool calls
            accumulated_text += content
            messages.append(msg)  # add assistant message with tool_calls

            for tc in tool_calls:
                total_tool_calls += 1
                func_name = tc.get("function", {}).get("name", "")
                func_args = tc.get("function", {}).get("arguments", {})

                if func_name == "retrieve":
                    query = func_args.get("query", "")
                    results = _do_retrieve(query, k=3)
                    for r in results:
                        all_retrieved_doc_ids.append(r["doc_id"])
                    # Build tool response
                    tool_response_content = json.dumps(
                        [{"doc_id": r["doc_id"], "score": r["score"],
                          "content": r["content"][:500]} for r in results],
                        ensure_ascii=False,
                    )
                else:
                    tool_response_content = json.dumps(
                        {"error": f"Unknown tool: {func_name}"}
                    )

                messages.append({
                    "role": "tool",
                    "content": tool_response_content,
                })

        # Exhausted rounds
        elapsed = time.time() - t0
        return {
            "response_text": accumulated_text,
            "latency_sec": round(elapsed, 4),
            "tool_calls_count": total_tool_calls,
            "retrieved_doc_ids": all_retrieved_doc_ids,
            "error_flag": False,
            "error_message": "",
        }

    except Exception as e:
        elapsed = time.time() - t0
        if elapsed >= CALL_TIMEOUT_SEC:
            return {
                "response_text": accumulated_text,
                "latency_sec": CALL_TIMEOUT_SEC,
                "tool_calls_count": total_tool_calls,
                "retrieved_doc_ids": all_retrieved_doc_ids,
                "error_flag": True,
                "error_message": "timeout",
            }
        return {
            "response_text": accumulated_text,
            "latency_sec": round(elapsed, 4),
            "tool_calls_count": total_tool_calls,
            "retrieved_doc_ids": all_retrieved_doc_ids,
            "error_flag": True,
            "error_message": str(e),
        }


# ── Provenance ───────────────────────────────────────────────────────────────

def _collect_provenance(backends: List[str]) -> Dict:
    """Collect provenance metadata for reproducibility."""
    prov: Dict[str, Any] = {}

    # Ollama version
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        )
        prov["ollama_version"] = result.stdout.strip()
    except Exception:
        prov["ollama_version"] = "unknown"

    # Model digests
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        model_lines = {}
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 2:
                model_lines[parts[0]] = parts[1]
        prov["model_digests"] = {
            b: model_lines.get(b, "not_found") for b in backends
        }
    except Exception:
        prov["model_digests"] = {}

    # Python version
    prov["python_version"] = sys.version

    # Package versions
    try:
        import ollama as _ollama_mod
        prov["ollama_package_version"] = getattr(_ollama_mod, "__version__", "unknown")
    except Exception:
        prov["ollama_package_version"] = "unknown"
    prov["pandas_version"] = pd.__version__
    prov["numpy_version"] = np.__version__

    # Hardware
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        prov["gpu"] = result.stdout.strip()
    except Exception:
        prov["gpu"] = "none_or_unavailable"

    prov["cpu"] = platform.processor()
    prov["platform"] = platform.platform()

    # Sampling params
    prov["sampling_params"] = SAMPLING_PARAMS
    prov["call_timeout_sec"] = CALL_TIMEOUT_SEC

    # Hostname (no usernames/paths)
    prov["hostname"] = socket.gethostname()
    prov["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return prov


# ── Main trace collection ───────────────────────────────────────────────────

def _load_existing(trace_path: Path) -> set:
    """Load already-collected (prompt_id, backend, tier) triples."""
    if not trace_path.exists():
        return set()
    try:
        df = pd.read_parquet(trace_path)
        return set(zip(df["prompt_id"], df["backend"], df["tier"]))
    except Exception:
        return set()


def _append_parquet(trace_path: Path, rows: List[Dict]) -> None:
    """Append rows to parquet file."""
    new_df = pd.DataFrame(rows)
    if trace_path.exists():
        existing_df = pd.read_parquet(trace_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(trace_path, index=False)


def run_trace_collection(
    prompts: List[Dict],
    backends: List[str],
    tiers: List[str],
    trace_path: Path,
    smoke: bool = False,
) -> None:
    """Run the trace collection loop."""
    # Load system prompts
    sys_prompts = {t: _load_system_prompt(t) for t in tiers}

    # Check for resume
    done_triples = _load_existing(trace_path)
    if done_triples:
        print(f"Resuming: {len(done_triples)} triples already collected")

    # Build work list
    work = []
    for prompt_row in prompts:
        for backend in backends:
            for tier in tiers:
                triple = (prompt_row["prompt_id"], backend, tier)
                if triple not in done_triples:
                    work.append((prompt_row, backend, tier))

    total = len(work)
    print(f"Total calls to make: {total}")
    if total == 0:
        print("Nothing to do.")
        return

    # Collect provenance
    prov = _collect_provenance(backends)
    prov["trace_path"] = str(trace_path)
    prov["smoke"] = smoke
    prov["total_calls"] = total
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f"Provenance written to {PROVENANCE_PATH}")

    # Run
    batch_buffer: List[Dict] = []
    t_start = time.time()

    for i, (prompt_row, backend, tier) in enumerate(tqdm(work, desc="Trace collection")):
        prompt_id = prompt_row["prompt_id"]
        prompt_text = prompt_row["text"]
        source = prompt_row.get("source", "unknown")
        complexity = prompt_row.get("complexity", "unknown")

        # Build messages
        messages = [
            {"role": "system", "content": sys_prompts[tier]},
            {"role": "user", "content": prompt_text},
        ]

        # Tier C: no tools. Tiers B and A: retrieval tool available.
        tools = None if tier == "C" else RETRIEVE_TOOL_SCHEMA
        max_tokens = TIER_MAX_TOKENS[tier]
        # Tier A gets 3 rounds, B gets 1, C gets 0
        max_rounds = {"C": 0, "B": 1, "A": 3}[tier]

        result = _timed_chat_with_tools(
            model=backend,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            max_rounds=max_rounds,
        )

        row = {
            "prompt_id": prompt_id,
            "source": source,
            "complexity": complexity,
            "backend": backend,
            "tier": tier,
            "tier_name": TIER_NAMES[tier],
            "latency_sec": result["latency_sec"],
            "response_text": result["response_text"],
            "response_length_chars": len(result["response_text"]),
            "tool_calls_count": result["tool_calls_count"],
            "retrieved_doc_ids": json.dumps(result["retrieved_doc_ids"]),
            "error_flag": result["error_flag"],
            "error_message": result["error_message"],
        }
        batch_buffer.append(row)

        # Progress line
        if (i + 1) % PROGRESS_INTERVAL == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{total}] elapsed={elapsed:.0f}s "
                f"rate={rate:.2f}/s ETA={eta:.0f}s"
            )

        # Batch write
        if len(batch_buffer) >= BATCH_SIZE or (i + 1) == total:
            _append_parquet(trace_path, batch_buffer)
            batch_buffer.clear()

    elapsed = time.time() - t_start
    print(f"\nDone. {total} calls in {elapsed:.1f}s")


# ── Smoke test ───────────────────────────────────────────────────────────────

def run_smoke_test() -> None:
    """Run smoke test: 10 prompts × 1 backend × 3 tiers = 30 rows."""
    print("=" * 70)
    print("  SMOKE TEST: 10 prompts × 1 backend × 3 tiers = 30 rows")
    print("=" * 70)

    # Load cyber-IR prompts only
    all_prompts = _load_prompts(CYBER_IR_PROMPTS_PATH)

    # Select 10 prompts: 4 simple, 4 medium, 2 complex
    simple = [p for p in all_prompts if p["complexity"] == "simple"][:4]
    medium = [p for p in all_prompts if p["complexity"] == "medium"][:4]
    cplx = [p for p in all_prompts if p["complexity"] == "complex"][:2]
    smoke_prompts = simple + medium + cplx

    print(f"Smoke prompts: {len(smoke_prompts)} "
          f"(simple={len(simple)}, medium={len(medium)}, complex={len(cplx)})")
    print(f"Backend: {SMOKE_BACKEND}")
    print(f"Tiers: C, B, A")
    print()

    # Delete existing smoke file for clean run
    if SMOKE_TRACE_PATH.exists():
        os.remove(SMOKE_TRACE_PATH)

    run_trace_collection(
        prompts=smoke_prompts,
        backends=[SMOKE_BACKEND],
        tiers=["C", "B", "A"],
        trace_path=SMOKE_TRACE_PATH,
        smoke=True,
    )

    # Print summary and run sanity checks
    _smoke_summary_and_checks()


def _smoke_summary_and_checks() -> None:
    """Print smoke summary table and run sanity checks."""
    if not SMOKE_TRACE_PATH.exists():
        print("ERROR: Smoke trace file not found.")
        return

    df = pd.read_parquet(SMOKE_TRACE_PATH)
    print("\n" + "=" * 70)
    print("  SMOKE SUMMARY")
    print("=" * 70)

    # Per-tier stats
    header = f"{'Tier':<15} {'Count':>6} {'Mean_Lat':>10} {'Med_Lat':>10} {'P90_Lat':>10} {'Mean_RespLen':>13}"
    print(header)
    print("-" * len(header))

    tier_stats = {}
    for tier in ["C", "B", "A"]:
        subset = df[df["tier"] == tier]
        count = len(subset)
        mean_lat = subset["latency_sec"].mean()
        med_lat = subset["latency_sec"].median()
        p90_lat = subset["latency_sec"].quantile(0.9)
        mean_len = subset["response_length_chars"].mean()
        tier_stats[tier] = {
            "count": count,
            "mean_lat": mean_lat,
            "med_lat": med_lat,
            "p90_lat": p90_lat,
            "mean_len": mean_len,
        }
        print(f"{tier:<15} {count:>6} {mean_lat:>10.3f} {med_lat:>10.3f} "
              f"{p90_lat:>10.3f} {mean_len:>13.1f}")

    # Sanity checks
    print("\n" + "=" * 70)
    print("  SANITY CHECKS")
    print("=" * 70)

    checks_passed = 0
    checks_total = 4

    # (a) Latency ordering: A > B > C
    lat_a = tier_stats["A"]["mean_lat"]
    lat_b = tier_stats["B"]["mean_lat"]
    lat_c = tier_stats["C"]["mean_lat"]
    if lat_a > lat_b > lat_c:
        print(f"  (a) Latency ordering:      PASS  (A={lat_a:.3f} > B={lat_b:.3f} > C={lat_c:.3f})")
        checks_passed += 1
    else:
        print(f"  (a) Latency ordering:      FAIL  (A={lat_a:.3f}, B={lat_b:.3f}, C={lat_c:.3f})")

    # (b) Response length ordering: A > B > C
    len_a = tier_stats["A"]["mean_len"]
    len_b = tier_stats["B"]["mean_len"]
    len_c = tier_stats["C"]["mean_len"]
    if len_a > len_b > len_c:
        print(f"  (b) Response length order: PASS  (A={len_a:.1f} > B={len_b:.1f} > C={len_c:.1f})")
        checks_passed += 1
    else:
        print(f"  (b) Response length order: FAIL  (A={len_a:.1f}, B={len_b:.1f}, C={len_c:.1f})")

    # (c) Tier C tool_calls_count == 0
    tier_c_tools = df[df["tier"] == "C"]["tool_calls_count"].sum()
    if tier_c_tools == 0:
        print(f"  (c) Tier C no tool calls:  PASS  (total={tier_c_tools})")
        checks_passed += 1
    else:
        print(f"  (c) Tier C no tool calls:  FAIL  (total={tier_c_tools})")

    # (d) No errors
    error_count = df["error_flag"].sum()
    if error_count == 0:
        print(f"  (d) No errors:             PASS  (errors={error_count})")
        checks_passed += 1
    else:
        print(f"  (d) No errors:             FAIL  (errors={error_count})")
        error_rows = df[df["error_flag"] == True]
        for _, row in error_rows.iterrows():
            print(f"      {row['prompt_id']} / {row['backend']} / {row['tier']}: "
                  f"{row['error_message']}")

    print()
    if checks_passed == checks_total:
        print(f"  ALL {checks_total} SANITY CHECKS PASSED")
    else:
        print(f"  {checks_passed}/{checks_total} SANITY CHECKS PASSED — DO NOT PROCEED")


# ── Full sweep orchestration (Step 6: designed, not executed) ────────────────

def run_full_sweep() -> None:
    """Run the full trace collection sweep."""
    print("=" * 70)
    print("  FULL TRACE COLLECTION SWEEP")
    print("=" * 70)

    prompts = _load_prompts(COMBINED_PROMPTS_PATH)

    # Cost estimation
    n_prompts = len(prompts)
    n_backends = len(BACKENDS)
    n_tiers = len(TIER_NAMES)
    total_calls = n_prompts * n_backends * n_tiers

    # Estimate from smoke if available
    if SMOKE_TRACE_PATH.exists():
        smoke_df = pd.read_parquet(SMOKE_TRACE_PATH)
        mean_lat_per_tier = {}
        for tier in ["C", "B", "A"]:
            subset = smoke_df[smoke_df["tier"] == tier]
            mean_lat_per_tier[tier] = subset["latency_sec"].mean()
        avg_lat = sum(mean_lat_per_tier.values()) / len(mean_lat_per_tier)
        est_hours = (total_calls * avg_lat) / 3600.0
    else:
        avg_lat = 30.0  # rough default
        est_hours = (total_calls * avg_lat) / 3600.0

    print(f"  Prompts:  {n_prompts}")
    print(f"  Backends: {n_backends} ({', '.join(BACKENDS)})")
    print(f"  Tiers:    {n_tiers}")
    print(f"  Total calls: {total_calls}")
    print(f"  Estimated avg latency/call: {avg_lat:.1f}s")
    print(f"  Estimated wall-clock: {est_hours:.1f} hours")
    print()

    run_trace_collection(
        prompts=prompts,
        backends=BACKENDS,
        tiers=["C", "B", "A"],
        trace_path=MAIN_TRACE_PATH,
        smoke=False,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="J=3 trace collection")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only")
    parser.add_argument("--full", action="store_true", help="Run full sweep")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_test()
    elif args.full:
        run_full_sweep()
    else:
        print("Usage: python Agent_V3_J3.py --smoke | --full")
        print("  --smoke  Run smoke test (30 rows)")
        print("  --full   Run full sweep (all prompts × backends × tiers)")


if __name__ == "__main__":
    main()
