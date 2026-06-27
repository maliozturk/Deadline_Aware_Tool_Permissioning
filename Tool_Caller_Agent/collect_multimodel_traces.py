# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
# ------------------------------------------------------------------------------
#  File: Tool_Caller_Agent/collect_multimodel_traces.py
#  Purpose: Collect counterfactual paired FAST/SLOW latency traces for an
#           additional model backend, using the EXACT same prompt set,
#           fast/slow instructions, and router tool schema as the canonical
#           Llama-3.1 collection in Agent_V3.py. Only the model changes.
#
#  This reproduces, for Qwen 2.5 7B and Mistral 7B Instruct, the same process
#  that produced Tool_Caller_Agent/trace_results_counterfactual.csv (Llama 3.1).
#
#  Usage (run locally, with Ollama serving the model):
#     ollama pull qwen2.5:7b
#     python Tool_Caller_Agent/collect_multimodel_traces.py --model qwen2.5:7b
#
#     ollama pull mistral:7b-instruct
#     python Tool_Caller_Agent/collect_multimodel_traces.py --model mistral:7b-instruct
#
#  Tip: smoke-test first with a small subset:
#     python Tool_Caller_Agent/collect_multimodel_traces.py --model qwen2.5:7b --limit 5
#
#  Output: Tool_Caller_Agent/trace_results_counterfactual_<model>.csv
#          (same columns as the canonical trace, plus a model_name column).
#          Rows are flushed as they complete, so an interrupted run keeps its
#          partial data.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sure we can import the canonical generator regardless of CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the EXACT prompts, instructions, schema, and call helpers from the
# canonical Llama-3.1 generator. Importing Agent_V3 has no collection side
# effects (collection only runs under its __main__ guard).
from Agent_V3 import (  # noqa: E402
    simple_prompts_en,
    complex_prompts_en,
    tools_schema,
    build_fast_instruction,
    build_slow_instruction,
    timed_chat,
    extract_router_choice,
)

# Deterministic prompt order (no shuffle) for reproducible collection.
ALL_PROMPTS = (
    [(p, "simple") for p in simple_prompts_en]
    + [(p, "complex") for p in complex_prompts_en]
)

CSV_COLUMNS = [
    "model_name",
    "run_iter",
    "prompt_index",
    "prompt_type",
    "prompt",
    "router_choice",
    "router_latency_sec",
    "router_raw_text",
    "fast_total_latency_sec",
    "fast_generation_only_sec",
    "fast_response_length_char",
    "fast_response_text",
    "fast_error",
    "slow_total_latency_sec",
    "slow_generation_only_sec",
    "slow_response_length_char",
    "slow_response_text",
    "slow_error",
    "created_at",
]


def _sanitize(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def collect(model: str, out_path: Path, repeats: int, limit: int | None,
            sleep_sec: float, num_ctx: int | None,
            sample_n: int | None, seed: int) -> None:
    base = ALL_PROMPTS
    if sample_n is not None and sample_n < len(ALL_PROMPTS):
        # Fixed-seed sample so every model sees the SAME prompt subset.
        base = random.Random(seed).sample(ALL_PROMPTS, sample_n)
    prompts = base if limit is None else base[:limit]
    options = {"num_ctx": num_ctx} if num_ctx else None

    print(f"System starting... Model: {model}")
    print(f"Output CSV: {out_path}")
    print(f"Prompts per repeat: {len(prompts)} (of {len(ALL_PROMPTS)} total"
          + (f", sampled seed={seed}" if sample_n is not None else "")
          + f") | repeats: {repeats} | num_ctx: "
          + (str(num_ctx) if num_ctx else 'default'))
    print("-" * 60)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fast_means: list[float] = []
    slow_means: list[float] = []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for run_iter in range(repeats):
            for idx, (prompt, ptype) in enumerate(prompts):
                print(f"[iter={run_iter}] [{idx + 1}/{len(prompts)}] "
                      f"{ptype.upper()}: {prompt[:55]}...")

                # --- Router phase (tools exposed; choice recorded only) ---
                router_resp, router_elapsed, router_err = timed_chat(
                    model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=tools_schema,
                    options=options,
                )
                router_choice = (
                    extract_router_choice(router_resp)
                    if router_err is None else "error"
                )
                router_raw_text = ""
                if router_resp is not None:
                    router_raw_text = (
                        router_resp.get("message", {}).get("content") or ""
                    )[:2000]
                if sleep_sec:
                    time.sleep(sleep_sec)

                # --- FAST counterfactual ---
                fast_instruction = build_fast_instruction(prompt)
                fast_total_start = time.time()
                fast_resp, fast_gen_elapsed, fast_err = timed_chat(
                    model,
                    messages=[{"role": "user", "content": fast_instruction}],
                    tools=None,
                    options=options,
                )
                fast_total_elapsed = time.time() - fast_total_start
                fast_text = ""
                if fast_resp is not None:
                    fast_text = fast_resp.get("message", {}).get("content") or ""
                if sleep_sec:
                    time.sleep(sleep_sec)

                # --- SLOW counterfactual ---
                slow_instruction = build_slow_instruction(prompt)
                slow_total_start = time.time()
                slow_resp, slow_gen_elapsed, slow_err = timed_chat(
                    model,
                    messages=[{"role": "user", "content": slow_instruction}],
                    tools=None,
                    options=options,
                )
                slow_total_elapsed = time.time() - slow_total_start
                slow_text = ""
                if slow_resp is not None:
                    slow_text = slow_resp.get("message", {}).get("content") or ""

                writer.writerow({
                    "model_name": model,
                    "run_iter": run_iter,
                    "prompt_index": idx,
                    "prompt_type": ptype,
                    "prompt": prompt,
                    "router_choice": router_choice,
                    "router_latency_sec": round(router_elapsed, 4),
                    "router_raw_text": router_raw_text,
                    "fast_total_latency_sec": round(fast_total_elapsed, 4),
                    "fast_generation_only_sec": round(fast_gen_elapsed, 4),
                    "fast_response_length_char": len(fast_text),
                    "fast_response_text": fast_text,
                    "fast_error": fast_err,
                    "slow_total_latency_sec": round(slow_total_elapsed, 4),
                    "slow_generation_only_sec": round(slow_gen_elapsed, 4),
                    "slow_response_length_char": len(slow_text),
                    "slow_response_text": slow_text,
                    "slow_error": slow_err,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                })
                f.flush()  # keep partial data on interrupt

                if fast_err is None:
                    fast_means.append(fast_gen_elapsed)
                if slow_err is None:
                    slow_means.append(slow_gen_elapsed)

                print(f"   router={router_choice} ({router_elapsed:.2f}s) | "
                      f"FAST gen {fast_gen_elapsed:.2f}s | "
                      f"SLOW gen {slow_gen_elapsed:.2f}s")

    print("-" * 60)
    print("Counterfactual trace collection done.")
    if fast_means:
        print(f"FAST gen-only mean: {sum(fast_means)/len(fast_means):.3f}s "
              f"(n={len(fast_means)})")
    if slow_means:
        print(f"SLOW gen-only mean: {sum(slow_means)/len(slow_means):.3f}s "
              f"(n={len(slow_means)})")
    print(f"Wrote: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect counterfactual FAST/SLOW traces for one model "
                    "(same process as the canonical Llama-3.1 run).")
    parser.add_argument("--model", required=True,
                        help="Ollama model tag, e.g. qwen2.5:7b or "
                             "mistral:7b-instruct")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: "
                             "Tool_Caller_Agent/trace_results_counterfactual_"
                             "<model>.csv)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of passes over the prompt set (default 1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N prompts (smoke test).")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to sleep between calls (default 0).")
    parser.add_argument("--num-ctx", type=int, default=4096,
                        help="Ollama context window (num_ctx). Default 4096; "
                             "pass 0 to use the model's default.")
    parser.add_argument("--sample", type=int, default=50,
                        help="Randomly sample N prompts (fixed seed, same "
                             "subset for every model). Default 50; pass 0 for "
                             "the full prompt set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the prompt sample (default 42).")
    args = parser.parse_args()

    out_path = (
        Path(args.out) if args.out
        else SCRIPT_DIR / f"trace_results_counterfactual_{_sanitize(args.model)}.csv"
    )
    collect(args.model, out_path, args.repeats, args.limit, args.sleep,
            args.num_ctx if args.num_ctx > 0 else None,
            args.sample if args.sample > 0 else None, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
