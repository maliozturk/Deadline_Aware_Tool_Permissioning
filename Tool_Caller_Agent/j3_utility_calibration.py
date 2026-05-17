"""J=3 Utility Calibration — Steps 1-5.

Generates reference answers (gpt-oss-120b), computes quality scores,
diagnoses tier separation, fits Gaussians, and runs sanity checks.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import re
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
TRACE_PATH = SCRIPT_DIR / "trace_results_counterfactual_j3.parquet"
CYBER_IR_PATH = SCRIPT_DIR / "j3_prompt_set_cyber_ir.jsonl"
REF_PATH = SCRIPT_DIR / "j3_reference_answers.parquet"
QUALITY_PATH = SCRIPT_DIR / "j3_quality_scores.parquet"
CALIBRATION_PATH = SCRIPT_DIR / "j3_utility_calibration.json"

# Clarifai config
CLARIFAI_PAT = "ebfb4b55ca984121b3e62a7ac8bed5db"
CLARIFAI_BASE_URL = "https://api.clarifai.com/v2/ext/openai/v1"
REF_MODEL = (
    "https://clarifai.com/openai/chat-completion/models/"
    "gpt-oss-120b-high-throughput/versions/ce70fc95cef1411898db183e409e98d8"
)
REF_MODEL_NAME = "gpt-oss-120b-high-throughput"

# Retriever
_retriever = None
def _get_retriever():
    global _retriever
    if _retriever is None:
        sys.path.insert(0, str(SCRIPT_DIR))
        from j3_retriever import Retriever
        _retriever = Retriever()
    return _retriever

def _do_retrieve(query, k=3):
    return _get_retriever().retrieve(query, k=k)

# Load tier-A system prompt
TIER_A_PROMPT_PATH = SCRIPT_DIR / "j3_prompts" / "tool_A_deep_investigation.txt"
TIER_A_SYSTEM = TIER_A_PROMPT_PATH.read_text(encoding="utf-8").strip()

# Abstention patterns
ABSTENTION_PATTERNS = [
    r"^\s*$", r"^unknown$", r"^i don'?t know$", r"^cannot determine$",
    r"^insufficient information$", r"^no answer$", r"^n/?a$",
    r"^i cannot", r"^i don'?t have",
]
ABSTENTION_RE = [re.compile(p, re.IGNORECASE) for p in ABSTENTION_PATTERNS]

def is_abstention(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    for pat in ABSTENTION_RE:
        if pat.match(t):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate reference answers
# ═══════════════════════════════════════════════════════════════════════════
def step1_generate_references():
    print("=" * 70)
    print("  STEP 1: Generate reference answers (gpt-oss-120b)")
    print("=" * 70)

    # Load cyber-IR prompts
    cyber_prompts = {}
    with open(CYBER_IR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            cyber_prompts[row["prompt_id"]] = row["text"]

    print(f"Cyber-IR prompts: {len(cyber_prompts)}")

    # Check resume
    existing = set()
    if REF_PATH.exists():
        edf = pd.read_parquet(REF_PATH)
        existing = set(edf["prompt_id"])
        print(f"Resuming: {len(existing)} already generated")

    todo = {pid: text for pid, text in cyber_prompts.items() if pid not in existing}
    print(f"To generate: {len(todo)}")

    if not todo:
        print("All references already generated.")
        return

    client = OpenAI(base_url=CLARIFAI_BASE_URL, api_key=CLARIFAI_PAT)
    rows = []
    batch_buffer = []

    from tqdm import tqdm
    for i, (pid, prompt_text) in enumerate(tqdm(todo.items(), desc="References")):
        # Do retrieval (3 rounds like tier A)
        queries = [prompt_text]
        words = prompt_text.split()
        if len(words) > 5:
            queries.append(" ".join(words[:len(words)//2]))
            queries.append(" ".join(words[len(words)//2:]))

        all_docs = []
        seen = set()
        retrieved_ids = []
        for q in queries[:3]:
            results = _do_retrieve(q, k=3)
            for r in results:
                if r["doc_id"] not in seen:
                    seen.add(r["doc_id"])
                    all_docs.append(r)
                    retrieved_ids.append(r["doc_id"])

        # Build messages with injected docs
        messages = [
            {"role": "system", "content": TIER_A_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        if all_docs:
            docs_text = "\n\n---\n\n".join(
                f"[{d['doc_id']}] (score={d['score']})\n{d['content'][:500]}"
                for d in all_docs
            )
            messages.append({
                "role": "user",
                "content": f"RETRIEVED DOCUMENTS:\n\n{docs_text}\n\n"
                           "Use these documents to answer the original question. "
                           "Cite documents by their doc_id."
            })

        try:
            resp = client.chat.completions.create(
                model=REF_MODEL, messages=messages,
                max_tokens=1500, temperature=0.0, stream=False,
            )
            ref_text = resp.choices[0].message.content or ""
        except Exception as e:
            ref_text = ""
            print(f"  ERROR on {pid}: {e}")

        batch_buffer.append({
            "prompt_id": pid,
            "reference_text": ref_text,
            "reference_length_chars": len(ref_text),
            "reference_model": REF_MODEL_NAME,
            "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "retrieved_doc_ids": json.dumps(retrieved_ids),
            "is_abstention": is_abstention(ref_text),
        })

        # Checkpoint every 20
        if len(batch_buffer) >= 20 or (i + 1) == len(todo):
            new_df = pd.DataFrame(batch_buffer)
            if REF_PATH.exists():
                old = pd.read_parquet(REF_PATH)
                combined = pd.concat([old, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(REF_PATH, index=False)
            batch_buffer.clear()

    df = pd.read_parquet(REF_PATH)
    n_abs = df["is_abstention"].sum()
    print(f"\nReference answers: {len(df)}")
    print(f"Abstentions: {n_abs} ({100*n_abs/len(df):.1f}%)")
    print(f"Saved to {REF_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Compute quality scores
# ═══════════════════════════════════════════════════════════════════════════
def step2_compute_quality():
    print("\n" + "=" * 70)
    print("  STEP 2: Compute quality scores (abstention-aware)")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    trace_df = pd.read_parquet(TRACE_PATH)
    ref_df = pd.read_parquet(REF_PATH)

    # Only cyber-IR prompts have references
    ref_map = dict(zip(ref_df["prompt_id"], ref_df.itertuples(index=False)))
    cyber_ids = set(ref_df["prompt_id"])

    # Filter trace to cyber-IR prompts
    trace_cyber = trace_df[trace_df["prompt_id"].isin(cyber_ids)].copy()
    print(f"Trace rows with references: {len(trace_cyber)}")

    rows = []
    # Pre-encode all reference texts
    ref_texts = ref_df["reference_text"].fillna("").tolist()
    ref_pids = ref_df["prompt_id"].tolist()
    ref_abs = ref_df["is_abstention"].tolist()

    # Build ref lookup
    ref_lookup = {}
    for pid, txt, abst in zip(ref_pids, ref_texts, ref_abs):
        ref_lookup[pid] = {"text": txt, "is_abstention": abst}

    # Batch encode all response texts and ref texts
    all_resp = trace_cyber["response_text"].fillna("").tolist()
    all_ref_for_rows = [ref_lookup[pid]["text"] for pid in trace_cyber["prompt_id"]]

    print("Encoding responses...")
    resp_emb = model.encode(all_resp, show_progress_bar=True, batch_size=64)
    print("Encoding references...")
    ref_emb = model.encode(all_ref_for_rows, show_progress_bar=True, batch_size=64)

    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    # Row-wise cosine sim
    cos_sims = np.array([
        cosine_similarity([resp_emb[i]], [ref_emb[i]])[0][0]
        for i in range(len(resp_emb))
    ])

    # Build quality scores
    for idx, (_, row) in enumerate(trace_cyber.iterrows()):
        pid = row["prompt_id"]
        tier = row["tier"]
        resp_text = row["response_text"] or ""
        resp_empty = is_abstention(resp_text)
        ref_info = ref_lookup[pid]
        ref_abstention = ref_info["is_abstention"]

        if ref_abstention:
            # CASE 1: reference is abstention
            q_score = 0.5
            q_raw = None
            case = "abstention_reference"
        elif resp_empty:
            # CASE 2: tier abstained on answerable question
            q_score = 0.2
            q_raw = None
            case = "tier_abstained_on_answerable"
        else:
            # CASE 3: both non-empty
            raw = float(cos_sims[idx])
            q_score = float(np.clip(raw, 0.0, 1.0))
            q_raw = raw
            case = "embedding_similarity"

        rows.append({
            "prompt_id": pid,
            "tier": tier,
            "quality_score": q_score,
            "quality_score_raw": q_raw,
            "case_handled": case,
            "response_is_empty": resp_empty,
            "reference_is_abstention": ref_abstention,
        })

    qdf = pd.DataFrame(rows)
    qdf.to_parquet(QUALITY_PATH, index=False)
    print(f"\nQuality scores saved to {QUALITY_PATH}")
    print(f"Total rows: {len(qdf)}")

    # Print stats
    print("\nPer-tier quality_score stats:")
    print(f"{'Tier':<6} {'n':>5} {'Mean':>8} {'Med':>8} {'P10':>8} {'P90':>8}")
    print("-" * 44)
    for t in ["C", "B", "A"]:
        s = qdf[qdf["tier"] == t]
        print(f"{t:<6} {len(s):>5} {s.quality_score.mean():>8.4f} "
              f"{s.quality_score.median():>8.4f} "
              f"{s.quality_score.quantile(0.1):>8.4f} "
              f"{s.quality_score.quantile(0.9):>8.4f}")

    print("\nPer-tier case_handled counts:")
    for t in ["C", "B", "A"]:
        s = qdf[qdf["tier"] == t]
        counts = s["case_handled"].value_counts()
        parts = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  Tier {t}: {parts}")

    # Text histogram
    print("\nQuality score histogram (10 bins):")
    for t in ["C", "B", "A"]:
        s = qdf[qdf["tier"] == t]["quality_score"].values
        hist, edges = np.histogram(s, bins=10, range=(0, 1))
        bar = " ".join(f"{h:>4}" for h in hist)
        print(f"  Tier {t}: [{bar}]")

    return qdf


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Diagnose tier separation
# ═══════════════════════════════════════════════════════════════════════════
def step3_diagnose(qdf):
    print("\n" + "=" * 70)
    print("  STEP 3: Diagnose tier separation source")
    print("=" * 70)

    # Breakdown by case_handled per tier
    print("\nMean quality_score by (tier, case_handled):")
    for t in ["C", "B", "A"]:
        s = qdf[qdf["tier"] == t]
        for case in s["case_handled"].unique():
            sub = s[s["case_handled"] == case]
            print(f"  Tier {t} / {case}: n={len(sub)}, mu={sub.quality_score.mean():.4f}")

    # Clean subset: all three tiers non-empty AND reference non-abstention
    # i.e., case_handled == "embedding_similarity" for all three tiers of same prompt_id
    clean_pids = set()
    for pid in qdf["prompt_id"].unique():
        sub = qdf[qdf["prompt_id"] == pid]
        if len(sub) == 3 and all(sub["case_handled"] == "embedding_similarity"):
            clean_pids.add(pid)

    print(f"\nClean subset: {len(clean_pids)} prompts (all 3 tiers non-empty, ref non-abstention)")
    clean = qdf[qdf["prompt_id"].isin(clean_pids)]

    clean_means = {}
    for t in ["C", "B", "A"]:
        mu = clean[clean["tier"] == t]["quality_score"].mean()
        clean_means[t] = mu
        print(f"  Tier {t} clean mu: {mu:.4f}")

    ordering_ok = clean_means["A"] > clean_means["B"] > clean_means["C"]
    if ordering_ok:
        print("\n  CLEAN SUBSET ORDERING: PASS (mu_A > mu_B > mu_C)")
        print("  Tier separation is REAL, not driven by abstention rate.")
    else:
        print("\n  CLEAN SUBSET ORDERING: FAIL")
        print("  Tier separation may be driven by abstention rate, not quality.")
        print("  STOPPING — report this to the user.")
        return False, clean_means, len(clean_pids)

    return True, clean_means, len(clean_pids)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Fit Gaussians
# ═══════════════════════════════════════════════════════════════════════════
def step4_fit_gaussians(qdf, clean_means, n_clean):
    print("\n" + "=" * 70)
    print("  STEP 4: Fit per-tier truncated Gaussian parameters")
    print("=" * 70)

    tiers_config = []
    tier_info = [
        (0, "C", "quick_lookup"),
        (1, "B", "structured_analysis"),
        (2, "A", "deep_investigation"),
    ]

    for idx, letter, name in tier_info:
        s = qdf[qdf["tier"] == letter]["quality_score"]
        mu = float(s.mean())
        sigma = float(s.std())
        print(f"  Tier {idx} ({name}): mu={mu:.4f}, sigma={sigma:.4f}")
        tiers_config.append({
            "index": idx, "name": name,
            "mu": round(mu, 4), "sigma": round(sigma, 4),
            "u_min": 0.0, "u_max": 1.2,
        })

    calibration = {
        "calibration_source": f"embedding similarity with abstention handling vs {REF_MODEL_NAME}",
        "reference_model": REF_MODEL_NAME,
        "n_prompts_calibrated": int(qdf["prompt_id"].nunique()),
        "abstention_handling": {
            "rule_1_abstention_reference": "score = 0.5 for all tiers",
            "rule_2_tier_abstained": "score = 0.2 for empty response against answerable ref",
            "rule_3_normal": "cosine similarity, clipped to [0,1]",
        },
        "diagnostic_clean_subset": {
            "n": n_clean,
            "mu_clean_per_tier": [
                round(clean_means.get("C", 0), 4),
                round(clean_means.get("B", 0), 4),
                round(clean_means.get("A", 0), 4),
            ],
        },
        "tiers": tiers_config,
    }

    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nCalibration saved to {CALIBRATION_PATH}")

    return calibration


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Sanity checks
# ═══════════════════════════════════════════════════════════════════════════
def step5_sanity(qdf, calibration, clean_means):
    print("\n" + "=" * 70)
    print("  STEP 5: Sanity checks")
    print("=" * 70)

    tiers = calibration["tiers"]
    mu = {t["index"]: t["mu"] for t in tiers}
    sigma = {t["index"]: t["sigma"] for t in tiers}

    checks = 0
    total = 3

    # (a) Full dataset ordering
    a_ok = mu[2] > mu[1] > mu[0]
    tag = "PASS" if a_ok else "FAIL"
    if a_ok: checks += 1
    print(f"  (a) mu_A > mu_B > mu_C (full): {tag}  "
          f"({mu[2]:.4f} > {mu[1]:.4f} > {mu[0]:.4f})")

    # (b) Clean subset ordering
    b_ok = clean_means["A"] > clean_means["B"] > clean_means["C"]
    tag = "PASS" if b_ok else "FAIL"
    if b_ok: checks += 1
    print(f"  (b) mu_A > mu_B > mu_C (clean): {tag}  "
          f"({clean_means['A']:.4f} > {clean_means['B']:.4f} > {clean_means['C']:.4f})")

    # (c) Sigma range
    c_ok = all(0.05 <= sigma[i] <= 0.4 for i in range(3))
    tag = "PASS" if c_ok else "FAIL"
    if c_ok: checks += 1
    print(f"  (c) sigma in [0.05, 0.4]: {tag}  "
          f"({sigma[0]:.4f}, {sigma[1]:.4f}, {sigma[2]:.4f})")

    # (d) Length confound check (diagnostic only)
    trace_df = pd.read_parquet(TRACE_PATH)
    cyber_ids = set(qdf["prompt_id"])
    trace_cyber = trace_df[trace_df["prompt_id"].isin(cyber_ids)].copy()

    # Merge quality scores
    merged = trace_cyber.merge(qdf[["prompt_id", "tier", "quality_score"]],
                                on=["prompt_id", "tier"], how="inner")

    print("  (d) Length-quality correlation (diagnostic):")
    for t, letter in [(0, "C"), (1, "B"), (2, "A")]:
        s = merged[merged["tier"] == letter]
        if len(s) > 2 and s["response_length_chars"].std() > 0:
            r = s["quality_score"].corr(s["response_length_chars"])
            flag = " ⚠ LENGTH CONFOUND" if abs(r) > 0.6 else ""
            print(f"      Tier {letter}: r={r:.4f}{flag}")
        else:
            print(f"      Tier {letter}: insufficient variance")

    print(f"\n  Result: {checks}/{total} hard checks passed.")

    if not a_ok or not b_ok:
        print("\n  VERDICT: HOLD — tier ordering failed.")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1
    step1_generate_references()

    # Step 2
    qdf = step2_compute_quality()

    # Step 3
    ordering_ok, clean_means, n_clean = step3_diagnose(qdf)
    if not ordering_ok:
        print("\nSTOPPED at Step 3. Clean subset ordering failed.")
        sys.exit(1)

    # Step 4
    calibration = step4_fit_gaussians(qdf, clean_means, n_clean)

    # Step 5
    ok = step5_sanity(qdf, calibration, clean_means)

    print("\n" + "=" * 70)
    if ok:
        print("  BOTTOM LINE: PROCEED to Prompt 4")
    else:
        print("  BOTTOM LINE: HOLD — see failures above")
    print("=" * 70)
