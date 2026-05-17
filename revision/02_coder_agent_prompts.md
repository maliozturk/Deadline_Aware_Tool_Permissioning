# FTC Paper — Coder Agent Prompt Sequence

**How to use this file:** Issue each prompt to your coder agent one at a time. Wait for the produced artifact (code + run output + plot) before sending the next prompt. Each prompt assumes the previous one's artifacts exist in the repo.

**Repository assumptions** (based on what the paper describes):
- A discrete-event simulator with a non-preemptive single-server queue, Poisson arrivals, FCFS, abandonment-at-deadline.
- A `policies/` module containing `always_fast`, `always_slow`, `static_mix`, `q_bb`, `f_bb`, `heuristic`, `drift_penalty`, and `ftc` policies.
- A trace-replay layer that samples service times from empirical paired traces collected from three local Ollama-backed LLMs (Llama 3.1, Qwen 2.5 7B Instruct, Mistral 7B Instruct).
- A trace-collection script that runs a tool-routing agent with two execution profiles (`fast_lookup`, `deep_reasoner`).
- An experiment runner that produces the current Tables 2–4 and Figures 1–4.

Adjust paths in the prompts below if your repo layout differs.

---

## PROMPT 1 — Generalize the simulator and policy interface to J modes

> The simulator and policy interface currently hardcode two modes ("fast" and "slow"). Refactor to support an arbitrary ordered tool menu of size J ≥ 2, indexed by integer 0..J-1, with the convention that index 0 is the fastest/lightest tool and index J-1 is the slowest/richest tool. Do not delete the two-mode path — instead, treat it as the J=2 special case.
>
> Specifically:
> 1. Add a `ToolTier` config dataclass with fields: `name: str`, `index: int`, `mean_service_time: float`, `service_trace: List[float]`, `mean_utility: float`, `utility_std: float`, `utility_min: float`, `utility_max: float`. The tier list is passed to the simulator at construction.
> 2. Modify the simulator's request-arrival handler to accept a policy callback that returns an integer mode index in `range(J)` instead of a `{'fast','slow'}` string. Adapt the existing two-mode policies to this new interface (they should return 0 or J-1).
> 3. Add a per-tier accounting layer: per-tier service times, per-tier utility, per-tier selection counts. Replace the single `slow_frac` metric with a `tier_fractions: List[float]` of length J.
> 4. Generalize the deadline-miss accounting: it stays a binary indicator per request, independent of J.
> 5. Generalize the existing FTC policy as follows. The J-mode rule selects the highest tier index j such that `R_k + Q_k * s_avg + s_j <= (1+epsilon) * delta_k`, where `s_avg = sum(rho_j * s_j for j in range(J))` and `rho` is a probability vector of length J. If no tier is feasible, select tier 0. For the J=2 case, `rho = (1-rho_scalar, rho_scalar)` recovers the existing rule.
> 6. Add unit tests in `tests/test_j_mode_simulator.py` verifying that with J=2 and the previous queue-mix scalar, the new simulator reproduces the existing Table 2 numbers within numerical tolerance over 5 fixed seeds. **This regression test is mandatory** — do not proceed to Prompt 2 until it passes.
>
> Deliverables: updated source files, the test file, and a printed comparison of old-vs-new Table 2 numbers showing equality within tolerance.

---

## PROMPT 2 — Collect J=3 trace data with three tool depths

> Extend the existing trace-collection harness to support three tool depths instead of two. Use the following naming and configuration (we are reframing the running example to LLM-orchestrated cyber incident response, but the tier definitions below are generic enough to also serve as general reasoning tiers):
>
> - **Tool C (tier 0, "quick_lookup")**: system prompt instructs the agent to produce a single concise factual answer with no external tool calls, max 200 tokens. Fastest, lowest expected utility.
> - **Tool B (tier 1, "structured_analysis")**: system prompt instructs the agent to perform a single retrieval step (a mock retrieval against a small local document store of ~50 docs) and then produce a structured answer with explicit reasoning, max 600 tokens. Medium latency, medium utility.
> - **Tool A (tier 2, "deep_investigation")**: system prompt instructs the agent to perform multi-hop retrieval (up to 3 rounds), produce a chain-of-thought, cite specific retrieved documents, and synthesize a final analysis, max 1500 tokens. Slowest, highest expected utility.
>
> Provide the three system prompts as separate text files in `prompts/`. For the small local document store, generate 50 mock cyber-incident-response documents (a mix of runbooks, IOC entries, and threat-intel notes) — generate them synthetically via the LLM itself if needed, but persist them on disk so trace collection is reproducible.
>
> For trace collection:
> 1. Use the same prompt set already in your trace-collection harness (the 1332-row corpus mentioned in the paper). Add 100–200 additional cyber-IR-themed prompts to broaden the corpus; representative prompts include "Investigate a sudden spike in failed SSH logins on host X", "Determine whether IOC `abc123` warrants a containment action", "Summarize the root cause of incident ticket #4521 using available runbook excerpts".
> 2. For each prompt and each of the three Ollama backends (Llama 3.1, Qwen 2.5 7B Instruct, Mistral 7B Instruct), execute the prompt under all three tool tiers (Tool C, B, A). This gives triplet-paired latency observations per (prompt, backend), generalizing the existing paired counterfactual design.
> 3. Persist the dataset as `traces/j3_trace_data.parquet` with one row per (prompt, backend, tier) triple, capturing prompt_id, backend, tier, latency_sec, response_text, response_length_chars, error_flag.
>
> Deliverables: the three system prompts, the 50-doc mock store, the trace-collection script, the parquet dataset, and a printed summary showing mean/median/p90 latency per (backend, tier).

---

## PROMPT 3 — Calibrate utility distributions for the three tiers

> The paper currently samples per-mode utility from mode-specific truncated Gaussians with hand-set parameters. We need to ground the J=3 utility distributions in something less arbitrary. Implement this calibration:
>
> 1. For each prompt in the cyber-IR-themed subset of the trace dataset, generate a **reference answer** using an external strong LLM (use the largest available local model, or document that GPT-4-class judging would be the equivalent step in production). Persist as `traces/reference_answers.parquet`.
> 2. For each triple (prompt, backend, tier) in the trace dataset, compute a similarity-based quality score against the reference answer: use cosine similarity over sentence-transformer embeddings (e.g., `all-MiniLM-L6-v2` from sentence-transformers, which is small and CPU-runnable). Quality score is in [0, 1].
> 3. For each tier, fit the empirical mean and std of quality scores across all (prompt, backend) pairs. Use these fitted values as the tier-utility distribution parameters (truncated Gaussian, clipped to [0, 1.2]).
> 4. The result should be: Tool A mean utility > Tool B > Tool C, as the paper's narrative requires. If empirical results do not reproduce this ordering for some backend, document it honestly and discuss in Section 8.
> 5. Save the fitted parameters as `config/utility_params_j3.json` for downstream simulation runs.
>
> Deliverables: reference-answer dataset, fitted utility parameters JSON, and a printed table of mean/std/min/max quality scores per tier and per backend.

---

## PROMPT 4 — Implement the three new baseline policies

> Add three new baseline policies to `policies/`, all working with the J-mode interface from Prompt 1:
>
> **1. FTC-MA (Mode-Aware FTC).** Identical to FTC except the backlog estimate uses the *actual* assigned modes of queued jobs: `W_hat_k = R_k + sum(s_j * n_j_in_queue for j in range(J))`. This requires the simulator to track per-tier counts in the queue, not just total `Q_k`. Add that tracking to the simulator.
>
> **2. OCP (Oracle-CDF Policy).** Uses the empirical service-time CDF `F_hat_m` computed offline from the trace dataset (per backend if known, otherwise pooled). At each arrival, compute `slack = delta_k - W_hat_k` (using the FTC backlog estimate), then for each tier j compute `expected_utility_j = u_hat_j * F_hat_j(slack)`. Select `argmax_j expected_utility_j`. If all tiers give zero (i.e., slack ≤ 0 or all empirical CDFs return 0), select tier 0.
>
> **3. FTC-SE (Service-Epoch FTC).** Same feasibility rule as FTC, but the *decision* is deferred until the request reaches the server head. At decision time, `R_k = 0` (server is idle, since the request is about to be served), and `Q_k` only counts arrivals that came in while waiting. Implement by attaching a "decide later" sentinel at arrival, and have the simulator invoke the policy a second time at service start. **Be careful**: the tool-startup cost is non-zero in real deployments, but we're modeling the upper bound here; document in the code comment that `FTC-SE` is an upper-bound benchmark, not a deployable policy.
>
> For each new policy, add to `tests/test_baselines_j3.py` a test that the policy returns a valid tier index for a few hand-crafted states.
>
> Deliverables: three new policy files, updated simulator to support mode-aware queue tracking and service-epoch decisions, tests passing.

---

## PROMPT 5 — Reproduce the main results table for J=3 across TTL regimes

> Re-run the main results experiment from the paper's Table 2, now with J=3 and the new baselines, on the J=3 trace dataset.
>
> Configuration:
> - Arrival rate: λ = 0.05 req/s (same as paper Table 2).
> - TTL regimes I–IV with the same (μ_δ, σ_δ) as the paper.
> - Policies compared: Always-A (the new "slow"), Always-C (the new "fast"), TTL-aware heuristic (generalize the rule to pick the richest tier whose mean fits αδ_k), Q-BB (generalize to threshold-based tier selection: top tier if Q_k=0, middle if Q_k < Q_B, lowest otherwise), Drift-style (generalize the V-weighted maximization over J tiers), FTC (default ρ=0.55 scalar, generalized to a (1−ρ)/2, (1−ρ)/2, ρ vector — but also try a uniform ρ=(1/3,1/3,1/3) and report whichever is better on the held-out config), FTC-MA, OCP, FTC-SE.
> - Per-tier utility from `config/utility_params_j3.json`.
> - Service times from `traces/j3_trace_data.parquet` (pooled across the three backends).
> - 5 seeds, seed_base = 1453, same as the paper.
>
> Metrics per policy: mean realized utility, deadline-miss rate, p95 response time, **tier-A fraction, tier-B fraction, tier-C fraction**.
>
> Output: a CSV `results/table2_j3.csv` and a pretty-printed markdown table identical in structure to the paper's Table 2 but with the new policies and tier-fraction columns.
>
> Deliverables: experiment script, CSV, markdown table. Print to console a brief sanity check: FTC's tier fractions should shift toward Tool A in Regime IV (loose deadlines) and toward Tool C in Regime I (tight deadlines).

---

## PROMPT 6 — Reproduce the load-sweep experiment for J=3

> Re-run the load-sweep experiment from the paper's Figures 1–2, now with J=3.
>
> Configuration:
> - TTL Regime II fixed.
> - λ ∈ {0.03, 0.05, ..., 0.19} (nine values, same as paper).
> - Policies: same set as Prompt 5 but include the per-λ-tuned ε variant of FTC (sweep ε ∈ {-0.20, -0.15, ..., 0.10} at each λ, pick the ε minimizing DMR).
> - 5 seeds each.
>
> Output two figures:
> - `results/fig1_utility_vs_lambda_j3.pdf`: mean realized utility vs λ, one line per policy.
> - `results/fig2_dmr_vs_lambda_j3.pdf`: deadline-miss rate vs λ, one line per policy.
>
> Use the same color/marker scheme as the existing Figures 1–2 for backward visual familiarity; assign new colors to FTC-MA, OCP, FTC-SE.
>
> Deliverables: the two PDFs plus the source data CSV at `results/load_sweep_j3.csv`.

---

## PROMPT 7 — Run the ε sweep and produce the utility–DMR frontier figure

> Re-run the ε sweep from the paper's Figure 3, now with J=3 and on the new trace dataset.
>
> Configuration:
> - TTL Regime II, λ = 0.05.
> - ε ∈ {-0.10, -0.05, 0, 0.05, 0.10}.
> - 5 seeds.
> - Plot FTC's (DMR, utility) points connected as a line, overlay the fixed baselines (Always-A, Always-C, Static-mix p=1/3, p=1/2, p=2/3, TTL-aware heuristic, Q-BB, Drift-style, FTC-MA, OCP) as scatter points.
>
> Output: `results/fig3_epsilon_tradeoff_j3.pdf` and the source CSV.

---

## PROMPT 8 — Implement and report the PI-controller experiment that was promised but never delivered

> The paper describes a PI-controller adaptive ε scheme in §7.5 with full equations but reports no results. Implement and run it.
>
> Configuration:
> - Window length W=25, gains (K_p, K_i) = (0.02, 0.002), bounds (ε_min, ε_max) = (-0.10, 0.10).
> - Target miss-rate setpoints θ ∈ {0.05, 0.10, 0.15, 0.20}.
> - TTL Regime II, λ following a step pattern: λ=0.05 for the first 5000 simulated seconds, then λ=0.13 for the next 5000 seconds, then back to λ=0.05 for the last 5000 seconds. This creates a controlled load shift the PI controller has to absorb.
> - 5 seeds.
>
> Output for each θ:
> - A two-panel figure: top panel is the windowed DMR estimate over time with the target θ as a horizontal dashed line; bottom panel is ε(t) over time, with ε_min and ε_max as horizontal dashed lines.
> - File `results/fig5_pi_controller_theta_{theta}.pdf`.
>
> Also output a summary table `results/table_pi_controller.csv` with columns: θ, mean DMR (steady state, last 1000 s of each load phase), mean ε (same window), tracking error (|mean DMR − θ|).
>
> Deliverables: the four PDFs plus the summary CSV. Print sanity check: tracking error should be < 0.03 for at least three of the four θ values.

---

## PROMPT 9 — End-to-end micro-experiment with a live LLM in the loop (no replay)

> R3 complained the experiments are "toy" because everything is offline replay. Run a small *live* end-to-end experiment to neutralize this.
>
> Setup:
> 1. Set up a real FTC orchestrator process that accepts incoming queries, makes a tier selection via the FTC rule, and dispatches the query to the actual Ollama backend under the corresponding system prompt (Tool A/B/C from Prompt 2). Use a single backend — pick Qwen 2.5 7B Instruct — to keep the experiment scope tight.
> 2. Use a workload generator that fires 100 queries (sampled from the cyber-IR prompt subset of the trace dataset) according to a Poisson process with λ = 0.08 req/s. **Do not** replay traces — execute against the real LLM.
> 3. Per query, log: arrival time, selected tier, actual service time, completion time, deadline (sampled from TTL Regime II), met-deadline indicator, response text, similarity-to-reference quality score (use the embedding similarity from Prompt 3).
> 4. Compare three configurations: FTC at ε=−0.05, FTC at ε=0, Always-A.
> 5. Repeat for 3 random seeds.
>
> Output:
> - `results/live_e2e_runs.parquet` with all per-query records across runs.
> - `results/fig6_live_e2e.pdf`: a small grouped bar chart showing realized utility and DMR for the three configurations, with error bars across seeds.
> - A 1-paragraph console summary suitable for paste-into-paper, in the form: "Across N live queries (Qwen 2.5 7B, Regime II, λ=0.08 req/s, 3 seeds), FTC at ε=0 achieved realized utility X.XX vs Always-A's Y.YY, with DMR Z% vs W%."
>
> This is intentionally a *small* live experiment. Its job is to demonstrate that the policy works end-to-end, not to replace the larger trace-driven study.
>
> Deliverables: orchestrator code, workload generator, the parquet, the figure, the summary paragraph.

---

## PROMPT 10 — Robustness, strict-priority extension, and notation cleanup

> Final cleanup pass:
>
> 1. **Rename noise-ratio parameter.** In all noise-injection robustness experiments (paper §7.3), rename the noise-ratio variable from `rho` to `eta` everywhere in code, configs, plot labels, table column headers. This fixes the notation collision flagged by Reviewer 1 #10.
>
> 2. **Re-run the noise-injection robustness experiment** with J=3 and the new trace dataset. Same noise levels η ∈ {0.0, 0.1, 0.2, 0.5}. Output a new Table 3 equivalent: `results/table3_robustness_j3.csv`.
>
> 3. **Re-run the strict-priority (premium) extension experiment** from paper §7.6 with J=3. Premium fraction 20% (same as paper). Same TTL regime, same λ. Premium utility means scaled up by 20% per tier as in the paper. Compare Always-A, TTL-aware heuristic, Drift-style, FTC. Output `results/table4_strict_priority_j3.csv`.
>
> 4. **Update the empirical-ECDF figure** (paper Fig 4) to show three curves per backend panel (Tool A, B, C) instead of two. Output `results/fig4_ecdf_j3.pdf`.
>
> 5. **Generate a master results-summary document** as `results/RESULTS_SUMMARY.md` that lists every figure and table produced across Prompts 5–10, with one-sentence captions and the path. This is the deliverable the researcher agent will use to write the results section.
>
> Deliverables: all updated tables/figures and the summary markdown.

---

## After all prompts complete

The researcher agent (instructions in `01_researcher_agent_instructions.md`) can then ingest `results/RESULTS_SUMMARY.md` and the regenerated figures/tables, and rewrite Sections 6, 7, and 8 of the paper to match.

## Quality gates between prompts

Before sending Prompt N+1, verify:
- The artifacts named in Prompt N's "Deliverables" line all exist.
- Any sanity-check assertion in Prompt N (e.g., "tier fractions should shift toward Tool A in Regime IV") is met. If not, do not advance — debug first.
- Test files added by the prompt pass.

## What to do if a prompt produces wrong numbers

The most likely cause of unexpected results is the J=3 ρ-vector calibration. The paper's J=2 result used ρ=0.55 (favoring slow-as-typical). For J=3, the equivalent default is `ρ = (1/3, 1/3, 1/3)` (uniform), but you may need to grid-search a non-uniform vector. Prompt 5 specifies running both — if the uniform vector wildly underperforms the binary baseline, that is a signal to tune more carefully, not a bug.
