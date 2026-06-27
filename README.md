# Context-Aware Dynamic Tool Resolution (CADTR) for Deadline-Constrained Autonomous Agents

This repository contains the simulation framework, data, and analysis code for the
paper **"Context-Aware Dynamic Tool Resolution for Deadline-Constrained Autonomous
Agents"** (PeerJ Computer Science, manuscript ID&nbsp;141785).

It provides a trace-driven, discrete-event simulator of a non-preemptive
single-server queue with **firm deadlines** and a multi-resolution tool menu
(a fast *Tactical* mode and a slower *Strategic* mode). It implements:

- **FTC** (*Firm-Deadline Tool Control*) — an arrival-epoch feasibility-gating policy.
- **CADTR** (*Context-Aware Dynamic Tool Resolution*) — FTC plus an event-driven,
  priority-dependent ε-shift that shields critical tasks under load (the proposed method).
- Baselines/oracles: **F-BB**, **FTC\***, **Mode-Aware Oracle**, **Myopic CDF Oracle**.

Running the steps below reproduces every figure and table in the paper.

---

## Description

Tool-augmented agents under firm deadlines must decide, per request, whether to
invoke a slow tool-rich pipeline (higher utility, higher latency/variance) or a
fast pipeline (base utility, low latency). A missed deadline yields zero utility.
The simulator replays empirically measured LLM service-time traces and evaluates
scheduling/permissioning policies on critical-task miss rate, overall miss rate,
realized utility, and Strategic-tool usage as the arrival rate λ varies.

## Repository layout

```
Configurations.py          # all simulation configuration dataclasses
Core/                      # discrete-event engine (Simulator, Events, Task, ToolTier)
Models/                    # Policies (FTC, CADTR, baselines, oracles), Distributions, Utility
Metrics/                   # metrics collection and summaries
General_Definitions/       # shared types
Tool_Caller_Agent/         # LLM trace generation + the empirical traces (data)
reproduce/                 # scripts that regenerate the paper's tables and figures
tests/                     # self-contained regression test
_archive/                  # superseded/legacy scripts (not needed for reproduction)
requirements.txt
```

## Requirements

- Python 3.9+ (tested on 3.11).
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  Core packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `pytest`.
- **Trace collection only** (optional, not needed to reproduce results from the
  shipped data): a local [Ollama](https://ollama.com) server plus `ollama` and
  `tqdm` Python packages.

## Dataset information

Empirical end-to-end service-time traces are collected by a tool-routing LLM agent.
For each prompt, **both** the FAST and SLOW pipelines are executed (counterfactual
pairing), recording paired latencies, router choice, response text/length, and errors.

| File | Backend | Description |
|------|---------|-------------|
| `Tool_Caller_Agent/trace_results_counterfactual.csv` | Llama&nbsp;3.1 | Canonical trace; the **service-time source for the simulator** (1335 prompts, 1332 usable after error filtering). |
| `Tool_Caller_Agent/trace_results_counterfactual_qwen2.5-7b.csv` | Qwen&nbsp;2.5&nbsp;7B | Cross-backend latency comparison (ECDF). |
| `Tool_Caller_Agent/trace_results_counterfactual_mistral-7b-instruct.csv` | Mistral&nbsp;7B Instruct | Cross-backend latency comparison (ECDF). |

Key columns: `prompt`, `prompt_type`, `router_choice`,
`fast_generation_only_sec`, `slow_generation_only_sec`,
`fast_response_length_char`, `slow_response_length_char`, `fast_error`, `slow_error`.

The simulator replays the **Llama&nbsp;3.1** trace; the Qwen and Mistral traces are
used only for the cross-backend service-time ECDF figure.

## Code information / methodology

- **System model:** single-server, FCFS, non-preemptive queue; Poisson arrivals at
  rate λ; firm deadlines (late or abandoned ⇒ zero utility); two priority classes
  (25% critical in the threat-burst experiment).
- **Utility:** Tactical = base survival utility (1.0); Strategic = base + intelligence
  bonus (1.5); critical tasks ×1.2.
- **TTL regime (main sweep):** fixed (deterministic) deadline budget of 35 s.
- **Policy parameters:** FTC ρ = 0.55, ε = 0; CADTR critical ε-shift Δ = −0.20.
- **Sweep:** λ ∈ {0.03, 0.05, …, 0.19}, 30 replications (seed base 42),
  simulation horizon 100,000 time units, warm-up 1,000.

Configuration lives in `Configurations.py`; the sweep settings are at the top of
`reproduce/run_sweep.py`.

## Usage — reproducing the paper

All commands run from the repository root. Outputs are written under `Results/`
(git-ignored, regenerable).

### 1. Main results: table + "hero" figures

```bash
python reproduce/run_sweep.py            # add --serial on machines that block multiprocessing
python reproduce/make_figures.py
```

- `run_sweep.py` writes `Results/lambda_sweep/sweep_stats.csv` (per-(λ,policy) mean/SD/CI),
  `sweep_raw.csv`, and `main_table.csv` (the representative table at λ ∈ {0.03, 0.11, 0.19}),
  and prints the paired t-tests at λ = 0.11.
- `make_figures.py` reads `sweep_stats.csv` and writes, to `Results/figures/`:

  | Paper figure | File |
  |--------------|------|
  | Critical-task miss rate ("hero graph") | `hero_graph_premium_mr.{png,pdf}` |
  | Strategic usage on critical tasks ("shield") | `shield_mechanism_slow_frac.{png,pdf}` |
  | Overall mean utility | `overall_utility.{png,pdf}` |
  | Overall deadline miss rate | `overall_miss_rate.{png,pdf}` |

  (If the manuscript figure directory `Tex/PeerJ/Results/CADTR/` exists, the PDFs
  are also copied there.)

### 2. Cross-backend service-time ECDF figure

```bash
python reproduce/make_multimodel_ecdf.py
```

Reads the three per-backend traces and writes
`Results/Trace_ECDF/ecdf_fast_vs_slow_multimodel.{png,pdf}` and
`ecdf_multimodel_summary.csv`.

### 3. (Optional) Re-collect LLM traces

Requires a local Ollama server. The Llama-3.1 trace is already provided; to add
other backends (same prompts, same fast/slow process, fixed 50-prompt sample):

```bash
ollama pull qwen2.5:7b
python Tool_Caller_Agent/collect_multimodel_traces.py --model qwen2.5:7b

ollama pull mistral:7b-instruct
python Tool_Caller_Agent/collect_multimodel_traces.py --model mistral:7b-instruct
```

The canonical Llama-3.1 trace was generated with `Tool_Caller_Agent/Agent_V3.py`.

### 4. Tests

```bash
pytest -q
```

A self-contained regression that checks the simulator reproduces known headline
values on the committed Llama-3.1 trace.

## Reproducibility notes

- All runs are seeded (`seed base 42`); results are deterministic per seed.
- `run_sweep.py --serial` avoids Python multiprocessing for restricted environments.
- The service-time source is set in `Configurations.py` (`Service_Config`); switch to
  a lognormal model by setting `service_time_source="lognormal"`.

## Citation

If you use this code or data, please cite:

> M. A. Öztürk and H. Artuner, "Context-Aware Dynamic Tool Resolution for
> Deadline-Constrained Autonomous Agents," PeerJ Computer Science (under review),
> manuscript ID 141785.

## License & contributions

See `LICENSE`. Issues and pull requests are welcome; please open an issue describing
the change before submitting a PR.

## Authors

- **Muhammet Ali Öztürk** — Hacettepe University, Computer Engineering (PhD student).
- **Assoc. Prof. Harun Artuner** — Advisor, Hacettepe University.
