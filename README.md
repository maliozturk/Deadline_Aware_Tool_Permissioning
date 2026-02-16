# Deadline-Aware Tool Permissioning (DATP) for Tool-Augmented LLM Serving Under Firm Deadlines

This repository provides a discrete-event simulation framework for deadline-constrained LLM serving with multiple inference modes. It introduces Deadline Aware Tool Permissioning (DATP), a policy that gates slow vs fast inference using deadline slack and an optional epsilon margin. The simulation uses empirical service-time traces (paired FAST/SLOW) or lognormal models, and supports standard baseline policies for rigorous comparisons.

The outputs are designed for publication-quality figures and tables. The proposed policy is labeled as "DATP*" in all plots and tables.

## System model
- Single-server queue, FCFS, non-preemptive service.
- Poisson arrivals with rate lambda (configurable in each experiment script).
- Deadlines drawn from a truncated Normal distribution with minimum TTL 0.05 seconds.
- Queue drop on deadline expiry enabled.
- Firm-deadline utility: missed deadlines yield zero utility.
- Simulation horizon 100000, warmup 1000.
- Empirical service-time source is the default (paired trace sampling).

## Policies and baselines
- AlwaysFast (AF)
- AlwaysSlow (AS)
- StaticMix (p=0.5)
- QueueThreshold (Q0=5)
- DriftPenaltyMyopic (V=1.0)
- TTL-aware heuristic
- Proposed: DATP* (Deadline Aware Tool Permissioning; epsilon can be fixed or adaptive)

## TTL regimes (used in outputs)
- TTL Regime I (mu=28, sigma=9) -> very_tight
- TTL Regime II (mu=35, sigma=10) -> tight
- TTL Regime III (mu=45, sigma=12) -> relaxed
- TTL Regime IV (mu=60, sigma=15) -> very_relaxed

All figures and tables map preset names to the "TTL Regime I-IV" labels.

## Empirical service-time data and counterfactual generation
Empirical service times are derived from counterfactual FAST/SLOW generations using an LLM agent:
- Generator: `Tool_Caller_Agent/Agent_V3.py`
- Model: Ollama `llama3.1`
- Tool schema: `fast_lookup` and `deep_reasoner`
- Prompt sets: `Tool_Caller_Agent/Prompts.py`

Tool definitions (router phase):
- `fast_lookup`: short, factual, single-sentence answers for simple questions.
- `deep_reasoner`: long, step-by-step reasoning for complex prompts.

Tool schema JSON (from `Tool_Caller_Agent/Agent_V3.py`):
```json
[
  {
    "type": "function",
    "function": {
      "name": "fast_lookup",
      "description": "Use this for simple, factual, quick questions that do not require reasoning.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The query to answer"
          }
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "deep_reasoner",
      "description": "Use this for complex, multi-step, philosophical, or coding tasks requiring detailed explanation.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The complex query"
          }
        },
        "required": ["query"]
      }
    }
  }
]
```

The router call exposes these tools to the model. For trace generation we still execute both FAST and SLOW instructions to record counterfactual latencies for the same prompt.
- Outputs: `Tool_Caller_Agent/trace_results_counterfactual.db` and `Tool_Caller_Agent/trace_results_counterfactual.csv`

The CSV contains prompt metadata, router choices, latencies, response lengths, and full response text. This database can be reused for other research tasks such as offline routing or response-length modeling.

Example paired responses (text truncated):

```text
Prompt: Which ocean is on the east coast of the United States?
FAST (0.3283s): The Atlantic Ocean is on the east coast of the United States.
SLOW (6.4724s): To answer this question, we must consider the geography of the United States and its relationship to the world's oceans. Step 1: Identify the major oceans surrounding the Americas. There are five major oceans that...

Prompt: Which planet is known as the Red Planet?
FAST (0.5113s): Mars is known as the Red Planet due to its reddish appearance caused by iron oxide in its soil.
SLOW (8.9127s): Let's break down this question step by step to understand which planet is indeed known as the Red Planet. Step 1: Understanding the question. The question asks us to identify a specific planet that is commonly...

Prompt: Compare Monte Carlo methods vs deterministic numerical methods for estimating integrals or uncertainties; provide use-cases.
FAST (0.6478s): Monte Carlo methods are stochastic and more accurate for complex problems, but deterministic numerical methods like Gaussian quadrature are faster and more efficient for simple problems with known distributions.
SLOW (14.9888s): Monte Carlo Methods vs Deterministic Numerical Methods: A Comparative Analysis. Numerical integration is a fundamental task in mathematics, physics, engineering, and other fields, where the objective is to estimate...
```

Summary utilities:
- `Tool_Caller_Agent/Agent_V3.py` prints descriptive stats and a lognormal fit snippet.

[//]: # (- `Analysis/Agent_Results_Stats.py` writes summary stats for the first 1335 usable FAST/SLOW samples.)

[//]: # (## Reproducing paper outputs)

[//]: # (All commands are run from the repository root.)

[//]: # ()
[//]: # (Recommended &#40;pipeline runner&#41;:)

[//]: # (```bash)

[//]: # (python Analysis/Run_Paper_Outputs.py)

[//]: # (```)

[//]: # (This executes the following scripts in order:)

[//]: # (- `Analysis/DP_vs_DATP_V_Sweep.py`)

[//]: # (- `Analysis/Trace_ECDF.py`)

[//]: # (- `Analysis/Policy_GridSearch_vs_DATP.py`)

[//]: # (- `Analysis/Agent_Results_Stats.py`)

[//]: # (- `Analysis/Journal_Experiments.py`)

[//]: # ()
[//]: # (Individual scripts:)

[//]: # (```bash)

[//]: # (python Analysis/Journal_Experiments.py)

[//]: # (python Analysis/DP_vs_DATP_V_Sweep.py)

[//]: # (python Analysis/Policy_GridSearch_vs_DATP.py)

[//]: # (python Analysis/Trace_ECDF.py)

[//]: # (python Analysis/Premium_Priority_Table.py)

[//]: # (```)

[//]: # ()
[//]: # (### Experiment outputs)

[//]: # (- Main table &#40;D1&#41;: `Results/Journal/main_table/main_table.csv`)

[//]: # (- Load sweep &#40;D2&#41;: `Results/Journal/load_sweep/utility_vs_lambda.png`, `Results/Journal/load_sweep/dmr_vs_lambda.png`)

[//]: # (- Epsilon trade-off &#40;D3&#41;: `Results/Journal/epsilon_tradeoff/DATP_epsilon_pareto.png`)

[//]: # (- ECDF plots: `Results/Trace_ECDF/ecdf_fast_vs_slow.png`, `Results/Trace_ECDF/ecdf_fast_vs_slow_by_prompt_type.png`)

[//]: # (- Premium priority table: `Results/Journal/premium_table/premium_policy_table.csv`)

[//]: # (- Agent stats table: `Results/Agent_Stats/agent_results_stats.csv`)

### Configuration points
- Global defaults: `Configurations.py`
- Journal experiments: `Analysis/Journal_Experiments.py` (lambda sweep, epsilon list, TTL presets)
- DATP settings: `Configurations.py` under `Policy_Config`
- Service-time source: `Configurations.py` under `Service_Config`

If you want to switch to a lognormal service model or to a different trace CSV, update `Configurations.py`.

## Repository layout
- `Analysis/`: experiment scripts and plotting utilities.
- `Core/`: simulation engine and task definitions.
- `Models/`: policies, distributions, and utility models.
- `Metrics/`: metrics collection and summary stats.
- `Tool_Caller_Agent/`: trace generation and raw data assets.
- `Results/`: generated figures and tables.

## Agent specifications
The trace generator agent uses:
- Ollama client with model `llama3.1`.
- Router tool schema with two tools: `fast_lookup` (short answer) and `deep_reasoner` (long answer).
- Counterfactual execution of both FAST and SLOW prompts to record paired service times.
- DB schema stored in `Tool_Caller_Agent/trace_results_counterfactual.db`.

## Compute environment (tested)
- OS: Windows 11 Home 10.0.26100 (64-bit)
- CPU: AMD Ryzen 7 5800H with Radeon Graphics
- GPU: RTX 3080 Laptop GPU (8GB)
- RAM: 31.4 GiB
- Python: 3.9.13
- Key packages: numpy 2.4.1, matplotlib 3.10.8
- Optional for trace collection: ollama, tqdm

## Notes on reproducibility
- All stochastic runs are seeded; see `Simulation_Config.seed_i32` and experiment scripts for seed offsets.
- Default service-time sampling uses the trace CSV; ensure it is available at `Tool_Caller_Agent/trace_results_counterfactual.csv`.
- Figures follow a unified serif style for publication quality (DPI 300, consistent font and line settings).

## Thanks
If you like our work, please cite/star the repository. Also feel free to write for comments/improvements. 

Primary Author: **Muhammet Ali Ozturk** (muhammetaliozturk.official@gmail.com) (Turkiye, Hacettepe University - Computer Engineering Department. PhD. Student)

Advisor: **Assoc. Prof. Harun Artuner** (harun.artuner@gmail.com)
