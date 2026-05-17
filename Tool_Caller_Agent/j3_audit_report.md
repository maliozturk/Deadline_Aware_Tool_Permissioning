# J=3 Trace Dataset Audit Report

## 1. Row Counts

- **Total rows**: 1686
- **Unique prompts**: 562
- **Backends**: `['trinity-mini-clarifai']`
- **Tiers**: `['C', 'B', 'A']`
- **Rows with error_flag=True**: 0
- **Rows with empty/zero-length response**: 352 (see §1a below)
- **Rows hitting ~120s timeout**: 0

### 1a. Empty Response Analysis

| Tier | Empty Rows | Total | % |
|------|-----------|-------|---|
| C | 321 | 562 | 57.1% |
| B | 12 | 562 | 2.1% |
| A | 19 | 562 | 3.4% |

**Tier C empties are expected behavior.** The Tier C system prompt instructs:
> *"If you do not know the answer, say 'Unknown' in one word."*

Trinity Mini often returns an empty string instead of "Unknown" — functionally
equivalent. These rows still have valid latency (mean 1.5s) and correctly
record zero tool calls. The non-empty Tier C responses are ultra-short
(e.g., `"UDP"`, `"443"`, `"CVE-2021-44228"`) as designed.

**Tier B/A empties (31 rows, 1.8%)**: The model occasionally returns an empty
content field despite non-trivial latency (3-17s), suggesting the model
processed but chose to return nothing. At 1.8% this is acceptable noise.

**For latency analysis**: All 1686 rows have valid `latency_sec` values — the
empty response text does not invalidate the timing measurement.

## 2. Per-Tier Statistics

| Tier | n | Lat Mean | Lat Med | Lat P10 | Lat P90 | Lat Min | Lat Max | Len Mean | Len Med | Len P10 | Len P90 | TC Mean | TC Max | TC>0 |
|------|---|----------|---------|---------|---------|---------|---------|----------|---------|---------|---------|---------|--------|------|
| C | 562 | 1.723 | 1.505 | 1.130 | 1.931 | 0.929 | 9.267 | 11 | 0 | 0 | 24 | 0.00 | 0 | 0 |
| B | 562 | 3.046 | 2.511 | 2.040 | 3.448 | 1.532 | 20.623 | 543 | 536 | 280 | 787 | 1.00 | 1 | 562 |
| A | 562 | 5.657 | 4.900 | 3.419 | 7.863 | 2.416 | 37.316 | 2079 | 1904 | 1210 | 3252 | 2.88 | 3 | 562 |

## 3. Tier × Complexity Grid

| Complexity | Tier | n | Mean Lat (s) | Mean Resp Len |
|------------|------|---|-------------|---------------|
| complex | C | 249 | 1.785 | 4 |
| complex | B | 249 | 3.278 | 641 |
| complex | A | 249 | 6.332 | 2331 |
| medium | C | 65 | 1.999 | 6 |
| medium | B | 65 | 3.603 | 634 |
| medium | A | 65 | 6.160 | 2749 |
| simple | C | 248 | 1.589 | 19 |
| simple | B | 248 | 2.666 | 419 |
| simple | A | 248 | 4.847 | 1650 |

## 4. Row-Level Tier Ordering

Total triples (prompt_ids with all 3 tiers): 562

### Latency Ordering
- **Strict (A>B>C)**: 484 (86.1%)
- **Partial**: 32 (5.7%)
- **Inversions**: 46 (8.2%)

### Response Length Ordering
- **Strict (A>B>C)**: 532 (94.7%)
- **Partial**: 0 (0.0%)
- **Inversions**: 30 (5.3%)

## 5. Sanity Checks

- **(a) Aggregate latency A>B>C**: **PASS** — A=5.657 > B=3.046 > C=1.723
- **(b) Aggregate length A>B>C**: **PASS** — A=2079 > B=543 > C=11
- **(c) Latency inversion rate <20%**: **PASS** — 8.2%
- **(d) Tier C tool_calls=0**: **PASS** — 0 rows with >0
- **(e) Error rate <2%**: **PASS** — 0.00% (0/1686)
- **(f) Timeout rate <5%**: **PASS** — 0.00% (0/1686)

**Result: 6/6 checks passed.**

## 6. Verdict

**PROCEED WITH CAVEATS**:

- 321 Tier C rows (57%) have empty response text. This is expected behavior
  (the model returns empty instead of "Unknown" for SOC questions). Latency
  values remain valid and the tier separation is clean. For downstream
  simulation, only `latency_sec` is consumed — response text is provenance
  only.

- 31 Tier B/A rows (1.8%) have empty responses with valid latency. Acceptable
  noise level. These can be filtered in analysis if response content is needed.