"""
Self-contained regression test for the FTC/CADTR discrete-event simulator.

Verifies that the simulator + policies reproduce known headline numbers from
the trace-driven lambda sweep (constant TTL=35, seed base 42, the configuration
that produced the paper's results). Uses the committed Llama-3.1 trace, so it
needs no externally generated files.

Run:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.run_sweep import run_single_simulation  # noqa: E402

ABS_TOL = 1e-6

# (lambda, policy_key) -> (premium_miss_rate, overall_utility), replication 0.
# Reference values produced by the current code on the committed trace.
REFERENCE = {
    (0.03, "cadtr"):      (0.04046639, 1.22666217),
    (0.11, "cadtr"):      (0.03820266, 1.07347037),
    (0.19, "cadtr"):      (0.03137545, 1.04558393),
    (0.11, "ftc_star"):   (0.05584961, 1.06196306),
    (0.11, "cdf_oracle"): (0.13214671, 0.98341270),
    (0.11, "mode_aware"): (0.10694598, 1.00564391),
    (0.11, "f_bb"):       (0.03526858, 1.09148835),
}

POLICY_LABEL = {
    "cadtr": "CADTR", "ftc_star": "FTC*", "cdf_oracle": "Myopic CDF Oracle",
    "mode_aware": "Mode-Aware Oracle", "f_bb": "F-BB",
}


@pytest.mark.parametrize("lam,key", list(REFERENCE.keys()))
def test_reproduces_reference(lam, key):
    exp_mr, exp_util = REFERENCE[(lam, key)]
    r = run_single_simulation((lam, POLICY_LABEL[key], key, 0))
    assert r["premium_miss_rate"] == pytest.approx(exp_mr, abs=ABS_TOL)
    assert r["overall_utility"] == pytest.approx(exp_util, abs=ABS_TOL)


def test_deterministic():
    a = run_single_simulation((0.11, "CADTR", "cadtr", 0))
    b = run_single_simulation((0.11, "CADTR", "cadtr", 0))
    assert a["premium_miss_rate"] == b["premium_miss_rate"]
    assert a["overall_utility"] == b["overall_utility"]


def test_cadtr_shields_critical_vs_oracles_at_load():
    """Central claim: under load CADTR misses fewer critical tasks than oracles."""
    cadtr = run_single_simulation((0.11, "CADTR", "cadtr", 0))["premium_miss_rate"]
    cdf = run_single_simulation((0.11, "Myopic CDF Oracle", "cdf_oracle", 0))["premium_miss_rate"]
    ma = run_single_simulation((0.11, "Mode-Aware Oracle", "mode_aware", 0))["premium_miss_rate"]
    assert cadtr < cdf
    assert cadtr < ma
