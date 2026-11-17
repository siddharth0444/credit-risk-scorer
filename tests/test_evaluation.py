import numpy as np
import pytest

from credit_risk.evaluation import (
    best_threshold,
    calibration_table,
    decision_summary,
    no_model_cost,
    risk_band,
    theoretical_threshold,
)


def test_decision_summary_counts_and_cost():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.05])
    s = decision_summary(y, p, threshold=0.5, cost_missed=5, cost_false_alarm=1)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (1, 1, 1, 2)
    assert s["recall"] == 0.5 and s["precision"] == 0.5
    assert s["cost_per_customer"] == pytest.approx((5 * 1 + 1 * 1) / 5)


def test_best_threshold_matches_theory_for_calibrated_probabilities():
    # Outcomes drawn from the probabilities themselves = perfectly calibrated.
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200_000)
    y = rng.binomial(1, p)
    found = best_threshold(y, p, cost_missed=5, cost_false_alarm=1)
    assert found == pytest.approx(theoretical_threshold(5, 1), abs=0.02)


def test_no_model_costs():
    costs = no_model_cost(np.array([1, 0, 0, 0]), cost_missed=5, cost_false_alarm=1)
    assert costs == {"flag_nobody": 1.25, "flag_everyone": 0.75}


def test_calibration_table_recovers_rates():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 50_000)
    table = calibration_table(rng.binomial(1, p), p, bins=5)
    assert len(table) == 5
    assert np.allclose(table["predicted"], table["observed"], atol=0.02)


@pytest.mark.parametrize("p, band", [(0.05, "Low"), (0.2, "Medium"), (0.4, "High"),
                                     (0.8, "Very high"), (1.0, "Very high")])
def test_risk_bands(p, band):
    assert risk_band(p) == band
