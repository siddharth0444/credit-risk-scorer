import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.explain import grouped_permutation_importance, reason_codes, reference_profile
from credit_risk.fairness import group_report, largest_gaps


class LatenessOnlyModel:
    """Toy model whose risk depends only on the latest repayment status."""

    def predict_proba(self, X):
        p = np.clip(0.1 + 0.2 * X["PAY_1"].clip(lower=0).to_numpy(), 0, 1)
        return np.column_stack([1 - p, p])


def customers(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.integers(0, 3, n) for c in config.STATUS})
    X["LIMIT_BAL"] = 100_000
    for c in [*config.BILLS, *config.PAYMENTS]:
        X[c] = rng.integers(0, 50_000, n)
    y = pd.Series(rng.binomial(1, 0.1 + 0.2 * X["PAY_1"]))
    return X[config.BEHAVIOUR], y


def test_reason_codes_point_at_the_factor_that_matters():
    X, y = customers()
    reference = reference_profile(X, y)
    risky = X.iloc[[0]].assign(PAY_1=3)
    reasons = reason_codes(LatenessOnlyModel(), risky, reference)
    assert reasons.iloc[0]["factor"] == "Latest repayment status (Sep)"
    assert reasons.iloc[0]["contribution"] > 0
    assert reasons.iloc[1:]["contribution"].abs().max() == pytest.approx(0)


def test_typical_customer_has_no_contributions():
    X, y = customers()
    reference = reference_profile(X, y)
    typical = pd.DataFrame([reference])
    assert reason_codes(LatenessOnlyModel(), typical, reference)["contribution"].abs().max() == 0


def test_reason_codes_require_one_customer():
    X, y = customers()
    with pytest.raises(ValueError):
        reason_codes(LatenessOnlyModel(), X.head(2), reference_profile(X, y))


def test_permutation_importance_ranks_the_used_feature_first():
    X, y = customers(n=3000)
    imp = grouped_permutation_importance(LatenessOnlyModel(), X, y, n_repeats=3)
    assert imp.iloc[0]["group"] == "Latest repayment status (Sep)"
    assert imp.iloc[0]["auc_drop"] > 0.05
    assert imp.iloc[1:]["auc_drop"].abs().max() < 1e-9


def test_group_report_rates():
    y = np.array([1, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.1])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    rep = group_report(y, p, groups, threshold=0.5, min_size=1)
    assert rep.loc["A", "flagged_rate"] == pytest.approx(2 / 3)
    assert rep.loc["A", "recall"] == 1.0 and rep.loc["B", "recall"] == 0.0
    assert rep.loc["A", "false_positive_rate"] == 0.5
    assert largest_gaps(rep)["recall"] == 1.0


def test_small_groups_are_skipped():
    rep = group_report(np.array([0, 1]), np.array([0.2, 0.7]), pd.Series(["A", "B"]),
                       threshold=0.5, min_size=2)
    assert rep.empty
