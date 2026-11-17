import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.features import add_features, feature_names


def customer(**overrides) -> pd.DataFrame:
    row = {"LIMIT_BAL": 100_000, **dict.fromkeys(config.STATUS, 0),
           **dict.fromkeys(config.BILLS, 50_000), **dict.fromkeys(config.PAYMENTS, 10_000)}
    row.update(overrides)
    return pd.DataFrame([row])


def test_lateness_counts():
    f = add_features(customer(PAY_1=2, PAY_2=1, PAY_3=-1, PAY_4=-2)).iloc[0]
    assert f["months_late"] == 2
    assert f["max_delay"] == 2
    assert f["months_paid_full"] == 1
    assert f["months_no_balance"] == 1
    assert f["months_revolving"] == 2


def test_utilization_ignores_credit_balances():
    f = add_features(customer(BILL_AMT1=-5_000)).iloc[0]
    assert f["utilization_latest"] == 0
    assert f["utilization_avg"] == pytest.approx(5 * 50_000 / 6 / 100_000)


def test_payment_ratio_compares_payment_with_previous_statement():
    # September's payment (PAY_AMT1) settles August's statement (BILL_AMT2).
    f = add_features(customer(PAY_AMT1=25_000, BILL_AMT2=50_000)).iloc[0]
    assert f["pay_ratio_latest"] == pytest.approx(0.5)


def test_nothing_owed_counts_as_fully_paid_and_no_division_by_zero():
    f = add_features(customer(**dict.fromkeys(config.BILLS, 0),
                              **dict.fromkeys(config.PAYMENTS, 0)))
    assert np.isfinite(f.to_numpy()).all()
    assert f.iloc[0]["pay_ratio_avg"] == 1.0


def test_demographics_are_ignored_and_names_match():
    with_demo = customer().assign(SEX=2, AGE=30)
    out = add_features(with_demo)
    assert "SEX" not in out and "AGE" not in out
    assert out.columns.tolist() == feature_names()
