import numpy as np
import pandas as pd
import pytest

from credit_risk import config
from credit_risk.data import RAW_TARGET


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic data in the raw source format, where lateness drives default."""
    rng = np.random.default_rng(seed)
    late = rng.choice([-2, -1, 0, 1, 2, 3], size=(n, 6), p=[0.1, 0.2, 0.45, 0.1, 0.1, 0.05])
    limit = rng.choice([20_000, 50_000, 100_000, 200_000], size=n)
    bills = (limit[:, None] * rng.uniform(0, 0.9, size=(n, 6))).round()
    payments = (bills * rng.uniform(0, 0.5, size=(n, 6))).round()
    risk = 1 / (1 + np.exp(-(-2.2 + 0.9 * late[:, 0].clip(0) + 0.3 * late[:, 1:].clip(0).sum(1))))
    frame = pd.DataFrame({
        "ID": np.arange(1, n + 1), "LIMIT_BAL": limit,
        "SEX": rng.choice([1, 2], n), "EDUCATION": rng.choice([0, 1, 2, 3, 5], n),
        "MARRIAGE": rng.choice([0, 1, 2, 3], n), "AGE": rng.integers(21, 70, n),
        "PAY_0": late[:, 0],
        **{f"PAY_{i}": late[:, i - 1] for i in range(2, 7)},
        **{c: bills[:, i] for i, c in enumerate(config.BILLS)},
        **{c: payments[:, i] for i, c in enumerate(config.PAYMENTS)},
        RAW_TARGET: rng.binomial(1, risk),
    })
    return frame


@pytest.fixture
def raw():
    return make_raw()


@pytest.fixture
def clean(raw):
    from credit_risk.data import clean
    return clean(raw)
