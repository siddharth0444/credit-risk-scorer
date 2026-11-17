"""Feature engineering.

`add_features` is a pure function used inside the model pipeline, so exactly the same
transformation runs during training, evaluation and in the app. It only uses
behaviour columns, never demographics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from . import config


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return the behaviour columns plus engineered features."""
    X = X[config.BEHAVIOUR].astype(float)
    status = X[config.STATUS]
    limit = X["LIMIT_BAL"]
    bills = X[config.BILLS]
    owed = bills.clip(lower=0)  # negative bill = credit balance, i.e. nothing owed

    # A payment in month k settles the statement issued the month before (month k+1).
    paid = X[config.PAYMENTS[:-1]].to_numpy()
    prior_bill = bills[config.BILLS[1:]].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(prior_bill > 0, paid / prior_bill, 1.0)  # nothing owed = fully paid
    ratio = np.clip(ratio, 0, 1.5)

    engineered = pd.DataFrame({
        "months_late": (status >= 1).sum(axis=1),
        "max_delay": status.clip(lower=0).max(axis=1),
        "months_paid_full": (status == -1).sum(axis=1),
        "months_no_balance": (status == -2).sum(axis=1),
        "months_revolving": (status == 0).sum(axis=1),
        "utilization_latest": owed["BILL_AMT1"] / limit,
        "utilization_avg": owed.mean(axis=1) / limit,
        "utilization_max": owed.max(axis=1) / limit,
        "pay_ratio_latest": ratio[:, 0],
        "pay_ratio_avg": ratio.mean(axis=1),
        "balance_growth": (bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit,
    }, index=X.index)
    return pd.concat([X, engineered], axis=1)


def _names_out(transformer, input_features) -> list[str]:
    # Module-level (not a lambda) so the fitted pipeline can be pickled with joblib.
    return feature_names()


def feature_transformer() -> FunctionTransformer:
    """sklearn-compatible wrapper around `add_features`."""
    return FunctionTransformer(add_features, validate=False, feature_names_out=_names_out)


def feature_names() -> list[str]:
    """Column names produced by `add_features`."""
    example = pd.DataFrame(np.ones((1, len(config.BEHAVIOUR))), columns=config.BEHAVIOUR)
    return add_features(example).columns.tolist()
