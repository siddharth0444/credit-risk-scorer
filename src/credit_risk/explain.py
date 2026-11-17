"""Explanations in business terms.

Both methods work on *groups* of raw inputs (e.g. all six monthly balances), because
the model's engineered features are built from several raw columns and a credit
officer thinks in concepts like "repayment history", not "BILL_AMT4".
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config


def grouped_permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                                   groups: dict[str, list[str]] = config.FEATURE_GROUPS,
                                   n_repeats: int = 10, seed: int = config.SEED) -> pd.DataFrame:
    """Drop in ROC-AUC when a whole group of columns is shuffled together.

    Shuffling breaks the link between that information and the outcome. A large drop
    means the model relies on it. Columns within a group are shuffled with the same
    permutation so their relationship to each other is kept.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[cols] = X[cols].to_numpy()[order]
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        rows.append({"group": name, "auc_drop": np.mean(drops), "auc_drop_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def reference_profile(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """A 'typical customer who paid': the median of each column among non-defaulters."""
    return X.loc[y == 0, config.BEHAVIOUR].median()


def reason_codes(model, customer: pd.DataFrame, reference: pd.Series,
                 groups: dict[str, list[str]] = config.FEATURE_GROUPS) -> pd.DataFrame:
    """How much each group of inputs moves this customer's risk away from typical.

    For each group, replace the customer's values with the reference profile and
    re-score. The change in probability is that group's contribution: positive means
    it raises the customer's risk. This is a simple, model-agnostic method. It can't
    capture every interaction, but it is easy to explain and to verify.
    """
    if len(customer) != 1:
        raise ValueError("Pass exactly one customer.")
    p = model.predict_proba(customer)[0, 1]
    rows = []
    for name, cols in groups.items():
        counterfactual = customer.copy()
        counterfactual[cols] = reference[cols].to_numpy()
        p_typical = model.predict_proba(counterfactual)[0, 1]
        rows.append({"factor": name, "contribution": p - p_typical})
    out = pd.DataFrame(rows)
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index) \
              .reset_index(drop=True).assign(probability=p)
