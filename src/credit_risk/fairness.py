"""Fairness audit: how the model's decisions differ across demographic groups.

The model never sees sex, age, education or marital status. That alone does not
make it fair: behaviour can correlate with demographics. So decisions are audited
by group after the fact.
"""

import numpy as np
import pandas as pd


def group_report(y: np.ndarray, p: np.ndarray, groups: pd.Series, threshold: float,
                 min_size: int = 100) -> pd.DataFrame:
    """Per-group outcome rates at a decision threshold.

    * flagged_rate: share of the group flagged (demographic parity)
    * recall: share of the group's defaulters that are caught (equal opportunity)
    * false_positive_rate: share of the group's good payers wrongly flagged
    * calibration_gap: mean predicted minus actual default rate (positive = overestimated)
    """
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p),
                          "group": np.asarray(groups)})
    frame["flagged"] = frame["p"] >= threshold
    rows = []
    for name, g in frame.groupby("group", observed=True):
        if len(g) < min_size:
            continue
        defaulters, payers = g[g["y"] == 1], g[g["y"] == 0]
        rows.append({
            "group": name, "customers": len(g),
            "default_rate": g["y"].mean(),
            "mean_predicted": g["p"].mean(),
            "calibration_gap": g["p"].mean() - g["y"].mean(),
            "flagged_rate": g["flagged"].mean(),
            "recall": defaulters["flagged"].mean() if len(defaulters) else np.nan,
            "false_positive_rate": payers["flagged"].mean() if len(payers) else np.nan,
        })
    columns = ["group", "customers", "default_rate", "mean_predicted", "calibration_gap",
               "flagged_rate", "recall", "false_positive_rate"]
    return pd.DataFrame(rows, columns=columns).set_index("group")


def largest_gaps(report: pd.DataFrame,
                 metrics: tuple[str, ...] = ("recall", "false_positive_rate")) -> pd.Series:
    """Max minus min across groups, for each metric."""
    return (report[list(metrics)].max() - report[list(metrics)].min()).rename("gap")
