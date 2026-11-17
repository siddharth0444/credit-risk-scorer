"""Model evaluation and the cost-based decision policy."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config


def cv(folds: int = config.CV_FOLDS, seed: int = config.SEED) -> StratifiedKFold:
    return StratifiedKFold(folds, shuffle=True, random_state=seed)


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Ranking quality (ROC-AUC, PR-AUC) and probability quality (Brier, log loss)."""
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }


def out_of_fold(model, X: pd.DataFrame, y: pd.Series, folds=None) -> np.ndarray:
    """Probability for every training row from a model that never saw that row."""
    return cross_val_predict(clone(model), X, y, cv=folds or cv(), method="predict_proba")[:, 1]


def compare(models: dict, X: pd.DataFrame, y: pd.Series, folds=None) -> tuple[pd.DataFrame, dict]:
    """Cross-validated scores per model (mean and std across folds) plus OOF predictions."""
    folds = folds or cv()
    rows, oof = [], {}
    for name, model in models.items():
        oof[name] = out_of_fold(model, X, y, folds)
        per_fold = [scores(y.iloc[te], oof[name][te]) for _, te in folds.split(X, y)]
        frame = pd.DataFrame(per_fold)
        rows.append({"model": name,
                     **{f"{k}": v for k, v in frame.mean().items()},
                     **{f"{k}_std": v for k, v in frame.std().items()}})
    return pd.DataFrame(rows).set_index("model"), oof


# --- Decision policy --------------------------------------------------------

def decision_summary(y: np.ndarray, p: np.ndarray, threshold: float,
                     cost_missed: float = config.COST_MISSED_DEFAULT,
                     cost_false_alarm: float = config.COST_FALSE_ALARM) -> dict[str, float]:
    """Outcome of flagging every customer with probability >= threshold."""
    y, flagged = np.asarray(y), np.asarray(p) >= threshold
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    n = len(y)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged_rate": (tp + fp) / n,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "cost_per_customer": (cost_missed * fn + cost_false_alarm * fp) / n,
    }


def cost_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray | None = None,
               **costs) -> pd.DataFrame:
    """Decision summary across a grid of thresholds."""
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2) if thresholds is None else thresholds
    return pd.DataFrame([decision_summary(y, p, t, **costs) for t in thresholds])


def best_threshold(y: np.ndarray, p: np.ndarray, **costs) -> float:
    """Threshold with the lowest expected cost per customer."""
    curve = cost_curve(y, p, **costs)
    return float(curve.loc[curve["cost_per_customer"].idxmin(), "threshold"])


def theoretical_threshold(cost_missed: float = config.COST_MISSED_DEFAULT,
                          cost_false_alarm: float = config.COST_FALSE_ALARM) -> float:
    """Optimal threshold *if* probabilities are perfectly calibrated.

    Flag when p * cost_missed > (1 - p) * cost_false_alarm,
    i.e. p > cost_false_alarm / (cost_false_alarm + cost_missed).
    """
    return cost_false_alarm / (cost_false_alarm + cost_missed)


def no_model_cost(y: np.ndarray, cost_missed: float = config.COST_MISSED_DEFAULT,
                  cost_false_alarm: float = config.COST_FALSE_ALARM) -> dict[str, float]:
    """Cost of the two strategies that need no model: flag nobody, or flag everyone."""
    rate = float(np.mean(y))
    return {"flag_nobody": cost_missed * rate, "flag_everyone": cost_false_alarm * (1 - rate)}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed default rate in equal-sized probability bins."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    out = frame.groupby("bin").agg(customers=("y", "size"), predicted=("p", "mean"),
                                   observed=("y", "mean"))
    return out.reset_index(drop=True)


def risk_band(p: float) -> str:
    for upper, label in config.RISK_BANDS:
        if p < upper:
            return label
    return config.RISK_BANDS[-1][1]
