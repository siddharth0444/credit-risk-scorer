"""Candidate models and hyperparameter tuning.

Every model is a full Pipeline from raw behaviour columns to probabilities, so
feature engineering can never leak information from validation folds.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import feature_transformer


def baseline() -> Pipeline:
    """Predicts the training default rate for everyone. Any real model must beat this."""
    return Pipeline([("features", feature_transformer()),
                     ("model", DummyClassifier(strategy="prior"))])


def logistic_regression() -> Pipeline:
    """Interpretable linear benchmark, similar to a traditional credit scorecard."""
    return Pipeline([("features", feature_transformer()),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=5000, C=1.0))])


def gradient_boosting(**params) -> Pipeline:
    """Histogram gradient boosting: strong on tabular data, no scaling needed."""
    defaults = {"random_state": config.SEED, "max_iter": 400, "early_stopping": True,
                "validation_fraction": 0.1, "n_iter_no_change": 20}
    return Pipeline([("features", feature_transformer()),
                     ("model", HistGradientBoostingClassifier(**{**defaults, **params}))])


def candidates() -> dict[str, Pipeline]:
    return {
        "Baseline (predict average)": baseline(),
        "Logistic regression": logistic_regression(),
        "Gradient boosting (default settings)": gradient_boosting(),
    }


SEARCH_SPACE = {
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__max_leaf_nodes": [7, 15, 31, 63],
    "model__min_samples_leaf": [20, 50, 100, 200, 400],
    "model__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "model__max_features": [0.5, 0.8, 1.0],
}


def tune_gradient_boosting(X: pd.DataFrame, y: pd.Series, n_iter: int = 30,
                           cv_folds: int = config.CV_FOLDS,
                           seed: int = config.SEED) -> RandomizedSearchCV:
    """Random search optimising log loss.

    Log loss rewards well-calibrated probabilities, not just correct ranking. That
    matters because decisions are made by comparing probabilities with costs.
    """
    search = RandomizedSearchCV(
        gradient_boosting(), SEARCH_SPACE, n_iter=n_iter, scoring="neg_log_loss",
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
        random_state=seed, n_jobs=-1, refit=False,
    )
    return search.fit(X, y)


def search_results(search: RandomizedSearchCV, top: int = 10) -> pd.DataFrame:
    """Top parameter combinations from a search, best first."""
    res = pd.DataFrame(search.cv_results_)
    params = res["params"].apply(pd.Series).rename(columns=lambda c: c.replace("model__", ""))
    out = pd.concat([params, pd.DataFrame({
        "log_loss": -res["mean_test_score"], "log_loss_std": res["std_test_score"],
    })], axis=1)
    return out.sort_values("log_loss").head(top).reset_index(drop=True)


def best_params(search: RandomizedSearchCV) -> dict:
    return {k.replace("model__", ""): (v.item() if isinstance(v, np.generic) else v)
            for k, v in search.best_params_.items()}
