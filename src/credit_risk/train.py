"""End-to-end training pipeline.

    python -m credit_risk.train

1. Split once into train (80%) and test (20%). The test set is used only in step 6.
2. Compare a baseline, logistic regression and gradient boosting with 5-fold CV.
3. Tune gradient boosting by random search (log loss).
4. Choose the decision threshold on out-of-fold training predictions.
5. Fit the final model on all training data.
6. Evaluate once on the test set: metrics, decisions, explanations, fairness.
7. Save the model, metadata and result tables.
"""

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config, data, evaluation, explain, fairness, models

AUDIT_COLUMNS = {"Sex": "sex", "Age": "age_band", "Education": "education",
                 "Marital status": "marriage"}


@dataclass
class TrainingResult:
    model: object
    metadata: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def run(df: pd.DataFrame, n_iter: int = 30, cv_folds: int = config.CV_FOLDS,
        importance_repeats: int = 10, log=print) -> TrainingResult:
    train, test = data.split(df)
    X_train, y_train = train[config.BEHAVIOUR], train[config.TARGET]
    X_test, y_test = test[config.BEHAVIOUR], test[config.TARGET]
    folds = evaluation.cv(cv_folds)
    log(f"Train {len(train):,} rows, test {len(test):,} rows, "
        f"default rate {y_train.mean():.1%}")

    log("Comparing candidate models with cross-validation...")
    comparison, oof = evaluation.compare(models.candidates(), X_train, y_train, folds)

    log(f"Tuning gradient boosting ({n_iter} combinations x {cv_folds} folds)...")
    search = models.tune_gradient_boosting(X_train, y_train, n_iter=n_iter, cv_folds=cv_folds)
    params = models.best_params(search)
    tuned = models.gradient_boosting(**params)
    tuned_row, tuned_oof = evaluation.compare({"Gradient boosting (tuned)": tuned},
                                              X_train, y_train, folds)
    comparison = pd.concat([comparison, tuned_row])
    oof.update(tuned_oof)
    oof_p = tuned_oof["Gradient boosting (tuned)"]

    threshold = evaluation.best_threshold(y_train, oof_p)
    log(f"Decision threshold (chosen on out-of-fold training predictions): {threshold:.2f}")

    log("Fitting final model and evaluating on the test set...")
    final = tuned.fit(X_train, y_train)
    benchmark = models.logistic_regression().fit(X_train, y_train)
    p_test = final.predict_proba(X_test)[:, 1]
    p_test_lr = benchmark.predict_proba(X_test)[:, 1]
    test_scores = pd.DataFrame({
        "Gradient boosting (tuned)": evaluation.scores(y_test, p_test),
        "Logistic regression": evaluation.scores(y_test, p_test_lr),
    }).T
    decision = evaluation.decision_summary(y_test, p_test, threshold)
    no_model = evaluation.no_model_cost(y_test)

    importance = explain.grouped_permutation_importance(final, X_test, y_test,
                                                        n_repeats=importance_repeats)
    audits = {label: fairness.group_report(y_test, p_test, test[col], threshold)
              for label, col in AUDIT_COLUMNS.items()}
    reference = explain.reference_profile(X_train, y_train)

    # A few real test customers at different risk levels, for the app's examples.
    targets = {"Low risk": 0.05, "Medium risk": 0.25, "Very high risk": 0.75}
    picks = {label: int(np.abs(p_test - t).argmin()) for label, t in targets.items()}
    examples = test.loc[list(picks.values()), [config.ID, *config.BEHAVIOUR]].assign(
        label=list(picks), probability=p_test[list(picks.values())])

    predictions = pd.concat([
        train[[config.ID, config.TARGET, *AUDIT_COLUMNS.values()]].assign(
            split="train_oof", probability=oof_p),
        test[[config.ID, config.TARGET, *AUDIT_COLUMNS.values()]].assign(
            split="test", probability=p_test, probability_benchmark=p_test_lr),
    ], ignore_index=True)

    metadata = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data": {"source": "UCI Default of Credit Card Clients (Taiwan, 2005)",
                 "train_rows": len(train), "test_rows": len(test),
                 "default_rate": float(df[config.TARGET].mean())},
        "inputs": config.BEHAVIOUR,
        "excluded_from_model": config.DEMOGRAPHICS,
        "hyperparameters": params,
        "decision": {"threshold": threshold,
                     "theoretical_threshold": evaluation.theoretical_threshold(),
                     "cost_missed_default": config.COST_MISSED_DEFAULT,
                     "cost_false_alarm": config.COST_FALSE_ALARM},
        "cv_scores": comparison.round(4).to_dict(orient="index"),
        "test_scores": test_scores.round(4).to_dict(orient="index"),
        "test_decision": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in decision.items()},
        "test_no_model_cost": {k: round(v, 4) for k, v in no_model.items()},
        "reference_profile": reference.round(2).to_dict(),
        "risk_bands": config.RISK_BANDS,
    }
    tables = {
        "cv_comparison": comparison,
        "tuning_top10": models.search_results(search),
        "test_scores": test_scores,
        "importance": importance,
        "predictions": predictions,
        "example_customers": examples,
        **{f"fairness_{label.lower().replace(' ', '_')}": rep for label, rep in audits.items()},
    }
    return TrainingResult(final, metadata, tables)


def save(result: TrainingResult, model_dir: Path = config.MODEL_DIR,
         tables_dir: Path = config.TABLES_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_dir / config.MODEL_PATH.name, compress=3)
    (model_dir / config.METADATA_PATH.name).write_text(json.dumps(result.metadata, indent=2))
    for name, table in result.tables.items():
        keep_index = name not in {"predictions", "importance", "tuning_top10", "example_customers"}
        table.to_csv(tables_dir / f"{name}.csv", index=keep_index)


def load_model(path: Path = config.MODEL_PATH):
    return joblib.load(path)


def load_metadata(path: Path = config.METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    n_iter = int(argv[0]) if argv else 30
    result = run(data.load_clean(), n_iter=n_iter)
    save(result)
    test = result.metadata["test_scores"]["Gradient boosting (tuned)"]
    d = result.metadata["test_decision"]
    print(f"\nTest ROC-AUC {test['roc_auc']:.3f} · PR-AUC {test['pr_auc']:.3f} · "
          f"Brier {test['brier']:.4f}")
    print(f"At threshold {d['threshold']:.2f}: flags {d['flagged_rate']:.0%} of customers, "
          f"catches {d['recall']:.0%} of defaults, precision {d['precision']:.0%}")
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
