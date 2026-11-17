"""Smoke test: the whole pipeline runs end to end and produces usable artifacts."""

import json

from credit_risk import config, train
from credit_risk.data import clean
from tests.conftest import make_raw


def test_pipeline_runs_and_saves_artifacts(tmp_path):
    result = train.run(clean(make_raw(n=800)), n_iter=2, cv_folds=3, importance_repeats=1,
                       log=lambda *_: None)

    meta = result.metadata
    assert 0 < meta["decision"]["threshold"] < 1
    assert meta["excluded_from_model"] == config.DEMOGRAPHICS
    assert meta["test_scores"]["Gradient boosting (tuned)"]["roc_auc"] > 0.6
    assert set(result.tables["predictions"]["split"]) == {"train_oof", "test"}

    train.save(result, model_dir=tmp_path / "models", tables_dir=tmp_path / "tables")
    model = train.load_model(tmp_path / "models" / config.MODEL_PATH.name)
    saved = json.loads((tmp_path / "models" / config.METADATA_PATH.name).read_text())
    assert saved["hyperparameters"] == meta["hyperparameters"]

    one = clean(make_raw(n=5, seed=9))[config.BEHAVIOUR].head(1)
    p = model.predict_proba(one)[0, 1]
    assert 0 <= p <= 1
