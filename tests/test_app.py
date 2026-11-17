"""The Streamlit app starts, scores the default example and reacts to the policy sliders."""

import pytest

from credit_risk import config

APP = config.PROJECT_ROOT / "app" / "streamlit_app.py"
pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                reason="trained model not present; run python -m credit_risk.train")


def run_app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_renders_a_score_without_errors():
    at = run_app()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Probability of default" in labels and "Risk band" in labels


def test_raising_cost_of_missed_default_lowers_threshold():
    at = run_app()
    before = at.sidebar.metric[0].value  # "Flag customers above"
    at.sidebar.slider[0].set_value(20.0).run()
    assert not at.exception
    after = at.sidebar.metric[0].value
    assert float(after.rstrip("%")) < float(before.rstrip("%"))
