import pytest

from credit_risk import config
from credit_risk.data import clean, split


def test_clean_renames_pay0_and_target(raw):
    df = clean(raw)
    assert "PAY_1" in df and "PAY_0" not in df
    assert config.TARGET in df
    assert (df["PAY_1"] == raw["PAY_0"]).all()


def test_undocumented_codes_become_unknown(raw):
    df = clean(raw)
    assert set(df.loc[raw["EDUCATION"].isin([0, 5]), "education"]) == {config.UNKNOWN}
    assert set(df.loc[raw["MARRIAGE"].isin([0, 3]), "marriage"]) == {config.UNKNOWN}
    assert set(df.loc[raw["SEX"] == 2, "sex"]) == {"Female"}


@pytest.mark.parametrize("column, value, message", [
    ("ID", 2, "Duplicate"),  # row 0 takes row 1's ID
    ("default payment next month", 2, "Target"),
    ("LIMIT_BAL", 0, "positive"),
    ("PAY_AMT1", -5, "negative"),
    ("PAY_0", 11, "status"),
])
def test_validation_rejects_bad_data(raw, column, value, message):
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        clean(raw)


def test_missing_column_is_reported(raw):
    with pytest.raises(ValueError, match="Missing columns"):
        clean(raw.drop(columns=["BILL_AMT3"]))


def test_split_is_stratified_and_disjoint(clean):
    train, test = split(clean, test_size=0.25)
    assert set(train["ID"]).isdisjoint(test["ID"])
    assert len(train) + len(test) == len(clean)
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.02
