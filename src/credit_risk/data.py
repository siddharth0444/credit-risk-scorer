"""Loading, validating, cleaning and splitting the dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

RAW_TARGET = "default payment next month"
RAW_COLUMNS = ["ID", "LIMIT_BAL", *config.DEMOGRAPHICS, "PAY_0", *config.STATUS[1:],
               *config.BILLS, *config.PAYMENTS, RAW_TARGET]


def load_raw(path: Path = config.RAW_DATA) -> pd.DataFrame:
    """Read the CSV written by scripts/download_data.py."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix naming quirks, validate, and add readable demographic labels for auditing.

    * The source names its first repayment column PAY_0 but the rest PAY_2..PAY_6;
      it is renamed PAY_1 so that month numbers line up with BILL_AMT1 / PAY_AMT1.
    * Undocumented EDUCATION (0, 4, 5, 6) and MARRIAGE (0, 3) codes become "Other / unknown".
    """
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = raw.rename(columns={"PAY_0": "PAY_1", RAW_TARGET: config.TARGET})
    validate(df)

    df["sex"] = df["SEX"].map(config.SEX_LABELS).fillna(config.UNKNOWN)
    df["education"] = df["EDUCATION"].map(config.EDUCATION_LABELS).fillna(config.UNKNOWN)
    df["marriage"] = df["MARRIAGE"].map(config.MARRIAGE_LABELS).fillna(config.UNKNOWN)
    df["age_band"] = pd.cut(df["AGE"], bins=config.AGE_BINS, labels=config.AGE_LABELS)
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the data breaks an assumption the model relies on."""
    if df[config.ID].duplicated().any():
        raise ValueError("Duplicate customer IDs.")
    if not df[config.TARGET].isin([0, 1]).all():
        raise ValueError("Target must be 0 or 1.")
    if df[config.BEHAVIOUR].isna().any().any():
        raise ValueError("Missing values in behaviour columns.")
    if (df["LIMIT_BAL"] <= 0).any():
        raise ValueError("Credit limits must be positive.")
    if (df[config.PAYMENTS] < 0).any().any():
        raise ValueError("Payment amounts cannot be negative.")
    if not df[config.STATUS].isin(list(config.STATUS_LABELS)).all().all():
        raise ValueError("Unexpected repayment status code.")


def load_clean(path: Path = config.RAW_DATA) -> pd.DataFrame:
    return clean(load_raw(path))


def split(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
          seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split. The test set is touched once, at the very end."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed,
                                   stratify=df[config.TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)
