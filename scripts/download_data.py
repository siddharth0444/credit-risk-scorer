"""Download the UCI 'Default of Credit Card Clients' dataset and save it as CSV.

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Licence: CC BY 4.0. Citation: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data
mining techniques for the predictive accuracy of probability of default of credit card
clients. Expert Systems with Applications, 36(2), 2473-2480.

Usage (from the project root):
    python scripts/download_data.py
"""

import io
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
EXCEL_NAME = "default of credit card clients.xls"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_default.csv"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with tempfile.TemporaryDirectory() as tmp:
        excel = Path(archive.extract(EXCEL_NAME, tmp))
        # Row 0 holds generic labels (X1, X2, ...); real column names are on row 1.
        df = pd.read_excel(excel, header=1, engine="calamine")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {OUT} ({len(df):,} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
