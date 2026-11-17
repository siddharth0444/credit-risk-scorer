"""Project-wide constants. Every modelling and business assumption is stated here."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "credit_default.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "credit_risk_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --- Columns ---------------------------------------------------------------
ID = "ID"
TARGET = "default"
# Month 1 is September 2005 (the month before the outcome), month 6 is April 2005.
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
STATUS = [f"PAY_{i}" for i in range(1, 7)]       # repayment status per month
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]    # statement balance per month (NT$)
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]  # amount paid per month (NT$)
BEHAVIOUR = ["LIMIT_BAL", *STATUS, *BILLS, *PAYMENTS]
DEMOGRAPHICS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

# Repayment status codes. -2 and 0 are not in the official documentation; their
# meaning was inferred from the data (notebook 01): -2 customers often have a zero
# bill, and 0 customers carry large revolving balances.
STATUS_LABELS = {
    -2: "No balance / card not used",
    -1: "Paid in full",
    0: "Revolving (minimum paid)",
    **{k: f"{k} month{'s' if k > 1 else ''} late" for k in range(1, 9)},
}

# Demographics are used only to audit the model for fairness, never as model inputs.
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school"}
MARRIAGE_LABELS = {1: "Married", 2: "Single"}
UNKNOWN = "Other / unknown"
AGE_BINS = [20, 30, 40, 50, 60, 80]
AGE_LABELS = ["21-30", "31-40", "41-50", "51-60", "61+"]

# --- Experiment settings ---------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Business assumptions ---------------------------------------------------
# Relative cost of each mistake when deciding which customers to flag for action
# (e.g. a credit-limit review or a proactive repayment-plan call).
# Missing a customer who then defaults costs far more (unpaid balance) than
# contacting a customer who would have paid anyway (staff time, some annoyance).
# 5:1 is an illustrative assumption; the app lets users change it.
COST_MISSED_DEFAULT = 5.0
COST_FALSE_ALARM = 1.0

# Risk bands shown to users, by predicted probability of default.
RISK_BANDS = [(0.10, "Low"), (0.30, "Medium"), (0.50, "High"), (1.01, "Very high")]

# Feature groups for explanations, in terms a credit officer would use.
FEATURE_GROUPS = {
    "Latest repayment status (Sep)": ["PAY_1"],
    "Earlier repayment history (Apr-Aug)": STATUS[1:],
    "Credit limit": ["LIMIT_BAL"],
    "Outstanding balances": BILLS,
    "Recent payment amounts": PAYMENTS,
}
