"""Credit Risk Scorer: interactive demo.

Run locally:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk import config, evaluation, explain, plots, train  # noqa: E402

st.set_page_config(page_title="Credit Risk Scorer", page_icon=":material/credit_card:",
                   layout="wide")
plots.set_style()

STATUS_OPTIONS = list(config.STATUS_LABELS.values())
LABEL_TO_CODE = {v: k for k, v in config.STATUS_LABELS.items()}


@st.cache_resource
def load_artifacts():
    meta = train.load_metadata()
    preds = pd.read_csv(config.TABLES_DIR / "predictions.csv")
    examples = pd.read_csv(config.TABLES_DIR / "example_customers.csv")
    examples.index = [f"{r.label} (customer {r.ID})" for r in examples.itertuples()]
    return train.load_model(), meta, preds, examples


model, meta, preds, examples = load_artifacts()
reference = pd.Series(meta["reference_profile"])[config.BEHAVIOUR]
oof = preds[preds["split"] == "train_oof"]
test = preds[preds["split"] == "test"]


# --- Sidebar: decision policy --------------------------------------------------
with st.sidebar:
    st.header("Decision policy")
    st.caption("How costly is each kind of mistake? The threshold is chosen to minimise "
               "the total cost, using out-of-fold predictions on the training data.")
    cost_missed = st.slider("Cost of missing a defaulter", 1.0, 20.0,
                            float(config.COST_MISSED_DEFAULT), 0.5)
    cost_alarm = st.slider("Cost of a false alarm", 0.5, 5.0, float(config.COST_FALSE_ALARM), 0.5)
    costs = {"cost_missed": cost_missed, "cost_false_alarm": cost_alarm}
    threshold = evaluation.best_threshold(oof[config.TARGET], oof["probability"], **costs)
    st.metric("Flag customers above", f"{threshold:.0%}")
    st.caption(f"Theory for perfectly calibrated probabilities: "
               f"{evaluation.theoretical_threshold(**costs):.0%}")

st.title("Credit Risk Scorer")
st.write("Predicts whether a credit card customer will miss next month's payment, explains "
         "why, and turns the prediction into a decision based on the cost of each mistake.")
st.info("Demo trained on public 2005 data from a Taiwanese bank "
        "([UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)). "
        "Amounts are in NT$. Not for real credit decisions.", icon=":material/info:")

score_tab, policy_tab, about_tab = st.tabs(["Score a customer", "Decision policy at work",
                                            "About the model"])

# --- Tab 1: score one customer -------------------------------------------------
with score_tab:
    choice = st.selectbox("Start from a real test-set customer, then edit any value",
                          list(examples.index), index=len(examples) - 1)
    row = examples.loc[choice]

    left, right = st.columns([3, 2], gap="large")
    with left:
        limit = st.number_input("Credit limit (NT$)", min_value=10_000, max_value=1_000_000,
                                value=int(row["LIMIT_BAL"]), step=10_000)
        months = pd.DataFrame({
            "Month (2005)": config.MONTHS,
            "Repayment status": [config.STATUS_LABELS[int(row[c])] for c in config.STATUS],
            "Statement balance": [int(row[c]) for c in config.BILLS],
            "Amount paid": [int(row[c]) for c in config.PAYMENTS],
        })
        edited = st.data_editor(
            months, hide_index=True, width="stretch", key=f"months_{choice}",
            disabled=["Month (2005)"],
            column_config={
                "Repayment status": st.column_config.SelectboxColumn(options=STATUS_OPTIONS,
                                                                     required=True),
                "Statement balance": st.column_config.NumberColumn(format="%d", step=1000),
                "Amount paid": st.column_config.NumberColumn(format="%d", step=1000, min_value=0),
            },
        )
        st.caption("Month order: September is the most recent month before the prediction.")

    customer = pd.DataFrame([{
        "LIMIT_BAL": limit,
        **{c: LABEL_TO_CODE[s] for c, s in zip(config.STATUS, edited["Repayment status"],
                                                strict=True)},
        **dict(zip(config.BILLS, edited["Statement balance"], strict=True)),
        **dict(zip(config.PAYMENTS, edited["Amount paid"], strict=True)),
    }])[config.BEHAVIOUR]

    reasons = explain.reason_codes(model, customer, reference)
    p = float(reasons["probability"].iloc[0])
    flagged = p >= threshold

    with right:
        a, b = st.columns(2)
        a.metric("Probability of default", f"{p:.0%}")
        b.metric("Risk band", evaluation.risk_band(p))
        if flagged:
            st.error(f"**Flag for action**: risk is above the {threshold:.0%} policy threshold.")
        else:
            st.success(f"**No action needed**: risk is below the {threshold:.0%} policy threshold.")

        st.subheader("Why this score?")
        st.caption("Each factor is compared with a typical customer who paid on time.")
        for _, r in reasons.head(3).iterrows():
            pts = r["contribution"] * 100
            if abs(pts) < 0.5:
                continue
            direction = "raises" if pts > 0 else "lowers"
            st.write(f"- **{r['factor']}** {direction} the risk by "
                     f"{abs(pts):.0f} percentage points")
        fig = plots.reasons(reasons, "Contribution of each factor")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

# --- Tab 2: the policy on the whole test set -----------------------------------
with policy_tab:
    summary = evaluation.decision_summary(test[config.TARGET], test["probability"], threshold,
                                          **costs)
    no_model = evaluation.no_model_cost(test[config.TARGET], **costs)
    best_alternative = min(no_model.values())
    st.write(f"Applying the current policy (flag above **{threshold:.0%}**) to the "
             f"**{len(test):,} held-out test customers**:")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers flagged", f"{summary['flagged_rate']:.0%}")
    c2.metric("Defaulters caught", f"{summary['recall']:.0%}")
    c3.metric("Flagged who default", f"{summary['precision']:.0%}")
    c4.metric("Cost vs best no-model option",
              f"{summary['cost_per_customer'] / best_alternative - 1:+.0%}")

    curve = evaluation.cost_curve(test[config.TARGET], test["probability"], **costs)
    fig = plots.cost_curve(curve, threshold, evaluation.theoretical_threshold(**costs), no_model)
    fig.axes[0].set_ylim(0, max(no_model.values()) * 1.15)
    fig.texts[-1].set_text(f"Cost units: missed default = {cost_missed:g}, "
                           f"false alarm = {cost_alarm:g}. Test set.")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.dataframe(pd.DataFrame({
        "": ["Actually defaulted", "Actually paid"],
        "Flagged": [summary["tp"], summary["fp"]],
        "Not flagged": [summary["fn"], summary["tn"]],
    }).set_index(""), width="content")
    st.caption("Try raising the cost of a missed defaulter: the threshold falls, more "
               "customers are flagged and more defaulters are caught.")

# --- Tab 3: model card ---------------------------------------------------------
with about_tab:
    s = meta["test_scores"]["Gradient boosting (tuned)"]
    st.subheader("Model")
    st.write(f"Histogram gradient boosting (scikit-learn), tuned by random search. Trained on "
             f"{meta['data']['train_rows']:,} customers and evaluated once on "
             f"{meta['data']['test_rows']:,} held-out customers.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test ROC-AUC", f"{s['roc_auc']:.3f}")
    m2.metric("Test PR-AUC", f"{s['pr_auc']:.3f}", help="Baseline (no skill): 0.22")
    m3.metric("Brier score", f"{s['brier']:.3f}", help="Lower is better. Baseline: 0.172")
    st.subheader("Inputs")
    st.write("Credit limit and six months of repayment status, balances and payments. "
             "**Sex, age, education and marital status are not used.** They are kept only "
             "to audit decisions for fairness; adding them improved accuracy by just 0.002 AUC.")
    st.subheader("Fairness audit (test set, at the default 5:1 policy)")
    def gap(series: pd.Series) -> str:
        return f"{series.max() - series.min():.0%}"

    fairness_rows = []
    for label in ["Sex", "Age", "Education", "Marital status"]:
        rep = pd.read_csv(config.TABLES_DIR / f"fairness_{label.lower().replace(' ', '_')}.csv",
                          index_col=0)
        fairness_rows.append({
            "Attribute": label,
            "Recall gap": gap(rep["recall"]),
            "False positive rate gap": gap(rep["false_positive_rate"]),
            "Largest calibration error": f"{rep['calibration_gap'].abs().max():.1%}",
        })
    st.dataframe(pd.DataFrame(fairness_rows).set_index("Attribute"))
    st.caption("Well calibrated for every group. The false positive rate gap by education "
               "follows real differences in default rates between groups.")
    st.subheader("Limitations")
    st.write("- Data from one bank in 2005; behaviour and economic conditions differ today.\n"
             "- The costs of each mistake are assumptions you can change in the sidebar.\n"
             "- Predicts one month ahead only.\n"
             "- Explanations replace one group of inputs at a time, so they cannot capture "
             "every interaction between factors.")
    st.caption("Source code: github.com/siddharth0444/credit-risk-scorer")
