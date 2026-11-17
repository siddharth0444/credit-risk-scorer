"""Charts for the notebooks and README. Each function draws one figure and can save it."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config

INK, MUTED, GRID, LIGHT = "#1F2937", "#6B7280", "#E5E7EB", "#CBD5E1"
BLUE, ORANGE, RED, GREEN = "#2563EB", "#EA580C", "#DC2626", "#16A34A"
SERIES = [BLUE, ORANGE, MUTED, GREEN]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "legend.frameon": False,
    })


def _finish(fig, path: Path | None, note: str | None = None):
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8.5, color=MUTED)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def _pct(ax, axis="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def default_by_status(df: pd.DataFrame, path: Path | None = None):
    """Default rate by the latest month's repayment status."""
    g = df.groupby("PAY_1")[config.TARGET].agg(["mean", "size"])
    g = g[g["size"] >= 50]
    labels = [config.STATUS_LABELS[int(k)] for k in g.index]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [GREEN if k < 1 else (ORANGE if k == 1 else RED) for k in g.index]
    ax.bar(range(len(g)), g["mean"], color=colors)
    for i, (rate, n) in enumerate(zip(g["mean"], g["size"], strict=True)):
        ax.annotate(f"{rate:.0%}\nn={n:,}", (i, rate), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axhline(df[config.TARGET].mean(), color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(g)), labels, rotation=20, ha="right")
    _pct(ax)
    ax.set_ylim(0, g["mean"].max() * 1.25)
    ax.set(title="Being 2+ months late in September means a 69%+ chance of default",
           ylabel="Default rate next month")
    return _finish(fig, path, "Dashed line: overall default rate (22%). Statuses with fewer "
                              "than 50 customers not shown.")


def model_comparison(cv: pd.DataFrame, path: Path | None = None):
    """Cross-validated ROC-AUC and PR-AUC per model, with fold-to-fold spread."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(cv))
    for ax, metric, title, floor in [(axes[0], "roc_auc", "ROC-AUC (ranking)", 0.5),
                                     (axes[1], "pr_auc", "PR-AUC (finding defaulters)",
                                      cv.loc[cv.index[0], "pr_auc"])]:
        colors = [LIGHT] + [MUTED] * (len(cv) - 2) + [BLUE]
        ax.barh(y, cv[metric], xerr=cv[f"{metric}_std"], color=colors, capsize=3)
        for yi, v, sd in zip(y, cv[metric], cv[f"{metric}_std"], strict=True):
            ax.annotate(f"{v:.3f}", (v + sd, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.axvline(floor, color=MUTED, lw=0.8, ls="--")
        ax.set(title=title, xlim=(floor - 0.05, cv[metric].max() + 0.06))
    axes[0].set_yticks(y, cv.index)
    axes[0].invert_yaxis()
    return _finish(fig, path, "5-fold cross-validation on the training set. Error bars: "
                              "standard deviation across folds. Dashed: baseline.")


def roc_pr(y: np.ndarray, probs: dict[str, np.ndarray], path: Path | None = None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    for (name, p), color in zip(probs.items(), SERIES, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a.plot(fpr, tpr, color=color, lw=2, label=name)
        b.plot(rec, prec, color=color, lw=2, label=name)
    a.plot([0, 1], [0, 1], color=LIGHT, ls="--")
    b.axhline(np.mean(y), color=LIGHT, ls="--")
    a.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    b.set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision")
    a.legend(loc="lower right")
    return _finish(fig, path)


def calibration(table: pd.DataFrame, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lim = max(table["predicted"].max(), table["observed"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=LIGHT, ls="--", label="Perfect calibration")
    ax.plot(table["predicted"], table["observed"], "o-", color=BLUE, lw=2, label="Model")
    _pct(ax, "x")
    _pct(ax, "y")
    ax.set(title="Predicted risk matches reality", xlabel="Mean predicted probability",
           ylabel="Observed default rate", xlim=(0, lim), ylim=(0, lim))
    ax.legend(loc="upper left")
    return _finish(fig, path, "Test set, 10 equal-sized groups of customers.")


def cost_curve(curve: pd.DataFrame, chosen: float, theoretical: float, no_model: dict,
               path: Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(curve["threshold"], curve["cost_per_customer"], color=BLUE, lw=2.2,
            label="Model at each threshold")
    ax.axhline(no_model["flag_everyone"], color=ORANGE, ls="--", lw=1.2, label="Flag everyone")
    ax.axhline(no_model["flag_nobody"], color=RED, ls="--", lw=1.2, label="Flag nobody")
    ax.axvline(theoretical, color=MUTED, ls=":", lw=1.2,
               label=f"Theoretical optimum ({theoretical:.2f})")
    best = curve.loc[(curve["threshold"] - chosen).abs().idxmin()]
    ax.scatter([chosen], [best["cost_per_customer"]], color=INK, zorder=5, s=40)
    ax.annotate(f"chosen threshold {chosen:.2f}\ncost {best['cost_per_customer']:.2f}",
                (chosen, best["cost_per_customer"]), xytext=(14, -26), textcoords="offset points",
                fontsize=9, color=INK)
    ax.set(title="Choosing the threshold by business cost, not accuracy",
           xlabel="Flag customers with predicted probability above", ylabel="Cost per customer")
    ax.set_ylim(0, no_model["flag_nobody"] * 1.15)
    ax.legend(loc="upper right", fontsize=9)
    return _finish(fig, path, f"Cost units: missed default = {config.COST_MISSED_DEFAULT:g}, "
                              f"false alarm = {config.COST_FALSE_ALARM:g}.")


def score_distribution(y: np.ndarray, p: np.ndarray, threshold: float, path: Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(p[np.asarray(y) == 0], bins=bins, color=BLUE, alpha=0.55, label="Paid", density=True)
    ax.hist(p[np.asarray(y) == 1], bins=bins, color=RED, alpha=0.55, label="Defaulted",
            density=True)
    ax.axvline(threshold, color=INK, ls="--", lw=1)
    ax.set(title="Predicted risk for customers who paid vs defaulted",
           xlabel="Predicted probability of default", ylabel="Density")
    ax.legend()
    return _finish(fig, path)


def importance(table: pd.DataFrame, path: Path | None = None):
    t = table.sort_values("auc_drop")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(t["group"], t["auc_drop"], xerr=t["auc_drop_std"], color=BLUE, capsize=3)
    ax.set(title="What the model relies on",
           xlabel="Drop in ROC-AUC when this information is shuffled")
    return _finish(fig, path, "Grouped permutation importance on the test set (10 repeats).")


def reasons(table: pd.DataFrame, title: str, path: Path | None = None):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    colors = [RED if c > 0 else GREEN for c in t["contribution"]]
    ax.barh(t["factor"], t["contribution"] * 100, color=colors)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(title=title, xlabel="Change in default probability vs a typical good payer "
                               "(percentage points)")
    return _finish(fig, path)


def fairness(reports: dict[str, pd.DataFrame], path: Path | None = None):
    """Recall and false positive rate by group, one row of panels per attribute."""
    fig, axes = plt.subplots(len(reports), 2, figsize=(10, 2.0 * len(reports) + 0.6),
                             squeeze=False)
    for row, (attr, rep) in enumerate(reports.items()):
        for col, (metric, label, color) in enumerate([
                ("recall", "Defaulters caught (recall)", BLUE),
                ("false_positive_rate", "Good payers flagged (FPR)", ORANGE)]):
            ax = axes[row, col]
            ax.barh(rep.index.astype(str), rep[metric], color=color)
            for i, v in enumerate(rep[metric]):
                ax.annotate(f"{v:.0%}", (v, i), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8.5)
            ax.set_xlim(0, 1)
            _pct(ax, "x")
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(attr, fontsize=10, fontweight="bold")
    fig.suptitle("Fairness audit: outcomes by group at the chosen threshold", x=0.01,
                 ha="left", fontsize=13, fontweight="bold", color=INK, y=1.01)
    fig.tight_layout()
    return _finish(fig, path, "Test set. Groups with fewer than 100 customers not shown. "
                              "The model never sees these attributes.")
