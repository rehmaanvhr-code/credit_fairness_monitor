"""Scores every monthly batch and tracks fairness metrics over time.

For each sensitive attribute the script computes approval rate per group,
the disparate impact ratio (lowest group approval rate over the highest),
the equalized odds difference (largest gap in true positive rate or false
positive rate between groups) and AUC per group. Breaches of the thresholds
in config.json are written to outputs/alerts.md.
"""

import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from train import FEATURES


def group_rates(df, attr, pred_col="approved"):
    rates = df.groupby(attr)[pred_col].mean()
    return rates


def disparate_impact(df, attr):
    rates = group_rates(df, attr)
    return float(rates.min() / rates.max())


def equalized_odds_diff(df, attr):
    tprs = []
    fprs = []
    for _, g in df.groupby(attr):
        pos = g[g["repaid"] == 1]
        neg = g[g["repaid"] == 0]
        tprs.append(pos["approved"].mean() if len(pos) else 0.0)
        fprs.append(neg["approved"].mean() if len(neg) else 0.0)
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


def auc_by_group(df, attr):
    out = {}
    for name, g in df.groupby(attr):
        if g["repaid"].nunique() < 2:
            out[name] = float("nan")
        else:
            out[name] = float(roc_auc_score(g["repaid"], g["score"]))
    return out


def score_month(model, month_df, config):
    month_df = month_df.copy()
    month_df["score"] = model.predict_proba(month_df[FEATURES])[:, 1]
    month_df["approved"] = (month_df["score"] >= config["approval_threshold"]).astype(int)
    rows = []
    for attr in config["sensitive_attributes"]:
        aucs = auc_by_group(month_df, attr)
        rates = group_rates(month_df, attr)
        row = {
            "month": int(month_df["month"].iloc[0]),
            "attribute": attr,
            "overall_approval_rate": float(month_df["approved"].mean()),
            "disparate_impact": disparate_impact(month_df, attr),
            "equalized_odds_diff": equalized_odds_diff(month_df, attr),
            "auc_gap": float(max(aucs.values()) - min(aucs.values())),
        }
        for name, r in rates.items():
            row[f"approval_rate_{name}"] = float(r)
        for name, a in aucs.items():
            row[f"auc_{name}"] = a
        rows.append(row)
    return rows


def check_thresholds(history, config):
    t = config["thresholds"]
    alerts = []
    for _, r in history.iterrows():
        if r["disparate_impact"] < t["disparate_impact_min"]:
            alerts.append((r["month"], r["attribute"], "disparate_impact", r["disparate_impact"], t["disparate_impact_min"]))
        if r["equalized_odds_diff"] > t["equalized_odds_diff_max"]:
            alerts.append((r["month"], r["attribute"], "equalized_odds_diff", r["equalized_odds_diff"], t["equalized_odds_diff_max"]))
        if r["auc_gap"] > t["auc_gap_max"]:
            alerts.append((r["month"], r["attribute"], "auc_gap", r["auc_gap"], t["auc_gap_max"]))
    return alerts


def write_alerts(alerts, path):
    lines = ["# Fairness monitoring alerts", ""]
    if not alerts:
        lines.append("No thresholds breached.")
    else:
        lines.append("| month | attribute | metric | value | threshold |")
        lines.append("|---|---|---|---|---|")
        for month, attr, metric, value, thr in alerts:
            lines.append(f"| {int(month)} | {attr} | {metric} | {value:.3f} | {thr} |")
        first = min(a[0] for a in alerts)
        lines.append("")
        lines.append(f"First breach in month {int(first)}. {len(alerts)} breaches in total.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_history(history, config, path):
    t = config["thresholds"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for attr, g in history.groupby("attribute"):
        axes[0].plot(g["month"], g["disparate_impact"], marker="o", label=attr)
        axes[1].plot(g["month"], g["equalized_odds_diff"], marker="o", label=attr)
        axes[2].plot(g["month"], g["auc_gap"], marker="o", label=attr)
    axes[0].axhline(t["disparate_impact_min"], color="red", linestyle=":")
    axes[1].axhline(t["equalized_odds_diff_max"], color="red", linestyle=":")
    axes[2].axhline(t["auc_gap_max"], color="red", linestyle=":")
    axes[0].set_title("Disparate impact ratio")
    axes[1].set_title("Equalized odds difference")
    axes[2].set_title("AUC gap between groups")
    for ax in axes:
        ax.set_xlabel("Month")
        ax.legend()
    axes[0].set_ylabel("Ratio")
    axes[1].set_ylabel("Difference")
    axes[2].set_ylabel("Difference")
    fig.suptitle("Fairness metrics by month (dotted line is the alert threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(df, model, config):
    rows = []
    for month, month_df in df.groupby("month"):
        rows.extend(score_month(model, month_df, config))
    history = pd.DataFrame(rows)
    alerts = check_thresholds(history, config)
    return history, alerts


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    df = pd.read_csv("data/applications.csv")
    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)
    history, alerts = run(df, model, config)
    history.to_csv("outputs/metrics_history.csv", index=False)
    write_alerts(alerts, "outputs/alerts.md")
    plot_history(history, config, "outputs/fairness_over_time.png")
    print(history[["month", "attribute", "disparate_impact", "equalized_odds_diff", "auc_gap"]].to_string(index=False))
    print(f"\n{len(alerts)} alerts written to outputs/alerts.md")
