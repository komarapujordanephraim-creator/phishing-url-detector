"""Phishing website detector: compare three classifiers with group-aware validation.

Data: a public 10,000-row phishing/legitimate website feature table (see README for the source
and its limits). The file is downloaded on first run into ./data and is not stored in this repo.

Key design choice: rows from the same domain never appear in both train and test
(GroupShuffleSplit / GroupKFold on the Domain column). Without this, a model can memorise
domains and the score looks far better than it really is.

Run:  python train.py
Outputs go to ./outputs (metrics.json, confusion_matrix.png, roc_curve.png,
feature_importance.png, model_comparison.png).
"""
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
HERE = Path(__file__).parent
OUT = HERE / "outputs"
DATA = HERE / "data" / "urldata.csv"
DATA_URL = ("https://raw.githubusercontent.com/shreyagopal/"
            "Phishing-Website-Detection-by-Machine-Learning-Techniques/master/DataFiles/5.urldata.csv")

# Label: 1 = phishing, 0 = legitimate.
# The first eight features are read from the URL text itself. The rest need a live lookup
# (DNS, traffic rank, domain age) or the page's HTML/JavaScript.
URL_FEATURES = ["Have_IP", "Have_At", "URL_Length", "URL_Depth", "Redirection",
                "https_Domain", "TinyURL", "Prefix/Suffix"]
LOOKUP_FEATURES = ["DNS_Record", "Web_Traffic", "Domain_Age", "Domain_End",
                   "iFrame", "Mouse_Over", "Right_Click", "Web_Forwards"]
ALL_FEATURES = URL_FEATURES + LOOKUP_FEATURES


def load_data(path=DATA):
    """Download the table once, then read it. Returns a DataFrame."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def clean(df):
    """Drop rows whose domain appears with both labels (ambiguous ground truth).

    Exact duplicate rows are kept on purpose: removing them would collapse the legitimate
    class (many legitimate pages share identical feature values) and make the data 86%
    phishing. Domain-grouped splitting already prevents train/test leakage.
    """
    df = df.reset_index(drop=True)
    labels_per_domain = df.groupby("Domain")["Label"].nunique()
    conflicting = labels_per_domain[labels_per_domain > 1].index
    return df[~df["Domain"].isin(conflicting)].reset_index(drop=True)


def group_split(df, test_size=0.25, seed=SEED):
    """Train/test split where no domain appears on both sides."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["Label"], groups=df["Domain"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


def evaluate(model, tr, te, features):
    cv = GroupKFold(n_splits=5)
    cv_auc = cross_val_score(model, tr[features], tr["Label"], cv=cv,
                             groups=tr["Domain"], scoring="roc_auc")
    model.fit(tr[features], tr["Label"])
    pred = model.predict(te[features])
    proba = model.predict_proba(te[features])[:, 1]
    y = te["Label"]
    return {
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_roc_auc_std": round(float(cv_auc.std()), 4),
        "test_accuracy": round(float(accuracy_score(y, pred)), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "test_specificity_legit_correct": round(float(((pred == 0) & (y == 0)).sum() / (y == 0).sum()), 4),
        "test_precision": round(float(precision_score(y, pred)), 4),
        "test_recall": round(float(recall_score(y, pred)), 4),
        "test_f1": round(float(f1_score(y, pred)), 4),
        "test_roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }


def main():
    OUT.mkdir(exist_ok=True)
    raw = load_data()
    df = clean(raw)
    tr, te = group_split(df)
    assert not set(tr["Domain"]) & set(te["Domain"]), "domain leak between train and test"

    results, fitted = {}, {}
    for name, model in build_models().items():
        results[name] = evaluate(model, tr, te, ALL_FEATURES)
        fitted[name] = model
        print(f"{name:20s} AUC {results[name]['test_roc_auc']:.3f}  bal.acc {results[name]['test_balanced_accuracy']:.3f}  "
              f"legit-correct {results[name]['test_specificity_legit_correct']:.3f}")

    best = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    # Ablation: how much do the lookup-based features add beyond the URL text alone?
    url_only = evaluate(build_models()[best], tr, te, URL_FEATURES)

    meta = {
        "source": "Public phishing-website feature table (see README for link and limits)",
        "rows_raw": int(len(raw)), "rows_after_cleaning": int(len(df)),
        "unique_domains": int(df["Domain"].nunique()),
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "test_phishing_share": round(float(te["Label"].mean()), 3),
        "majority_class_baseline_accuracy": round(float(max(te["Label"].mean(), 1 - te["Label"].mean())), 4),
        "test_legit_domains": int(te.loc[te["Label"] == 0, "Domain"].nunique()),
        "train_legit_domains": int(tr.loc[tr["Label"] == 0, "Domain"].nunique()),
        "validation": "GroupShuffleSplit + GroupKFold(5) on Domain (no domain in both train and test)",
        "best_model_by_cv_auc": best,
        "url_text_features_only_for_best_model": url_only,
        "results": results,
    }
    (OUT / "metrics.json").write_text(json.dumps(meta, indent=2))

    model = fitted[best]
    pred = model.predict(te[ALL_FEATURES])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(te["Label"], pred, display_labels=["Legitimate", "Phishing"],
                                            cmap="Greys", ax=ax, colorbar=False)
    ax.set_title(f"{best}: confusion matrix (test)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, m in fitted.items():
        RocCurveDisplay.from_estimator(m, te[ALL_FEATURES], te["Label"], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC curves (test, unseen domains)")
    fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=160); plt.close(fig)

    imp = permutation_importance(model, te[ALL_FEATURES], te["Label"], scoring="roc_auc",
                                 n_repeats=10, random_state=SEED)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(ALL_FEATURES)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#444")
    ax.set_xlabel("Drop in test ROC AUC when the feature is shuffled")
    ax.set_title(f"{best}: permutation importance")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = list(results)
    ax.bar(names, [results[n]["test_roc_auc"] for n in names], color="#444", label="All 16 features")
    ax.axhline(url_only["test_roc_auc"], color="#b8892f", ls="--",
               label=f"{best}, URL-text features only")
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Test ROC AUC"); ax.legend(loc="lower right")
    ax.set_title("Model comparison on unseen domains")
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=160); plt.close(fig)
    print("best:", best, "| URL-only AUC:", url_only["test_roc_auc"])


if __name__ == "__main__":
    main()
