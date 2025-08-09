#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_model_recommender.py
A professional script that:
  • asks for a CSV filename,
  • infers task type (classification vs regression),
  • benchmarks linear vs non-linear models with cross-validation,
  • explains *why* the data appears linear or non-linear,
  • recommends the best ML/DL model family,
  • writes a human-readable report and a compact JSON with suggestions.

Console messages and report are in ENGLISH as requested.
"""

from __future__ import annotations
import sys, os, json, warnings, textwrap
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     cross_validate)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             r2_score, mean_squared_error, make_scorer)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(2025)


# ----------------------------- Utilities ----------------------------- #

def ask_filename() -> str:
    print("Enter CSV filename (path): ", end="", flush=True)
    fname = input().strip().strip('"').strip("'")
    if not os.path.isfile(fname):
        print(f"[ERROR] File not found: {fname}")
        sys.exit(1)
    return fname


def detect_target(df: pd.DataFrame) -> str:
    """
    Heuristic:
      - if a column is named like 'target' or 'label', use it.
      - otherwise, assume the LAST column is the target.
    """
    candidates = [c for c in df.columns if c.lower() in {"target", "label", "y"}]
    if candidates:
        return candidates[0]
    return df.columns[-1]


def is_classification(y: pd.Series) -> bool:
    """Classification if y is non-numeric or a small set of unique values."""
    if not pd.api.types.is_numeric_dtype(y):
        return True
    nunique = y.nunique(dropna=True)
    n = len(y)
    # If the number of unique values is small relative to n, treat as classification
    return nunique <= max(10, int(0.02 * n))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    try:
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return None


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    try:
        r, _ = spearmanr(x, y)
        return float(r)
    except Exception:
        return None


def split_columns(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target])
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale: bool = True) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        *([("scale", StandardScaler())] if scale else [])
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )
    return pre


def pick_cv(y: pd.Series, classification: bool):
    if classification:
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    return KFold(n_splits=5, shuffle=True, random_state=2025)


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    is_linear_family: bool


# --------------------------- Model Zoo --------------------------- #

def model_zoo_classification(pre: ColumnTransformer) -> List[ModelSpec]:
    models = []

    # Linear family
    models.append(ModelSpec(
        "LogisticRegression(l2, saga)",
        Pipeline([("pre", pre),
                  ("clf", LogisticRegression(penalty="l2", solver="saga", max_iter=2000, n_jobs=-1))]),
        True
    ))
    models.append(ModelSpec(
        "RidgeClassifier",
        Pipeline([("pre", pre), ("clf", RidgeClassifier())]),
        True
    ))
    models.append(ModelSpec(
        "LogReg + Polynomial(deg=2)",
        Pipeline([
            ("pre", pre),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale2", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(penalty="l2", solver="saga", max_iter=2000, n_jobs=-1))
        ]),
        False  # polynomial -> non-linear boundary in original space
    ))

    # Non-linear
    models.append(ModelSpec(
        "RandomForestClassifier",
        Pipeline([("pre", pre),
                  ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=2025))]),
        False
    ))
    models.append(ModelSpec(
        "GradientBoostingClassifier",
        Pipeline([("pre", pre),
                  ("clf", GradientBoostingClassifier())]),
        False
    ))
    models.append(ModelSpec(
        "RBF-SVM (SVC)",
        Pipeline([("pre", pre),
                  ("clf", SVC(kernel="rbf", probability=True))]),
        False
    ))
    models.append(ModelSpec(
        "MLPClassifier",
        Pipeline([("pre", pre),
                  ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                        max_iter=300, random_state=2025))]),
        False
    ))
    return models


def model_zoo_regression(pre: ColumnTransformer) -> List[ModelSpec]:
    models = []

    # Linear family
    models.append(ModelSpec(
        "LinearRegression",
        Pipeline([("pre", pre), ("reg", LinearRegression())]),
        True
    ))
    models.append(ModelSpec(
        "Ridge(alpha=1.0)",
        Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))]),
        True
    ))
    models.append(ModelSpec(
        "Ridge + Polynomial(deg=2)",
        Pipeline([
            ("pre", pre),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale2", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=1.0))
        ]),
        False  # polynomial basis -> non-linear in original space
    ))

    # Non-linear
    models.append(ModelSpec(
        "RandomForestRegressor",
        Pipeline([("pre", pre),
                  ("reg", RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=2025))]),
        False
    ))
    models.append(ModelSpec(
        "GradientBoostingRegressor",
        Pipeline([("pre", pre),
                  ("reg", GradientBoostingRegressor(random_state=2025))]),
        False
    ))
    models.append(ModelSpec(
        "RBF-SVR",
        Pipeline([("pre", pre),
                  ("reg", SVR(kernel="rbf"))]),
        False
    ))
    models.append(ModelSpec(
        "MLPRegressor",
        Pipeline([("pre", pre),
                  ("reg", MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                                       max_iter=400, random_state=2025))]),
        False
    ))
    return models


# ---------------------- Scoring & Diagnostics ---------------------- #

def classification_scoring(y: pd.Series) -> Dict[str, Any]:
    # Use balanced accuracy if imbalanced? Keep it simple but robust
    scorers = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }
    # Add ROC AUC if binary
    if y.nunique() == 2:
        scorers["roc_auc"] = "roc_auc"
    return scorers


def regression_scoring() -> Dict[str, Any]:
    return {
        "rmse": make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False),
        "r2": "r2"
    }


def summarize_scores(cvres: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {}
    for k, v in cvres.items():
        if k.startswith("test_"):
            key = k.replace("test_", "")
            out[key] = float(np.mean(v))
    return out


# ---------------------- Linearity Heuristics ---------------------- #

def linearity_evidence(X: pd.DataFrame, y: pd.Series, classification: bool,
                       num_cols: List[str], cat_cols: List[str]) -> Dict[str, Any]:
    """
    Provide multiple small tests that *tend* to indicate nonlinearity:
      1) Spearman vs Pearson average absolute correlation over numeric features.
      2) Mutual Information average over top features.
      3) Performance gap: best non-linear vs best linear (filled later by caller).
    """
    evid = {}

    # 1) Spearman vs Pearson across numeric features
    pearsons, spearmans = [], []
    for c in num_cols:
        x = X[c].values
        if classification:
            # If y categorical, encode to ranks for correlation sense
            y_enc = pd.factorize(y)[0]
        else:
            y_enc = y.values
        rp = safe_pearson(x, y_enc)
        rs = safe_spearman(x, y_enc)
        if rp is not None:
            pearsons.append(abs(rp))
        if rs is not None:
            spearmans.append(abs(rs))

    evid["mean_abs_pearson"] = float(np.mean(pearsons)) if pearsons else None
    evid["mean_abs_spearman"] = float(np.mean(spearmans)) if spearmans else None
    evid["spearman_minus_pearson"] = (None if (evid["mean_abs_pearson"] is None or evid["mean_abs_spearman"] is None)
                                      else float(evid["mean_abs_spearman"] - evid["mean_abs_pearson"]))

    # 2) Mutual information
    try:
        if classification:
            mi = mutual_info_classif(X[num_cols].fillna(X[num_cols].median()), pd.factorize(y)[0], random_state=2025)
        else:
            mi = mutual_info_regression(X[num_cols].fillna(X[num_cols].median()), y.values, random_state=2025)
        evid["mean_mutual_information"] = float(np.mean(mi)) if len(mi) else None
        evid["top5_mutual_information_mean"] = float(np.mean(sorted(mi, reverse=True)[:min(5, len(mi))])) if len(mi) else None
    except Exception:
        evid["mean_mutual_information"] = None
        evid["top5_mutual_information_mean"] = None

    # Placeholders for perf gap; caller fills
    evid["cv_linear_score"] = None
    evid["cv_nonlinear_score"] = None
    evid["nonlinear_minus_linear"] = None

    return evid


# ------------------------------ Main ------------------------------ #

def main():
    print("=== Auto Model Recommender (English outputs) ===")
    csv_path = ask_filename()
    print(f"[INFO] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    if df.empty or df.shape[1] < 2:
        print("[ERROR] CSV must have at least 2 columns (features + target).")
        sys.exit(1)

    target = detect_target(df)
    print(f"[INFO] Target column inferred as: '{target}'")
    X = df.drop(columns=[target])
    y = df[target]

    # Basic cleaning: drop columns that are entirely NA or constant
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[WARN] Dropping constant columns: {const_cols}")
        X = X.drop(columns=const_cols)

    num_cols, cat_cols = split_columns(pd.concat([X, y], axis=1), target)
    print(f"[INFO] Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    task_is_clf = is_classification(y)
    print(f"[INFO] Inferred task: {'Classification' if task_is_clf else 'Regression'}")

    pre_scaled = build_preprocessor(num_cols, cat_cols, scale=True)
    cv = pick_cv(y, task_is_clf)

    # Linearity evidence (to be completed later with CV gap)
    evid = linearity_evidence(X, y, task_is_clf, num_cols, cat_cols)

    # Models & scoring
    if task_is_clf:
        zoo = model_zoo_classification(pre_scaled)
        scorers = classification_scoring(y)
        primary_metric = "roc_auc" if ("roc_auc" in scorers and y.nunique() == 2) else "f1_macro"
        print(f"[INFO] Primary metric: {primary_metric}")
    else:
        zoo = model_zoo_regression(pre_scaled)
        scorers = regression_scoring()
        primary_metric = "r2"  # we will also track RMSE (negative)
        print(f"[INFO] Primary metric: {primary_metric}")

    # Evaluate all models
    results: List[Dict[str, Any]] = []
    best_linear = None
    best_nonlinear = None

    print("[INFO] Running 5-fold cross-validation on candidate models...")
    for spec in zoo:
        try:
            cvres = cross_validate(spec.pipeline, X, y, cv=cv, scoring=scorers, n_jobs=-1, error_score="raise")
            summary = summarize_scores(cvres)
            entry = {"model": spec.name, "is_linear": spec.is_linear_family, **summary}
            results.append(entry)
            pscore = entry.get(primary_metric)
            if pscore is not None:
                if spec.is_linear_family:
                    if (best_linear is None) or (pscore > best_linear.get(primary_metric, -np.inf)):
                        best_linear = entry
                else:
                    if (best_nonlinear is None) or (pscore > best_nonlinear.get(primary_metric, -np.inf)):
                        best_nonlinear = entry
            print(f"  - {spec.name:<28} => {entry}")
        except Exception as e:
            print(f"[WARN] Skipping {spec.name} due to error: {e}")

    if not results:
        print("[ERROR] No model could be evaluated. Check your data.")
        sys.exit(1)

    # Determine overall best
    overall_best = max(results, key=lambda d: d.get(primary_metric, -np.inf))
    print(f"[INFO] Overall best by {primary_metric}: {overall_best['model']}")

    # Fill linearity evidence using perf gap
    evid["cv_linear_score"] = (None if best_linear is None else float(best_linear.get(primary_metric)))
    evid["cv_nonlinear_score"] = (None if best_nonlinear is None else float(best_nonlinear.get(primary_metric)))
    if evid["cv_linear_score"] is not None and evid["cv_nonlinear_score"] is not None:
        evid["nonlinear_minus_linear"] = float(evid["cv_nonlinear_score"] - evid["cv_linear_score"])

    # Decision logic for linear vs non-linear nature
    # thresholds chosen to be conservative but informative
    reasons: List[str] = []
    linearity_label = "Unclear"

    # reason A: Spearman >> Pearson suggests monotonic but non-linear relations
    spearman_minus_pearson = evid.get("spearman_minus_pearson")
    if spearman_minus_pearson is not None and spearman_minus_pearson > 0.10:
        reasons.append(
            f"Mean |Spearman| exceeds |Pearson| by {spearman_minus_pearson:.3f}, suggesting monotonic but non-linear associations."
        )

    # reason B: MI > small threshold
    top5_mi = evid.get("top5_mutual_information_mean")
    if top5_mi is not None and top5_mi > 0.01:
        reasons.append(
            f"Top-5 mean mutual information = {top5_mi:.3f}, indicating non-linear dependency captured by MI."
        )

    # reason C: CV performance gap
    perf_gap = evid.get("nonlinear_minus_linear")
    if perf_gap is not None:
        if perf_gap > (0.03 if task_is_clf else 0.02):  # 3% f1/auc or 0.02 R2 improvement
            reasons.append(f"Non-linear models outperform linear ones by Δ={perf_gap:.3f} on {primary_metric}.")
        elif perf_gap < -(0.01 if task_is_clf else 0.01):
            reasons.append(f"Linear models slightly outperform non-linear ones by Δ={-perf_gap:.3f} on {primary_metric}.")

    # Decide label
    if perf_gap is not None and perf_gap > (0.03 if task_is_clf else 0.02):
        linearity_label = "Predominantly Non-Linear"
    elif perf_gap is not None and perf_gap < -(0.01 if task_is_clf else 0.01):
        linearity_label = "Predominantly Linear"
    else:
        # fallback: use correlation signal
        if spearman_minus_pearson is not None and spearman_minus_pearson > 0.10:
            linearity_label = "Likely Non-Linear"
        else:
            linearity_label = "Likely Linear or Mixed"

    # Recommendation
    if linearity_label.startswith("Predominantly Linear"):
        recommended = "Linear family (Ridge / Logistic Regression)"
    elif linearity_label.startswith("Likely Linear"):
        recommended = "Start with Linear (Ridge / Logistic) + simple interactions; consider GBM if underfitting"
    elif linearity_label.startswith("Likely Non-Linear"):
        recommended = "Tree-based ensemble (Gradient Boosting / Random Forest) or RBF-kernel SVM"
    else:
        # Predominantly Non-Linear
        recommended = "Gradient Boosting or Random Forest; consider MLP if data is large and well-scaled"

    # If the overall best is MLP, surface a DL-ish recommendation explicitly
    if "MLP" in overall_best["model"]:
        recommended += " (MLP performed best in CV; ensure enough data and tune regularization)."

    # Build report
    lines = []
    lines.append("=== Auto Model Recommender Report ===")
    lines.append(f"File: {os.path.abspath(csv_path)}")
    lines.append(f"Rows: {len(df):,} | Features: {X.shape[1]} (numeric={len(num_cols)}, categorical={len(cat_cols)})")
    lines.append(f"Task: {'Classification' if task_is_clf else 'Regression'}")
    lines.append("")
    lines.append("Linearity Assessment:")
    lines.append(f"  • Label: {linearity_label}")
    if evid.get("mean_abs_pearson") is not None and evid.get("mean_abs_spearman") is not None:
        lines.append(f"  • Mean |Pearson| = {evid['mean_abs_pearson']:.3f}")
        lines.append(f"  • Mean |Spearman| = {evid['mean_abs_spearman']:.3f}")
        lines.append(f"  • Spearman − Pearson = {spearman_minus_pearson:.3f}")
    if top5_mi is not None:
        lines.append(f"  • Top-5 MI mean = {top5_mi:.3f}")
    if evid["cv_linear_score"] is not None and evid["cv_nonlinear_score"] is not None:
        lines.append(f"  • CV {primary_metric} (best linear)    = {evid['cv_linear_score']:.4f}")
        lines.append(f"  • CV {primary_metric} (best non-linear) = {evid['cv_nonlinear_score']:.4f}")
        lines.append(f"  • Non-linear − Linear gap = {evid['nonlinear_minus_linear']:.4f}")
    if reasons:
        lines.append("  • Reasons:")
        for r in reasons:
            lines.append(f"    - {r}")
    lines.append("")
    lines.append(f"Overall Best (by {primary_metric}): {overall_best['model']}")
    # Include a compact summary of scores
    lines.append("Cross-Validation Summary (mean scores):")
    for r in sorted(results, key=lambda d: d.get(primary_metric, -np.inf), reverse=True):
        msg = f"  - {r['model']:<28} | "
        parts = []
        for k, v in r.items():
            if k in {"model", "is_linear"}: 
                continue
            parts.append(f"{k}={v:.4f}")
        msg += ", ".join(parts)
        lines.append(msg)

    lines.append("")
    lines.append("Recommendation:")
    lines.append(f"  • {recommended}")
    lines.append("")
    lines.append("Next Steps:")
    lines.append("  • If underfitting: add domain features and try Gradient Boosting (tune learning_rate, n_estimators).")
    lines.append("  • If overfitting: strengthen regularization, reduce polynomial degree, or limit tree depth.")
    lines.append("  • For MLP: scale inputs, apply early stopping, tune hidden sizes and alpha (L2).")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # Save artifacts
    report_path = "model_recommendation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[INFO] Saved report: {report_path}")

    suggestions = {
        "file": os.path.abspath(csv_path),
        "task": "classification" if task_is_clf else "regression",
        "linearity_label": linearity_label,
        "primary_metric": primary_metric,
        "overall_best": overall_best,
        "evidence": evid,
        "suggested_family": recommended
    }
    with open("model_recommendation_suggestions.json", "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)
    print("[INFO] Saved suggestions: model_recommendation_suggestions.json")

    print("\n[DONE] All outputs printed in ENGLISH. Thank you.\n")


if __name__ == "__main__":
    main()
