#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from threshold_finder import ThresholdFinder

# ============================================================
# 1️⃣ Logging setup
# ============================================================
log_filename = "ensemble.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

def stage(msg: str):
    sep = "=" * 80
    log.info(f"\n{sep}\n🟢 STAGE: {msg}\n{sep}")

# ============================================================
# 2️⃣ Load and split data
# ============================================================
stage("Loading and splitting dataset")

df = pd.read_csv("prepared_train_data.csv")
if "target" not in df.columns:
    raise ValueError("❌ 'target' column missing in dataset")

X = df.drop("target", axis=1)
y = df["target"].astype(int)

# Detect a price column (optional for profitability)
price_col = None
for c in X.columns:
    if "close" in c.lower():
        price_col = c
        break
prices = X[price_col].values if price_col else np.arange(len(X))

X = X.select_dtypes(include=[np.number])
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

total_len = len(X)
live_size = 4000
threshold_size = int(total_len * 0.05)

X_train_full = X.iloc[:-(live_size + threshold_size)]
y_train_full = y.iloc[:-(live_size + threshold_size)]

X_thresh = X.iloc[-(live_size + threshold_size):-live_size]
y_thresh = y.iloc[-(live_size + threshold_size):-live_size]

X_live = X.tail(live_size)
y_live = y.tail(live_size)
price_live = prices[-live_size:] if len(prices) >= live_size else np.arange(live_size)

X_train, X_test, y_train, y_test = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

log.info(f"Dataset split:")
log.info(f"  • Train: {len(X_train)} rows")
log.info(f"  • Threshold: {len(X_thresh)} rows (~5%)")
log.info(f"  • Test: {len(X_test)} rows")
log.info(f"  • Live: {len(X_live)} rows")

# ============================================================
# 3️⃣ Define models and grids
# ============================================================
stage("Defining models and hyperparameter grids")

models_config = {
    "XGBoost": {
        "model": XGBClassifier(random_state=2025, n_jobs=-1, eval_metric="logloss"),
        "scaler": None,
        "param_grid": {
            "n_estimators": [200, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 1.5, 2.0],
        },
    },
    "HistGradientBoosting": {
        "model": HistGradientBoostingClassifier(random_state=2025),
        "scaler": None,
        "param_grid": {
            "max_iter": [200, 400],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6, 8],
            "l2_regularization": [0.0, 0.5, 1.0],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=5000, random_state=2025, solver="saga"),
        "scaler": StandardScaler(),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"]
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=2025, n_jobs=-1),
        "scaler": MinMaxScaler(),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [6, 8, 10],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced_subsample"],
        },
    },
}

# ============================================================
# 4️⃣ Training, Bias, Threshold, and Stability Analysis
# ============================================================
stage("Training models, checking bias & overfitting, and computing stability indices")

results = []
for name, cfg in models_config.items():
    try:
        model, scaler, params = cfg["model"], cfg["scaler"], cfg["param_grid"]
        log.info(f"\n🔹 Optimizing {name} parameters with GridSearch ...")

        if scaler is not None:
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_thresh_scaled = scaler.transform(X_thresh)
            X_live_scaled = scaler.transform(X_live)
        else:
            X_train_scaled, X_test_scaled, X_thresh_scaled, X_live_scaled = (
                X_train, X_test, X_thresh, X_live
            )

        grid = GridSearchCV(model, params, scoring="f1", cv=3, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        log.info(f"✅ {name}: Best Params = {best_params}")

        # Bias check
        y_pred_train = best_model.predict(X_train_scaled)
        buy_ratio = np.mean(y_pred_train == 1)
        sell_ratio = np.mean(y_pred_train == 0)
        bias_penalty = 0.0
        if buy_ratio > 0.9 or sell_ratio > 0.9:
            bias_penalty = 0.1
            log.warning(f"⚠️ {name} shows bias → BUY={buy_ratio:.2f}, SELL={sell_ratio:.2f}")
        else:
            log.info(f"✅ {name} output distribution OK → BUY={buy_ratio:.2f}, SELL={sell_ratio:.2f}")

        # Evaluate
        y_pred_test = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred_test) - bias_penalty
        bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Thresholds
        y_proba_thresh = best_model.predict_proba(X_thresh_scaled)[:, 1]
        tf = ThresholdFinder(steps=600, min_predictions_ratio=0.9)
        neg_thr, pos_thr, thr_acc, _, _ = tf.find_best_thresholds(y_proba_thresh, y_thresh.values)

        # Live
        y_proba_live = best_model.predict_proba(X_live_scaled)[:, 1]
        y_pred_live = np.full_like(y_live, -1)
        y_pred_live[y_proba_live <= neg_thr] = 0
        y_pred_live[y_proba_live >= pos_thr] = 1

        mask_live = y_pred_live != -1
        acc_live = accuracy_score(y_live[mask_live], y_pred_live[mask_live]) if mask_live.any() else 0
        bal_live = balanced_accuracy_score(y_live[mask_live], y_pred_live[mask_live]) if mask_live.any() else 0
        f1_live = f1_score(y_live[mask_live], y_pred_live[mask_live]) if mask_live.any() else 0

        # ======================================================
        # Stability & Drift metrics
        # ======================================================
        log.info(f"\n📈 Stability & Drift Analysis for {name}")

        try:
            mean_proba_train = np.mean(best_model.predict_proba(X_train_scaled)[:, 1])
            mean_proba_test = np.mean(best_model.predict_proba(X_test_scaled)[:, 1])
            mean_proba_thresh = np.mean(y_proba_thresh)
            mean_proba_live = np.mean(y_proba_live)
            drift_train_live = abs(mean_proba_live - mean_proba_train)
            drift_test_live = abs(mean_proba_live - mean_proba_test)
            std_proba_live = np.std(y_proba_live)
            stability_index = 1 - (drift_train_live + drift_test_live + std_proba_live)
            stability_index = max(0, min(stability_index, 1))
            log.info(f"Avg_Proba: train={mean_proba_train:.3f}, test={mean_proba_test:.3f}, thresh={mean_proba_thresh:.3f}, live={mean_proba_live:.3f}")
            log.info(f"Drift: train→live={drift_train_live:.3f}, test→live={drift_test_live:.3f}")
            log.info(f"Stability Index={stability_index:.3f} (1=perfect stability, 0=unstable)")
        except Exception as e:
            stability_index = 0.5
            log.warning(f"⚠️ Stability calculation issue for {name}: {e}")

        results.append({
            "Model": name,
            "Best_Params": best_params,
            "Bias_Penalty": bias_penalty,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "F1": f1,
            "NegThr": neg_thr,
            "PosThr": pos_thr,
            "Live_Accuracy": acc_live,
            "Live_F1": f1_live,
            "Stability_Index": stability_index,
            "Best_Model": best_model,
            "Scaler": scaler,
            "Y_proba_live": y_proba_live,
        })

    except Exception as e:
        log.error(f"❌ Error training {name}: {e}")

# ============================================================
# 5️⃣ Full report summary
# ============================================================
stage("Comprehensive Model Stability Summary")

stability_df = pd.DataFrame(results)[["Model", "Accuracy", "Live_Accuracy", "F1", "Live_F1", "Stability_Index", "Bias_Penalty"]]
stability_df["Performance_Drift"] = abs(stability_df["F1"] - stability_df["Live_F1"])
stability_df["Composite_Score"] = (stability_df["Stability_Index"] * 0.5) + (1 - stability_df["Performance_Drift"]) * 0.5 - stability_df["Bias_Penalty"]

log.info("\n📊 MODEL STABILITY ANALYSIS REPORT:")
log.info(stability_df.sort_values("Composite_Score", ascending=False).to_string(index=False))
log.info("==============================================================")
log.info("✅ Stability and drift analysis completed successfully.")
