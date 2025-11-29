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
import os

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

if not os.path.exists("prepared_train_data.csv"):
    raise FileNotFoundError("❌ File 'prepared_train_data.csv' not found in current directory.")

df = pd.read_csv("prepared_train_data.csv")
if "target" not in df.columns:
    raise ValueError("❌ 'target' column missing in dataset")

X = df.drop("target", axis=1)
y = df["target"].astype(int)

price_col = next((c for c in X.columns if "close" in c.lower()), None)
prices = X[price_col].values if price_col else np.arange(len(X))

X = X.select_dtypes(include=[np.number])
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

total_len = len(X)
live_size = 2000
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

log.info(f"✅ Dataset split complete:")
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
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 2.0],
        },
    },
    "HistGradientBoosting": {
        "model": HistGradientBoostingClassifier(random_state=2025),
        "scaler": None,
        "param_grid": {
            "max_iter": [200, 400],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6],
            "l2_regularization": [0.0, 1.0],
        },
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=5000, random_state=2025, solver="saga"),
        "scaler": StandardScaler(),
        "param_grid": {
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "class_weight": [None, "balanced"]
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=2025, n_jobs=-1),
        "scaler": MinMaxScaler(),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [6, 8],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced_subsample"],
        },
    },
}

# ============================================================
# 4️⃣ Training, Bias, Threshold, and Stability Analysis
# ============================================================
stage("Training models, checking bias & stability")

results = []
for name, cfg in models_config.items():
    try:
        model, scaler, params = cfg["model"], cfg["scaler"], cfg["param_grid"]
        log.info(f"\n🔹 Optimizing {name} parameters with GridSearch ...")

        if scaler:
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_thresh_s = scaler.transform(X_thresh)
            X_live_s = scaler.transform(X_live)
        else:
            X_train_s, X_test_s, X_thresh_s, X_live_s = X_train, X_test, X_thresh, X_live

        grid = GridSearchCV(model, params, scoring="f1", cv=3, n_jobs=-1)
        grid.fit(X_train_s, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        y_pred_train = best_model.predict(X_train_s)
        buy_ratio, sell_ratio = np.mean(y_pred_train == 1), np.mean(y_pred_train == 0)
        bias_penalty = 0.1 if buy_ratio > 0.9 or sell_ratio > 0.9 else 0
        log.info(f"🔸 {name} bias → BUY={buy_ratio:.2f}, SELL={sell_ratio:.2f}, penalty={bias_penalty}")

        y_pred_test = best_model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred_test) - bias_penalty
        bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        y_proba_thresh = best_model.predict_proba(X_thresh_s)[:, 1]
        tf = ThresholdFinder(steps=600, min_predictions_ratio=0.9)
        neg_thr, pos_thr, thr_acc, *_ = tf.find_best_thresholds(y_proba_thresh, y_thresh.values)
        y_proba_live = best_model.predict_proba(X_live_s)[:, 1]

        y_pred_live = np.full_like(y_live, -1)
        y_pred_live[y_proba_live <= neg_thr] = 0
        y_pred_live[y_proba_live >= pos_thr] = 1
        mask = y_pred_live != -1
        acc_live = accuracy_score(y_live[mask], y_pred_live[mask]) if mask.any() else 0
        bal_live = balanced_accuracy_score(y_live[mask], y_pred_live[mask]) if mask.any() else 0
        f1_live = f1_score(y_live[mask], y_pred_live[mask]) if mask.any() else 0

        mean_proba_train = np.mean(best_model.predict_proba(X_train_s)[:, 1])
        mean_proba_live = np.mean(y_proba_live)
        drift = abs(mean_proba_live - mean_proba_train)
        stability_index = max(0, 1 - drift)
        log.info(f"🧩 {name} drift={drift:.4f}, stability={stability_index:.3f}")

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
# 5️⃣ Export stability report
# ============================================================
stage("Exporting model stability report")
if results:
    stab_df = pd.DataFrame(results)[[
        "Model", "Accuracy", "Live_Accuracy", "F1", "Live_F1", "Stability_Index", "Bias_Penalty"
    ]]
    stab_df["Performance_Drift"] = abs(stab_df["F1"] - stab_df["Live_F1"])
    stab_df.to_csv("stability_report.csv", index=False)
    log.info("💾 stability_report.csv saved successfully.")
else:
    log.warning("⚠️ No model results available to export stability report.")

# ============================================================
# 6️⃣ Voting and prediction export
# ============================================================
stage("Creating ensemble voting and predictions")

votes, probas = [], []
for r in results:
    y_proba = r["Y_proba_live"]
    y_pred = np.full(len(y_proba), -1)
    y_pred[y_proba <= r["NegThr"]] = 0
    y_pred[y_proba >= r["PosThr"]] = 1
    votes.append(y_pred)
    probas.append(y_proba)

votes = np.array(votes)
probas = np.array(probas)
vote_sum = np.sum(votes == 1, axis=0)
vote_conf = np.sum(votes != -1, axis=0)
mean_conf = np.nanmean(np.where(votes != -1, probas, np.nan), axis=0)

ens_df = pd.DataFrame({
    "Index": np.arange(len(y_live)),
    "y_true": y_live.values,
    "Votes_BUY": vote_sum,
    "Confident_Models": vote_conf,
    "Mean_Confidence": mean_conf
})
ens_df.to_csv("ensemble_predictions.csv", index=False)
log.info("💾 ensemble_predictions.csv created successfully.")

# ============================================================
# 7️⃣ Generate signals.csv + Coverage report
# ============================================================
stage("Generating final BUY/SELL signals and computing live accuracy + coverage")

signals = np.full(len(vote_sum), "NONE", dtype=object)
signals[vote_sum >= 3] = "BUY"
signals[(vote_conf >= 3) & (vote_sum <= 1)] = "SELL"

sig_df = pd.DataFrame({
    "Index": np.arange(len(signals)),
    "Signal": signals,
    "Price": price_live,
    "Confidence": mean_conf,
    "Votes_BUY": vote_sum,
    "Confident_Models": vote_conf
})
sig_df.to_csv("signals.csv", index=False)
log.info("💾 signals.csv saved successfully.")

# Count categories
buy_count = np.sum(signals == "BUY")
sell_count = np.sum(signals == "SELL")
none_count = np.sum(signals == "NONE")
total_count = len(signals)
coverage = ((buy_count + sell_count) / total_count) * 100 if total_count > 0 else 0

log.info(f"✅ BUY={buy_count}, SELL={sell_count}, NONE={none_count}")
log.info(f"📈 COVERAGE (Predictable Ratio): {coverage:.2f}% of live data covered")

# 🔹 Compute overall ensemble accuracy on live data (using voting)
mask_live = signals != "NONE"
if np.any(mask_live):
    y_pred_final = np.where(signals[mask_live] == "BUY", 1, 0)
    acc_live_final = accuracy_score(y_live[mask_live], y_pred_final)
    bal_acc_live_final = balanced_accuracy_score(y_live[mask_live], y_pred_final)
    f1_live_final = f1_score(y_live[mask_live], y_pred_final)
    log.info(f"\n📊 FINAL ENSEMBLE LIVE ACCURACY REPORT:")
    log.info(f"  Accuracy: {acc_live_final:.4f}")
    log.info(f"  Balanced Accuracy: {bal_acc_live_final:.4f}")
    log.info(f"  F1 Score: {f1_live_final:.4f}")
    log.info(f"  Coverage: {coverage:.2f}%  (BUY={buy_count}, SELL={sell_count}, NONE={none_count})")
else:
    log.warning("⚠️ No confident live predictions for accuracy computation.")

log.info("\n✅ All CSV files successfully generated and saved.")
