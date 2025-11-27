#!/usr/bin/env python3
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from datetime import datetime

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
    """Display stage separator"""
    sep = "=" * 65
    log.info(f"\n{sep}\n🟢 STAGE: {msg}\n{sep}")

# ============================================================
# 2️⃣ Load prepared dataset
# ============================================================
stage("Loading dataset")
df = pd.read_csv("prepared_train_data.csv")
if "target" not in df.columns:
    raise ValueError("❌ 'target' column missing in dataset")

X = df.drop("target", axis=1)
y = df["target"].astype(int)

# Clean non-numeric and NaNs
X = X.select_dtypes(include=[np.number])
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Split last 1000 rows as live data
live_size = 1000
X_live, y_live = X.tail(live_size), y.tail(live_size)
X_train_full, y_train_full = X.iloc[:-live_size], y.iloc[:-live_size]

# Split training data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

log.info(f"Dataset loaded: total={len(df)}, train={len(X_train)}, test={len(X_test)}, live={len(X_live)}")

# ============================================================
# 3️⃣ Define models and hyperparameter grids
# ============================================================
stage("Defining models and hyperparameter grids")

models_config = {
    "XGBoost": {
        "model": XGBClassifier(random_state=2025, n_jobs=-1),
        "scaler": None,
        "param_grid": {
            "n_estimators": [200, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
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
    "SVM (RBF)": {
        "model": SVC(probability=True, random_state=2025),
        "scaler": StandardScaler(),
        "param_grid": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", 0.1, 0.01],
            "kernel": ["rbf"],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=2025, n_jobs=-1),
        "scaler": MinMaxScaler(),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [6, 8, 10],
            "max_features": ["sqrt", "log2"],
        },
    },
}

# ============================================================
# 4️⃣ Training and evaluation on training/test data
# ============================================================
stage("Model training and evaluation")
results = []

for name, cfg in models_config.items():
    try:
        model = cfg["model"]
        scaler = cfg["scaler"]
        params = cfg["param_grid"]

        log.info(f"\n🔹 Starting GridSearch for {name} ...")
        if scaler is not None:
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_live_scaled = scaler.transform(X_live)
        else:
            X_train_scaled, X_test_scaled, X_live_scaled = X_train, X_test, X_live

        grid = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        log.info(f"✅ {name}: Best Params = {best_params}")

        # Evaluate on test data
        y_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        wins = int(((y_pred == 1) & (y_test == 1)).sum())
        loses = int(((y_pred == 1) & (y_test == 0)).sum())

        log.info(f"--------------------------- {name} ---------------------------")
        log.info(f"Accuracy:           {acc:.4f}")
        log.info(f"Balanced Accuracy:  {bal_acc:.4f}")
        log.info(f"F1 Score:           {f1:.4f}")
        log.info(f"Wins (pred=1,y=1):  {wins}")
        log.info(f"Loses(pred=1,y=0):  {loses}")
        log.info(f"-------------------------------------------------------------")

        results.append({
            "Model": name,
            "Best_Params": best_params,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "F1": f1,
            "Wins": wins,
            "Loses": loses,
            "Best_Model": best_model,
            "Scaler": scaler,
            "X_live_scaled": X_live_scaled,
        })

    except Exception as e:
        log.error(f"❌ Error training {name}: {e}")

# ============================================================
# 5️⃣ Evaluation on LIVE data
# ============================================================
stage("Evaluating models on LIVE data")

for r in results:
    try:
        name = r["Model"]
        model = r["Best_Model"]
        scaler = r["Scaler"]
        X_live_scaled = r["X_live_scaled"]

        log.info(f"\n🔹 Evaluating {name} on live data ...")
        y_live_pred = model.predict(X_live_scaled)

        acc = accuracy_score(y_live, y_live_pred)
        bal_acc = balanced_accuracy_score(y_live, y_live_pred)
        f1 = f1_score(y_live, y_live_pred)
        wins = int(((y_live_pred == 1) & (y_live == 1)).sum())
        loses = int(((y_live_pred == 1) & (y_live == 0)).sum())

        log.info(f"--------------------------- {name} (LIVE) ---------------------------")
        log.info(f"Accuracy:           {acc:.4f}")
        log.info(f"Balanced Accuracy:  {bal_acc:.4f}")
        log.info(f"F1 Score:           {f1:.4f}")
        log.info(f"Wins (pred=1,y=1):  {wins}")
        log.info(f"Loses(pred=1,y=0):  {loses}")
        log.info(f"Best Params:        {r['Best_Params']}")
        log.info(f"-------------------------------------------------------------")

        r["Live_Accuracy"] = acc
        r["Live_BalAcc"] = bal_acc
        r["Live_F1"] = f1

    except Exception as e:
        log.error(f"❌ Error evaluating {name} on live data: {e}")

# ============================================================
# 6️⃣ Summary comparison
# ============================================================
stage("Summary comparison")
results_df = pd.DataFrame(results)[
    [
        "Model",
        "Accuracy",
        "Balanced_Accuracy",
        "F1",
        "Live_Accuracy",
        "Live_BalAcc",
        "Live_F1",
    ]
].sort_values("F1", ascending=False).reset_index(drop=True)

log.info("\n📊 FINAL COMPARISON:")
log.info(results_df.to_string(index=False))

# Additional qualitative analysis
log.info("\n🔍 Qualitative model insights:")
for _, r in results_df.iterrows():
    strengths, weaknesses = [], []
    if r["F1"] > results_df["F1"].mean():
        strengths.append("High stability on training/test data")
    else:
        weaknesses.append("Below-average F1")

    if r["Live_F1"] > results_df["Live_F1"].mean():
        strengths.append("Strong live generalization")
    else:
        weaknesses.append("Weak live performance")

    if r["Balanced_Accuracy"] > 0.7:
        strengths.append("Handles class imbalance well")
    else:
        weaknesses.append("Might struggle with class imbalance")

    log.info(f"\n📈 {r['Model']}:")
    log.info(f"   Strengths : {', '.join(strengths) if strengths else '—'}")
    log.info(f"   Weaknesses: {', '.join(weaknesses) if weaknesses else '—'}")

log.info("\n✅ All stages completed successfully.")
