#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import pandas as pd

LOG_FILE = "compare.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def stage(title: str):
    sep = "=" * 80
    log.info("\n%s\n🟢 STAGE: %s\n%s", sep, title, sep)

def read_csv_safe(path: str, required_cols=None, empty_cols=None, note=""):
    """Read CSV if exists; otherwise return empty DF with desired columns."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if required_cols and not set(required_cols).issubset(df.columns):
                log.warning(f"⚠️ {path} missing some required columns. Found={list(df.columns)} Needed={required_cols}")
            log.info(f"✅ Loaded {path} shape={df.shape} {note}")
            return df
        except Exception as e:
            log.error(f"❌ Failed to read {path}: {e}")
    else:
        log.warning(f"⚠️ Missing file: {path} {note}")
    # empty
    cols = empty_cols or (required_cols or [])
    return pd.DataFrame(columns=cols)

def normalize_ml_model_table(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    ML stability_report.csv columns (typical):
      Model, Accuracy, Live_Accuracy, F1, Live_F1, Stability_Index, Performance_Drift, Bias_Penalty
    We map to a common schema.
    """
    if df_ml.empty:
        return pd.DataFrame(columns=[
            "Family","Model","Test_Acc","Test_BalAcc","Test_F1","Live_Acc","Live_BalAcc","Live_F1","Drift_F1","Bias_Penalty"
        ])
    out = pd.DataFrame()
    out["Family"] = "ML"
    out["Model"]  = df_ml.get("Model", pd.Series(["?"]*len(df_ml)))
    out["Test_Acc"]     = df_ml.get("Accuracy", np.nan)
    # ML report ممکن است Test_BalAcc جدا نداشته باشد
    out["Test_BalAcc"]  = np.nan
    out["Test_F1"]      = df_ml.get("F1", np.nan)
    out["Live_Acc"]     = df_ml.get("Live_Accuracy", np.nan)
    out["Live_BalAcc"]  = np.nan  # در ML برخی اسکریپت‌ها ست نشده؛ اگر داشتی می‌تواند اضافه شود
    out["Live_F1"]      = df_ml.get("Live_F1", np.nan)
    # Drift_F1: اگر در ML موجود نبود با |F1 - Live_F1| محاسبه شود
    drift = None
    if "Performance_Drift" in df_ml.columns:
        drift = df_ml["Performance_Drift"]
    else:
        drift = (df_ml.get("F1", np.nan) - df_ml.get("Live_F1", np.nan)).abs()
    out["Drift_F1"]     = drift
    out["Bias_Penalty"] = df_ml.get("Bias_Penalty", np.nan)
    return out

def normalize_deep_model_table(df_deep: pd.DataFrame) -> pd.DataFrame:
    """
    Deep stability report columns (from deep_ensemble_tabular_voting.py):
      Model, Test_Accuracy, Test_BalAcc, Test_F1, Live_Accuracy, Live_BalAcc, Live_F1, NegThr, PosThr, Bias_Penalty, Perf_Drift(F1)
    """
    if df_deep.empty:
        return pd.DataFrame(columns=[
            "Family","Model","Test_Acc","Test_BalAcc","Test_F1","Live_Acc","Live_BalAcc","Live_F1","Drift_F1","Bias_Penalty"
        ])
    out = pd.DataFrame()
    out["Family"] = "Deep"
    out["Model"]  = df_deep.get("Model", pd.Series(["?"]*len(df_deep)))
    out["Test_Acc"]     = df_deep.get("Test_Accuracy", np.nan)
    out["Test_BalAcc"]  = df_deep.get("Test_BalAcc", np.nan)
    out["Test_F1"]      = df_deep.get("Test_F1", np.nan)
    out["Live_Acc"]     = df_deep.get("Live_Accuracy", np.nan)
    out["Live_BalAcc"]  = df_deep.get("Live_BalAcc", np.nan)
    out["Live_F1"]      = df_deep.get("Live_F1", np.nan)
    drift = df_deep.get("Perf_Drift(F1)", np.nan)
    if drift.isna().all():
        drift = (df_deep.get("Test_F1", np.nan) - df_deep.get("Live_F1", np.nan)).abs()
    out["Drift_F1"]     = drift
    out["Bias_Penalty"] = df_deep.get("Bias_Penalty", np.nan)
    return out

def compute_ensemble_metrics(signals_path, ensemble_pred_path, label="ML"):
    """
    Returns dict with coverage, counts and accuracy on confident predictions.
    signals.csv / deep_signals.csv:
      Index, Signal (BUY/SELL/NONE)
    ensemble_predictions.csv / deep_ensemble_predictions.csv:
      Index, y_true, Votes_BUY, Confident_Models, Mean_Confidence
    """
    req_sig = ["Index","Signal"]
    req_pred = ["Index","y_true"]

    sig = read_csv_safe(signals_path, required_cols=req_sig, empty_cols=req_sig, note=f"[{label} signals]")
    ens = read_csv_safe(ensemble_pred_path, required_cols=req_pred, empty_cols=req_pred, note=f"[{label} ensemble_predictions]")

    # coverage
    total = len(sig)
    buy_n  = int((sig["Signal"] == "BUY").sum()) if total else 0
    sell_n = int((sig["Signal"] == "SELL").sum()) if total else 0
    none_n = int((sig["Signal"] == "NONE").sum()) if total else 0
    coverage = ((buy_n + sell_n) / total * 100) if total > 0 else 0.0

    # accuracy on confident (merge with y_true)
    acc = bal = f1 = np.nan
    if not sig.empty and not ens.empty and set(["Index","y_true"]).issubset(ens.columns):
        m = sig.merge(ens[["Index","y_true"]], on="Index", how="inner")
        m_conf = m[m["Signal"] != "NONE"].copy()
        if not m_conf.empty:
            y_pred = np.where(m_conf["Signal"] == "BUY", 1, 0)
            y_true = m_conf["y_true"].astype(int).values
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            acc = accuracy_score(y_true, y_pred)
            bal = balanced_accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred)
        else:
            log.warning(f"⚠️ No confident rows for accuracy in {label} ensemble.")
    else:
        log.warning(f"⚠️ Missing inputs for accuracy calculation in {label} ensemble.")

    return {
        "Family": label,
        "Coverage(%)": coverage,
        "BUY": buy_n, "SELL": sell_n, "NONE": none_n,
        "Live_Acc": acc, "Live_BalAcc": bal, "Live_F1": f1
    }

def main():
    stage("Load per-model reports")
    ml_stab   = read_csv_safe("stability_report.csv")
    deep_stab = read_csv_safe("deep_stability_report.csv")

    stage("Normalize model-level tables")
    ml_norm   = normalize_ml_model_table(ml_stab)
    deep_norm = normalize_deep_model_table(deep_stab)

    model_cmp = pd.concat([ml_norm, deep_norm], ignore_index=True)
    out_model = "model_level_comparison.csv"
    model_cmp.to_csv(out_model, index=False)
    log.info(f"💾 Saved {out_model} shape={model_cmp.shape}")

    stage("Compute ensemble-level metrics (coverage, distribution, accuracy)")
    ml_ens = compute_ensemble_metrics("signals.csv", "ensemble_predictions.csv", label="ML")
    dp_ens = compute_ensemble_metrics("deep_signals.csv", "deep_ensemble_predictions.csv", label="Deep")

    ens_cmp = pd.DataFrame([ml_ens, dp_ens])
    out_ens = "ensemble_level_comparison.csv"
    ens_cmp.to_csv(out_ens, index=False)
    log.info(f"💾 Saved {out_ens} shape={ens_cmp.shape}")

    stage("Summary")
    log.info("Model-level preview:\n%s", model_cmp.head(10).to_string(index=False))
    log.info("Ensemble-level summary:\n%s", ens_cmp.to_string(index=False))
    log.info("\n✅ Compare pipeline finished. Outputs:\n  - %s\n  - %s", out_model, out_ens)

if __name__ == "__main__":
    main()
