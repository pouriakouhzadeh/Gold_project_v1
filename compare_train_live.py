#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_train_live.py — v2.3
- Console-first logging (stage banners + samples)
- Limit heavy reports to last ANALYZE_LAST rows
- Robust scaler check (handles NotFitted safely)
- Single vs. cumulative-batch prediction diff

Outputs (در مسیر جاری):
  1) features_compare_detailed.csv
  2) features_compare_summary.csv
  3) scaler_check.csv
  4) predictions_compare.csv
  5) live_single_vs_batch_preds.csv
"""

from __future__ import annotations
import os, sys, csv, math, logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.exceptions import NotFittedError

# ماژول‌های پروژه شما
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN       # type: ignore
from ModelSaver import ModelSaver                               # type: ignore

# ================== CONFIG ==================
SYMBOL       = "XAUUSD"
MAIN_TF      = "30T"
MODEL_PATH   = "best_model.pkl"

# چند سطر آخر که برای «همهٔ گزارش‌ها» تحلیل می‌کنیم (برای کنترل حجم)
ANALYZE_LAST = 2000

ATOL         = 1e-9
RTOL         = 1e-9

OUT_DETAILED = "features_compare_detailed.csv"
OUT_SUMMARY  = "features_compare_summary.csv"
OUT_SCALER   = "scaler_check.csv"
OUT_PREDS    = "predictions_compare.csv"
OUT_WARMUP   = "live_single_vs_batch_preds.csv"

CHECK_WEEK_LAST_BAR = True
CHECK_DAY_LAST_BAR  = True

LOG_LEVEL    = logging.INFO
# ============================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("compare_train_live_v2_3")

def banner(msg: str):
    LOGGER.info("\n" + "="*80 + f"\n{msg}\n" + "="*80)

def sample_df(df: pd.DataFrame, name: str, n: int = 3):
    if df.empty:
        LOGGER.info("📄 %s: EMPTY", name); return
    LOGGER.info("📄 %s: shape=%s  head:", name, df.shape)
    LOGGER.info("\n%s", df.head(n).to_string(index=False))
    LOGGER.info("… tail:")
    LOGGER.info("\n%s", df.tail(n).to_string(index=False))

def _build_filepaths(symbol: str) -> dict:
    return {
        "30T": f"./{symbol}_M30.csv",
        "15T": f"./{symbol}_M15.csv",
        "5T":  f"./{symbol}_M5.csv",
        "1H":  f"./{symbol}_H1.csv",
    }

@dataclass
class LoadedModel:
    pipeline: any
    window_size: int
    neg_thr: float
    pos_thr: float
    train_window_cols: List[str]
    model_dir: str

def load_model_payload(path: str) -> LoadedModel:
    payload = ModelSaver(filename=os.path.basename(path),
                         model_dir=os.path.abspath(os.path.dirname(path))).load_full()
    pipeline   = payload["pipeline"]
    window_sz  = int(payload["window_size"])
    neg_thr    = float(payload["neg_thr"])
    pos_thr    = float(payload["pos_thr"])
    cols_order = list(payload.get("train_window_cols") or payload.get("feats") or [])
    return LoadedModel(
        pipeline=pipeline,
        window_size=window_sz,
        neg_thr=neg_thr,
        pos_thr=pos_thr,
        train_window_cols=cols_order,
        model_dir=os.path.abspath(os.path.dirname(path))
    )

def compute_true_targets(raw: pd.DataFrame, main_tf: str) -> pd.Series:
    close_col = f"{main_tf}_close"
    if close_col not in raw.columns:
        raise KeyError(f"Missing column: {close_col}")
    return (raw[close_col].shift(-1) - raw[close_col] > 0).astype("int64")

def build_chimney(prep: PREPARE_DATA_FOR_TRAIN, raw: pd.DataFrame,
                  cols: List[str], window: int, main_tf: str) -> pd.DataFrame:
    X, y, feats, price, t_idx = prep.ready(
        raw,
        window=window,
        selected_features=cols,
        mode="train",
        with_times=True,
        predict_drop_last=False,
        train_drop_last=True
    )
    Xc = X.copy()
    Xc["__time__"] = pd.to_datetime(t_idx)
    Xc = Xc[cols + ["__time__"]]
    return Xc.reset_index(drop=True)

def build_live(prep: PREPARE_DATA_FOR_TRAIN, raw: pd.DataFrame,
               cols: List[str], window: int, main_tf: str) -> pd.DataFrame:
    total = len(raw)
    # برای سرعت: فقط بازه‌ای تولید کنیم که انتهای آن منطبق با ANALYZE_LAST شود
    # ابتدا یک بار چیمنی می‌سازیم تا length نهایی را بدانیم؛ در این نسخه فرض می‌گیریم کل بازه لازم است
    start_t = max(0, total - (ANALYZE_LAST + window*3 + 2000))  # حاشیه امن برای اندیکاتورها
    rows = []
    for t in range(start_t, total):
        known = raw.iloc[:t].copy()
        X_full, _, feats, price, t_idx = prep.ready(
            known,
            window=window,
            selected_features=cols,
            mode="predict",
            with_times=True,
            predict_drop_last=True
        )
        if X_full.empty or t_idx is None or len(t_idx) == 0:
            continue
        X_last = X_full.tail(1).copy()
        ts = pd.to_datetime(t_idx.iloc[-1])
        row = X_last.iloc[0].copy()
        row["__time__"] = ts
        rows.append(row)
    Xl = pd.DataFrame(rows)
    if Xl.empty:
        return pd.DataFrame(columns=cols + ["__time__"])
    for c in cols:
        if c not in Xl.columns:
            Xl[c] = np.nan
    Xl = Xl[cols + ["__time__"]]
    return Xl.reset_index(drop=True)

def exact_join_last(chim: pd.DataFrame, live: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = "__time__"
    c2 = chim.set_index(key)
    l2 = live.set_index(key)
    common_idx = c2.index.intersection(l2.index).sort_values()
    if len(common_idx) == 0:
        return pd.DataFrame(columns=chim.columns), pd.DataFrame(columns=live.columns)
    # فقط 2000 سطر آخرِ مشترک
    if len(common_idx) > ANALYZE_LAST:
        common_idx = common_idx[-ANALYZE_LAST:]
    c3 = c2.loc[common_idx].reset_index()
    l3 = l2.loc[common_idx].reset_index()
    return c3, l3

def week_last_bar_flags(times: pd.Series) -> pd.Series:
    if times.empty: return pd.Series([], dtype=bool)
    idx = pd.to_datetime(times)
    wk = idx.dt.isocalendar().week
    yr = idx.dt.isocalendar().year
    df = pd.DataFrame({"t": idx, "w": wk, "y": yr})
    last_ts = df.groupby(["y","w"])["t"].transform("max")
    return idx.eq(last_ts)

def day_last_bar_flags(times: pd.Series) -> pd.Series:
    if times.empty: return pd.Series([], dtype=bool)
    idx = pd.to_datetime(times)
    last_ts = idx.groupby(idx.dt.date).transform("max")
    return idx.eq(last_ts)

def classify_with_band(prob1: float, neg_thr: float, pos_thr: float) -> int:
    if prob1 <= neg_thr: return 0
    if prob1 >= pos_thr: return 1
    return -1

def map_true_for_times(y_full: pd.Series, raw: pd.DataFrame,
                       main_tf: str, times: pd.Series) -> np.ndarray:
    tcol = f"{main_tf}_time" if f"{main_tf}_time" in raw.columns else "time"
    time_to_y = pd.Series(y_full.values, index=pd.to_datetime(raw[tcol])).to_dict()
    return np.array([int(time_to_y.get(pd.to_datetime(t), -999)) for t in times], dtype=int)

def compute_financial_metrics(price_ser: pd.Series, preds: np.ndarray) -> Tuple[float, float]:
    prices = pd.to_numeric(price_ser, errors="coerce").astype(float).values
    if len(prices) < 2 or len(preds) < 2:
        return 0.0, 0.0
    ret_mkt = np.diff(prices) / prices[:-1]
    L = min(len(ret_mkt), len(preds) - 1)
    pos = np.zeros_like(ret_mkt, dtype=float)
    pos[:L] = (preds[:-1][:L] == 1).astype(float)
    ret_str = ret_mkt * pos
    ret_ser = pd.Series(ret_str)
    sharpe = (ret_ser.mean() / ret_ser.std()) * np.sqrt(252 * 48) if ret_ser.std() > 0 else 0.0
    cum_eq = ret_ser.add(1).cumprod()
    maxdd  = (cum_eq.cummax() - cum_eq).max()
    return float(sharpe), float(maxdd)

def write_scaler_report(cols: List[str],
                        Cvals: pd.DataFrame, Lvals: pd.DataFrame,
                        scaler: Optional[any], note: str) -> None:
    rows = []
    if scaler is not None:
        try:
            Ctr = pd.DataFrame(scaler.transform(Cvals), columns=cols)
            Ltr = pd.DataFrame(scaler.transform(Lvals), columns=cols)
            for c in cols:
                mc, sc = float(np.mean(Ctr[c])), float(np.std(Ctr[c]))
                ml, sl = float(np.mean(Ltr[c])), float(np.std(Ltr[c]))
                rows.append([c, mc, sc, ml, sl, abs(ml-mc), abs(sl-sc), "scaled_ok"])
        except NotFittedError:
            for c in cols:
                mc, sc = float(np.mean(Cvals[c])), float(np.std(Cvals[c]))
                ml, sl = float(np.mean(Lvals[c])), float(np.std(Lvals[c]))
                rows.append([c, mc, sc, ml, sl, abs(ml-mc), abs(sl-sc), "scaler_not_fitted"])
        except Exception as e:
            for c in cols:
                mc, sc = float(np.mean(Cvals[c])), float(np.std(Cvals[c]))
                ml, sl = float(np.mean(Lvals[c])), float(np.std(Lvals[c]))
                rows.append([c, mc, sc, ml, sl, abs(ml-mc), abs(sl-sc), f"scaler_error:{type(e).__name__}"])
    else:
        for c in cols:
            mc, sc = float(np.mean(Cvals[c])), float(np.std(Cvals[c]))
            ml, sl = float(np.mean(Lvals[c])), float(np.std(Lvals[c]))
            rows.append([c, mc, sc, ml, sl, abs(ml-mc), abs(sl-sc), note or "no_scaler"])

    pd.DataFrame(rows, columns=[
        "column","chimney_mean","chimney_std","live_mean","live_std",
        "abs_delta_mean","abs_delta_std","note"
    ]).to_csv(OUT_SCALER, index=False, encoding="utf-8")
    LOGGER.info("📦 [out] scaler check → %s (%d rows)", os.path.abspath(OUT_SCALER), len(rows))
    LOGGER.info("🧪 scaler notes counts:\n%s",
                pd.Series([r[-1] for r in rows]).value_counts().to_string())

def main():
    banner("⏳ LOAD MODEL")
    if not os.path.isfile(MODEL_PATH):
        LOGGER.error("❌ Model file not found: %s", os.path.abspath(MODEL_PATH)); sys.exit(1)
    mdl = load_model_payload(MODEL_PATH)
    LOGGER.info("✅ model loaded: window=%d  n_cols=%d  thr=(neg=%.4f, pos=%.4f)",
                mdl.window_size, len(mdl.train_window_cols), mdl.neg_thr, mdl.pos_thr)

    banner("⏳ LOAD & SORT DATA")
    filepaths = _build_filepaths(SYMBOL)
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe=MAIN_TF, verbose=True)
    raw = prep.load_data()
    tcol = f"{MAIN_TF}_time" if f"{MAIN_TF}_time" in raw.columns else "time"
    close_col = f"{MAIN_TF}_close"
    raw[tcol] = pd.to_datetime(raw[tcol])
    raw.sort_values(tcol, inplace=True)
    raw.reset_index(drop=True, inplace=True)
    LOGGER.info("✅ raw rows=%d  time-range=[%s … %s]", len(raw), raw[tcol].min(), raw[tcol].max())

    banner("⏳ BUILD CHIMNEY & LIVE FEATURES")
    cols = mdl.train_window_cols[:]
    window = mdl.window_size
    chim_full = build_chimney(prep, raw, cols, window, MAIN_TF)
    live_full = build_live(prep, raw, cols, window, MAIN_TF)
    LOGGER.info("✅ built: chimney=%d rows, live=%d rows", len(chim_full), len(live_full))
    sample_df(chim_full, "chimney_full", 2)
    sample_df(live_full, "live_full", 2)

    banner("⏳ ALIGN & SLICE LAST ANALYZE_LAST")
    chim2, live2 = exact_join_last(chim_full, live_full)
    if len(chim2) == 0:
        LOGGER.error("❌ No overlapping timestamps between chimney and live."); sys.exit(2)
    LOGGER.info("✅ aligned rows (last %d): %d", ANALYZE_LAST, len(chim2))
    LOGGER.info("⏱  range: [%s … %s]", chim2["__time__"].min(), chim2["__time__"].max())
    sample_df(chim2, "chimney_aligned_tail", 2)
    sample_df(live2, "live_aligned_tail", 2)

    if CHECK_WEEK_LAST_BAR:
        wk_last = week_last_bar_flags(chim2["__time__"])
        LOGGER.info("ℹ️ last-bar-of-week rows: %d", int(wk_last.sum()))
    if CHECK_DAY_LAST_BAR:
        day_last = day_last_bar_flags(chim2["__time__"])
        LOGGER.info("ℹ️ last-bar-of-day rows: %d", int(day_last.sum()))

    banner("⏳ FEATURE DIFFS (DETAILED & SUMMARY)")
    detailed_rows = []
    mismatch_counts: Dict[str, int] = {c: 0 for c in cols}
    abs_sums: Dict[str, float] = {c: 0.0 for c in cols}
    abs_max: Dict[str, float] = {c: 0.0 for c in cols}

    for i in range(len(chim2)):
        ts = chim2["__time__"].iloc[i]
        row_c = chim2.iloc[i]
        row_l = live2.iloc[i]
        for c in cols:
            vc = float(row_c[c]) if pd.notna(row_c[c]) else np.nan
            vl = float(row_l[c]) if pd.notna(row_l[c]) else np.nan
            if pd.isna(vc) and pd.isna(vl):
                equal = True; diff = rel = 0.0
            elif pd.isna(vc) or pd.isna(vl):
                equal = False; diff = np.nan; rel = np.nan
            else:
                diff = vl - vc
                denom = max(1e-15, abs(vc))
                rel = abs(diff) / denom
                equal = math.isclose(vl, vc, rel_tol=RTOL, abs_tol=ATOL)
            if not equal:
                mismatch_counts[c] += 1
                if not pd.isna(diff):
                    ad = abs(diff)
                    abs_sums[c] += ad
                    if ad > abs_max[c]:
                        abs_max[c] = ad
            detailed_rows.append([ts.isoformat(), c, vc, vl, diff, rel, 0 if equal else 1])

    pd.DataFrame(detailed_rows, columns=[
        "timestamp","column","chimney_value","live_value",
        "diff_live_minus_chimney","relative_diff","mismatch_flag"
    ]).to_csv(OUT_DETAILED, index=False, encoding="utf-8")
    LOGGER.info("📦 [out] detailed diffs → %s (%d rows)",
                os.path.abspath(OUT_DETAILED), len(detailed_rows))

    total_rows = len(chim2)
    summary_rows = []
    for c in cols:
        mis = mismatch_counts[c]
        pct = (mis / total_rows) * 100.0 if total_rows else 0.0
        mean_abs = (abs_sums[c] / mis) if mis else 0.0
        summary_rows.append([c, mis, total_rows, pct, mean_abs, abs_max[c]])
    summary_rows.sort(key=lambda r: (-r[3], -r[5]))
    df_sum = pd.DataFrame(summary_rows, columns=[
        "column","mismatch_count","rows_compared","mismatch_percent",
        "mean_abs_diff_on_mismatches","max_abs_diff"
    ])
    df_sum.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    LOGGER.info("📦 [out] summary → %s (%d features)", os.path.abspath(OUT_SUMMARY), len(summary_rows))
    LOGGER.info("🧪 TOP-10 mismatched features:\n%s", df_sum.head(10).to_string(index=False))

    banner("⏳ SCALER CHECK (SAFE)")
    scaler = None
    try:
        scaler = mdl.pipeline.get_scaler()
    except Exception:
        scaler = None
    Cvals = chim2[cols].astype(float)
    Lvals = live2[cols].astype(float)
    note = "no_scaler_found" if scaler is None else ""
    write_scaler_report(cols, Cvals, Lvals, scaler, note)

    banner("⏳ PREDICTIONS (CHIMNEY vs LIVE)")
    Xc = chim2[cols].astype(float)
    Xl = live2[cols].astype(float)
    times = pd.to_datetime(chim2["__time__"])
    prob1_chim = mdl.pipeline.predict_proba(Xc)[:, 1].astype(float)
    prob1_live = mdl.pipeline.predict_proba(Xl)[:, 1].astype(float)
    pred_chim = np.array([classify_with_band(p, mdl.neg_thr, mdl.pos_thr) for p in prob1_chim], dtype=int)
    pred_live = np.array([classify_with_band(p, mdl.neg_thr, mdl.pos_thr) for p in prob1_live], dtype=int)

    # y_true
    y_full = compute_true_targets(raw, MAIN_TF)
    y_true = map_true_for_times(y_full, raw, MAIN_TF, times)

    mask_trade_chim = (pred_chim != -1)
    mask_trade_live = (pred_live != -1)

    def safe_acc_bacc(y, yhat, mask):
        if mask.any():
            return (
                float(accuracy_score(y[mask], yhat[mask])),
                float(balanced_accuracy_score(y[mask], yhat[mask])),
                float(mask.mean())
            )
        return (0.0, 0.0, 0.0)

    acc_c, bacc_c, cov_c = safe_acc_bacc(y_true, pred_chim, mask_trade_chim)
    acc_l, bacc_l, cov_l = safe_acc_bacc(y_true, pred_live, mask_trade_live)

    # قیمت هم‌تراز (ایمن)
    raw_idxed = raw.set_index(tcol)
    raw_aligned = raw_idxed.reindex(times)
    price_ser = raw_aligned[close_col].fillna(method="ffill").reset_index(drop=True)

    sharpe_c, maxdd_c = compute_financial_metrics(price_ser, pred_chim)
    sharpe_l, maxdd_l = compute_financial_metrics(price_ser, pred_live)

    # ذخیرهٔ سطر به سطر + خلاصه
    with open(OUT_PREDS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp",
            "prob1_chimney","prob1_live","diff_prob",
            "pred_chimney","pred_live","agree_pred",
            "y_true","is_correct_chim","is_correct_live"
        ])
        for i in range(len(times)):
            ts = times.iloc[i]
            p_c = float(prob1_chim[i]); p_l = float(prob1_live[i])
            pr_c = int(pred_chim[i]);   pr_l = int(pred_live[i])
            yt = int(y_true[i])
            agree = 1 if pr_c == pr_l else 0
            corr_c = "" if pr_c == -1 or yt == -999 else (1 if pr_c == yt else 0)
            corr_l = "" if pr_l == -1 or yt == -999 else (1 if pr_l == yt else 0)
            w.writerow([ts.isoformat(), f"{p_c:.6f}", f"{p_l:.6f}", f"{(p_l-p_c):.6f}",
                        pr_c, pr_l, agree, yt, corr_c, corr_l])

        w.writerow([])
        w.writerow(["__SUMMARY__"])
        w.writerow(["chimney_trades", int(mask_trade_chim.sum())])
        w.writerow(["chimney_acc_on_trades", f"{acc_c:.6f}"])
        w.writerow(["chimney_bacc_on_trades", f"{bacc_c:.6f}"])
        w.writerow(["chimney_coverage", f"{cov_c:.6f}"])
        w.writerow(["chimney_sharpe", f"{sharpe_c:.6f}"])
        w.writerow(["chimney_maxdd", f"{maxdd_c:.6f}"])
        w.writerow([])
        w.writerow(["live_trades", int(mask_trade_live.sum())])
        w.writerow(["live_acc_on_trades", f"{acc_l:.6f}"])
        w.writerow(["live_bacc_on_trades", f"{bacc_l:.6f}"])
        w.writerow(["live_coverage", f"{cov_l:.6f}"])
        w.writerow(["live_sharpe", f"{sharpe_l:.6f}"])
        w.writerow(["live_maxdd", f"{maxdd_l:.6f}"])
    LOGGER.info(
        "📦 [out] predictions compare → %s  | CHIMNEY: trades=%d acc=%.4f bacc=%.4f cov=%.2f%% | LIVE: trades=%d acc=%.4f bacc=%.4f cov=%.2f%%",
        os.path.abspath(OUT_PREDS),
        int(mask_trade_chim.sum()), acc_c, bacc_c, 100*cov_c,
        int(mask_trade_live.sum()), acc_l, bacc_l, 100*cov_l
    )

    banner("⏳ SINGLE vs CUMULATIVE-BATCH (LIVE)")
    warm_rows = []
    Xl_all = live2[cols].astype(float)
    for i in range(len(Xl_all)):
        x_single = Xl_all.iloc[i:i+1]
        p_single = float(mdl.pipeline.predict_proba(x_single)[:,1][0])
        pr_single = classify_with_band(p_single, mdl.neg_thr, mdl.pos_thr)

        x_batch = Xl_all.iloc[:i+1]
        p_batch_last = float(mdl.pipeline.predict_proba(x_batch)[:,1][-1])
        pr_batch_last = classify_with_band(p_batch_last, mdl.neg_thr, mdl.pos_thr)

        ts = chim2["__time__"].iloc[i]
        warm_rows.append([
            pd.to_datetime(ts).isoformat(),
            f"{p_single:.6f}", pr_single,
            f"{p_batch_last:.6f}", pr_batch_last,
            f"{(p_batch_last - p_single):.6f}",
            1 if pr_single == pr_batch_last else 0
        ])

    pd.DataFrame(warm_rows, columns=[
        "timestamp",
        "prob_single","pred_single",
        "prob_batch_last_cumulative","pred_batch_last_cumulative",
        "diff_prob_batch_minus_single","agree_pred"
    ]).to_csv(OUT_WARMUP, index=False, encoding="utf-8")
    LOGGER.info("📦 [out] warm-up diff → %s (%d rows)", os.path.abspath(OUT_WARMUP), len(warm_rows))

    # جمع‌بندی کوتاه روی کنسول
    banner("✅ DONE / QUICK SUMMARY")
    LOGGER.info("Compared last %d aligned rows in total.", len(chim2))
    LOGGER.info("Top-5 mismatched features (by %% mismatch):\n%s",
                df_sum.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
