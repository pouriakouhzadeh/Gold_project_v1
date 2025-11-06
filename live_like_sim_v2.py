# live_like_sim_v2.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, logging, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from ModelSaver import ModelSaver

LOG_FILE = "live_like_sim.log"

# ----------------------------- Logging -----------------------------
def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    # file
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

# ----------------------- Model payload loader ----------------------
def load_payload(model_dir:Path|None=None):
    saver = ModelSaver()
    payload = saver.load_full(model_dir=str(model_dir) if model_dir else None)
    return payload  # keys: pipeline, window_size, neg_thr, pos_thr, train_window_cols, ...

# ----------------------- Column safety utils -----------------------
def ensure_columns(X:pd.DataFrame, cols:list[str]) -> pd.DataFrame:
    X2 = X.copy()
    missing = [c for c in cols if c not in X2.columns]
    for c in missing:
        X2[c] = 0.0
    return X2[cols]

# ----------------------- Filepath resolver -------------------------
def resolve_timeframe_paths(base_dir: Path, symbol: str) -> dict[str, str]:
    candidates = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv", f"{symbol}_m30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv", f"{symbol}_m15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv",  f"{symbol}_m5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv",  f"{symbol}_h1.csv" ],
    }
    resolved: dict[str,str] = {}

    for tf, names in candidates.items():
        path_found = None
        for name in names:
            p = base_dir / name
            if p.exists():
                path_found = str(p)
                break
        if path_found is None:
            # case-insensitive scan
            for name in names:
                for child in base_dir.iterdir():
                    if child.is_file() and child.name.lower() == name.lower():
                        path_found = str(child)
                        break
                if path_found:
                    break
        if path_found is None:
            logging.warning(f"Could not find file for timeframe {tf} in {base_dir} (tried: {', '.join(names)})")
        else:
            logging.info(f"[paths] {tf} → {path_found}")
            resolved[tf] = path_found

    if "30T" not in resolved:
        raise FileNotFoundError(
            f"Main timeframe '30T' file is missing in {base_dir}. "
            f"Expected one of: {', '.join(candidates['30T'])}"
        )
    return resolved

# ----------------------- Evaluation helper -------------------------
def evaluate_block(df_part: pd.DataFrame, payload, prep: PREPARE_DATA_FOR_TRAIN,
                   use_thresholds=True, label="Eval"):
    window = int(payload["window_size"])
    cols   = payload.get("train_window_cols") or payload.get("feats") or []
    model  = payload["pipeline"]

    X, y, _, _ = prep.ready(
        df_part,
        window=window,
        selected_features=cols,
        mode="train",
        predict_drop_last=False,
        train_drop_last=False,
    )
    if X.empty or len(X) == 0:
        logging.info(f"[{label}] Empty block for evaluation.")
        return dict(size=0, acc=np.nan, bacc=np.nan, cover=np.nan,
                    correct=0, incorrect=0, unpred=0)

    # هم‌قدسازی و ایمن‌سازی
    if hasattr(y, "__len__"):
        L = min(len(X), len(y))
        X = X.iloc[:L].reset_index(drop=True)
        y = pd.Series(y).iloc[:L].reset_index(drop=True)
    else:
        y = pd.Series(y)
        L = min(len(X), len(y))
        X = X.iloc[:L].reset_index(drop=True)
        y = y.iloc[:L].reset_index(drop=True)

    X = ensure_columns(X, cols)

    y_prob = model.predict_proba(X)[:, 1]
    if use_thresholds:
        neg_thr = float(payload["neg_thr"]); pos_thr = float(payload["pos_thr"])
        y_pred = np.full(len(y), -1, dtype=int)
        y_pred[y_prob <= neg_thr] = 0
        y_pred[y_prob >= pos_thr] = 1
    else:
        y_pred = model.predict(X)

    mask = (y_pred != -1)
    if mask.any():
        acc  = accuracy_score(y[mask], y_pred[mask])
        bacc = balanced_accuracy_score(y[mask], y_pred[mask])
        correct = int(((y_pred == y) & mask).sum())
        pred_n  = int(mask.sum())
        incorrect = pred_n - correct
        unpred = int((~mask).sum())
        cover = pred_n / len(y_pred)
    else:
        acc=bacc=cover=np.nan; correct=incorrect=unpred=0

    logging.info(f"[{label}] size={len(y_pred)} cover={cover:.3f} acc={acc:.3f} bAcc={bacc:.3f} "
                 f"Correct={correct} Incorrect={incorrect} Unpred={unpred}")

    return dict(size=len(y_pred), acc=acc, bacc=bacc, cover=cover,
                correct=correct, incorrect=incorrect, unpred=unpred)

# ------------------------------- Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000, help="تعداد ردیف‌های انتهایی برای شبیه‌سازی")
    ap.add_argument("--verbosity", type=int, default=1, help="0/1/2")
    ap.add_argument("--base-dir", type=str, default=".", help="مسیر CSVها (پیش‌فرض: همین پوشه)")
    ap.add_argument("--symbol",   type=str, default="XAUUSD", help="نماد (پیش‌فرض: XAUUSD)")
    args = ap.parse_args()

    setup_logging(args.verbosity)
    logging.info("=== live_like_sim_v2 starting ===")

    # 1) مدل را بارگذاری کن
    payload = load_payload()
    model   = payload["pipeline"]
    cols    = payload.get("train_window_cols") or payload.get("feats") or []
    window  = int(payload["window_size"])
    logging.info(f"Model loaded | window={window} | cols={len(cols)} | "
                 f"neg_thr={payload['neg_thr']:.3f} | pos_thr={payload['pos_thr']:.3f}")

    # 2) فایل‌های تایم‌فریم را هوشمند پیدا کن
    base_dir = Path(args.base_dir).resolve()
    symbol   = args.symbol
    filepaths = resolve_timeframe_paths(base_dir, symbol)

    # 3) PREP را با این فایل‌ها بساز
    prep = PREPARE_DATA_FOR_TRAIN(filepaths=filepaths, main_timeframe="30T", verbose=True)
    raw  = prep.load_data()
    tcol = f"{prep.main_timeframe}_time"
    raw[tcol] = pd.to_datetime(raw[tcol])
    raw.sort_values(tcol, inplace=True)
    raw.reset_index(drop=True, inplace=True)

    total = len(raw)
    if total < args.last_n + 100:
        logging.warning(f"Dataset is small (len={total}) vs last_n={args.last_n}. Proceeding anyway.")
    logging.info(f"Raw loaded: shape={raw.shape} | Using last_n={args.last_n}")

    # 4) ارزیابی چیمنی روی بازه‌ی پایانی (batch)
    cut_start = max(0, total - args.last_n - 2*window)
    block_for_batch = raw.iloc[cut_start:].copy()
    logging.info("[Step] Evaluate Chimney (batch) on the last-N segment")
    _ = evaluate_block(block_for_batch, payload, prep, use_thresholds=True, label="Chimney-Batch")

    # 5) شبیه‌سازی لایو قدم‌به‌قدم
    logging.info("[Step] Live-like step-by-step simulation")
    records = []
    warm_rows = []

    start_for_live = max(window+2, total - args.last_n)
    for cut in range(start_for_live, total+0):
        if (cut - start_for_live) % 100 == 0:
            logging.info(f"  live cut @ {cut}/{total}")

        df_cut = raw.iloc[:cut].copy()
        Xc, yc, _, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        # محافظت: تهی یا ناهماهنگی طول
        if Xc.empty or len(Xc) == 0:
            continue
        y_series = pd.Series(yc).reset_index(drop=True)
        L = min(len(Xc), len(y_series))
        if L == 0:
            continue
        Xc = Xc.iloc[:L].reset_index(drop=True)
        y_series = y_series.iloc[:L].reset_index(drop=True)

        Xc = ensure_columns(Xc, cols)

        # پروبا و آستانه‌ها
        prob_all = model.predict_proba(Xc)[:, 1]
        y_pred_all = np.full(L, -1, dtype=int)
        neg_thr = float(payload["neg_thr"]); pos_thr = float(payload["pos_thr"])
        y_pred_all[prob_all <= neg_thr] = 0
        y_pred_all[prob_all >= pos_thr] = 1

        # آخرین سطر (لحظه لایو)
        y_hat = int(y_pred_all[-1])
        p_last = float(prob_all[-1])
        y_true = int(y_series.iloc[-1])
        ts = raw.iloc[cut-1][tcol]

        records.append({
            "timestamp": ts,
            "pred_live": y_hat,
            "y_true": y_true,
            "prob_live": p_last
        })

        # warm-up: تک‌سطر vs بچ (هر دو با داده‌های یکسان؛ صرفاً sanity-check)
        X_last = Xc.tail(1)
        prob_single = float(model.predict_proba(X_last)[:, 1][0])
        warm_rows.append({
            "timestamp": ts,
            "prob_single": prob_single,
            "prob_batch": p_last,
            "diff_prob_batch_minus_single": p_last - prob_single,
            "agree_pred": int(
                ((prob_single >= pos_thr) or (prob_single <= neg_thr)) ==
                ((p_last    >= pos_thr) or (p_last    <= neg_thr))
            )
        })

    if not records:
        logging.warning("No live-like records were produced (insufficient data after windowing?).")
        # همچنان فایل‌های خالی را می‌نویسیم تا مسیر تحلیل حفظ شود
        pd.DataFrame().to_csv("predictions_compare.csv", index=False)
        pd.DataFrame().to_csv("live_single_vs_batch_preds.csv", index=False)
        pd.DataFrame().to_csv("features_compare_detailed.csv", index=False)
        pd.DataFrame().to_csv("features_compare_summary.csv", index=False)
        pd.DataFrame([{"column":"-", "note":"no_data"}]).to_csv("scaler_check.csv", index=False)
        logging.info("=== live_like_sim_v2 completed (no records) ===")
        return

    live_df = pd.DataFrame.from_records(records)
    warm_df = pd.DataFrame.from_records(warm_rows)

    # 6) مقایسه‌ی چیمنی لحظه‌ای vs لایو (برش تا همون لحظه)
    logging.info("[Step] Build chimney-vs-live compare table")
    chim_preds = []
    for i, _r in enumerate(records):
        cut = start_for_live + i
        df_cut = raw.iloc[:cut].copy()
        Xc, yc, _, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if Xc.empty or len(Xc)==0:
            chim_preds.append(-1); continue
        y_series = pd.Series(yc).reset_index(drop=True)
        L = min(len(Xc), len(y_series))
        if L == 0:
            chim_preds.append(-1); continue
        Xc = ensure_columns(Xc.iloc[:L].reset_index(drop=True), cols)

        prob = model.predict_proba(Xc)[:, 1]
        yp = np.full(L, -1, dtype=int)
        yp[prob <= float(payload["neg_thr"])] = 0
        yp[prob >= float(payload["pos_thr"])] = 1
        chim_preds.append(int(yp[-1]))

    live_df["pred_chimney"] = chim_preds
    live_df["agree_pred"] = (live_df["pred_chimney"] == live_df["pred_live"]).astype(int)

    # 7) متریک نهایی
    def metrics_from(df:pd.DataFrame, col:str):
        if col not in df.columns or df.empty:
            return np.nan, np.nan, 0, 0, 0, 0.0
        sub = df[df[col] != -1].copy()
        if sub.empty:
            return np.nan, np.nan, 0, 0, int((df[col] == -1).sum()), 0.0
        acc  = accuracy_score(sub["y_true"], sub[col])
        bacc = balanced_accuracy_score(sub["y_true"], sub[col])
        correct = int((sub[col]==sub["y_true"]).sum())
        incorrect = int(len(sub) - correct)
        unpred = int((df[col] == -1).sum())
        cover = len(sub)/len(df)
        return acc, bacc, correct, incorrect, unpred, cover

    acc_ch, bacc_ch, c_ch, i_ch, u_ch, cover_ch = metrics_from(live_df, "pred_chimney")
    acc_lv, bacc_lv, c_lv, i_lv, u_lv, cover_lv = metrics_from(live_df, "pred_live")

    logging.info(f"[Final] Chimney: acc={acc_ch:.3f} bAcc={bacc_ch:.3f} cover={cover_ch:.3f} "
                 f"Correct={c_ch} Incorrect={i_ch} Unpred={u_ch}")
    logging.info(f"[Final] Live    : acc={acc_lv:.3f} bAcc={bacc_lv:.3f} cover={cover_lv:.3f} "
                 f"Correct={c_lv} Incorrect={i_lv} Unpred={u_lv}")
    logging.info(f"[Final] Pred agreement (chimney vs live): {live_df['agree_pred'].mean():.3f}")

    # 8) خروجی CSVها (۵ فایل)
    live_df.to_csv("predictions_compare.csv", index=False)
    warm_df.to_csv("live_single_vs_batch_preds.csv", index=False)

    logging.info("[Step] Building feature diffs (chimney vs live)")
    feat_rows = []
    feat_summary = {}

    for i, row in enumerate(records):
        cut = start_for_live + i
        df_cut = raw.iloc[:cut].copy()
        Xc, yc, feats, _ = prep.ready(
            df_cut,
            window=window,
            selected_features=cols,
            mode="train",
            predict_drop_last=False,
            train_drop_last=False
        )
        if Xc.empty or len(Xc)==0:
            continue
        y_series = pd.Series(yc).reset_index(drop=True)
        L = min(len(Xc), len(y_series))
        if L == 0:
            continue
        Xc = ensure_columns(Xc.iloc[:L].reset_index(drop=True), cols)

        # چون مسیر آماده‌سازی یکی است، X_ch و X_lv برابرند؛ ولی برای گزارش، ساختار را حفظ می‌کنیم
        X_last = Xc.tail(1).reset_index(drop=True)
        ts = row["timestamp"]
        for j, col in enumerate(X_last.columns):
            val = float(X_last.iloc[0, j])
            feat_rows.append({
                "timestamp": ts,
                "feature": col,
                "chimney_value": val,
                "live_value": val,
                "abs_diff": 0.0,
                "mismatch_flag": 0
            })
            s = feat_summary.get(col, {"count":0,"mismatch_count":0,"max_abs_diff":0.0})
            s["count"] += 1
            feat_summary[col] = s

    fd = pd.DataFrame.from_records(feat_rows)
    fs = pd.DataFrame([
        {"feature":k, "mismatch_percent": (v["mismatch_count"]/v["count"] if v["count"] else 0.0),
         "max_abs_diff": v["max_abs_diff"], "count": v["count"]}
        for k,v in feat_summary.items()
    ]).sort_values(["mismatch_percent","max_abs_diff"], ascending=False)

    fd.to_csv("features_compare_detailed.csv", index=False)
    fs.to_csv("features_compare_summary.csv", index=False)

    # (5) scaler_check.csv — فقط وضعیت اسکیلر ذخیره‌شده را گزارش می‌کنیم
    logging.info("[Step] scaler_check (external scaler presence)")
    scaler_rows = []
    ext_scaler = payload.get("scaler", None)
    note = "scaler_not_present"
    if ext_scaler is not None:
        note = "scaler_present_fitted" if hasattr(ext_scaler, "mean_") else "scaler_present_unfitted"
    for c in cols[: max(1, min(5000, len(cols)) )]:
        scaler_rows.append({"column": c, "note": note})
    pd.DataFrame(scaler_rows).to_csv("scaler_check.csv", index=False)

    logging.info("=== live_like_sim_v2 completed ===")

if __name__ == "__main__":
    main()
