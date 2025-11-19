#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, logging
import numpy as np
import pandas as pd

LOGFMT = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGFMT, datefmt="%Y-%m-%d %H:%M:%S")
L = logging.getLogger("compare_feeds")

def to_dt(s):
    x = pd.to_datetime(s, errors="coerce", utc=False)
    # نرمال‌سازی به datetime64[ns] بدون timezone
    return pd.DatetimeIndex(x).tz_localize(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-feed", default="sim_X_feed_tail200.csv")
    ap.add_argument("--dep-feed", default="deploy_X_feed_tail200.csv")
    ap.add_argument("--sim-preds", default="sim_predictions.csv")
    ap.add_argument("--dep-preds", default="deploy_predictions.csv")
    args = ap.parse_args()

    # ---------- 1) مقایسهٔ فیچر ----------
    sim = pd.read_csv(args.sim_feed)
    dep = pd.read_csv(args.dep_feed)
    if "timestamp" not in sim.columns or "timestamp" not in dep.columns:
        raise RuntimeError("Both feeds must have 'timestamp' column")
    sim["timestamp"] = to_dt(sim["timestamp"])
    dep["timestamp"] = to_dt(dep["timestamp"])

    # نام ستون‌های مشترک (به‌جز timestamp و y_true)
    common_cols = sorted(list(set(sim.columns) & set(dep.columns) - {"timestamp"}))
    joined = sim[["timestamp"] + common_cols].merge(
        dep[["timestamp"] + common_cols],
        on="timestamp", suffixes=("_sim","_dep"), how="inner"
    )
    L.info("Joined rows (features): %d", len(joined))

    # detailed
    det_rows = []
    for c in common_cols:
        d = (joined[f"{c}_sim"] - joined[f"{c}_dep"]).astype(float)
        tmp = pd.DataFrame({"timestamp": joined["timestamp"], "feature": c, "abs_diff": d.abs().values})
        det_rows.append(tmp)
    detailed = pd.concat(det_rows, ignore_index=True)
    detailed.sort_values(["feature","timestamp"], inplace=True)
    detailed.to_csv("features_compare_detailed.csv", index=False)

    # summary
    summ = (detailed
            .groupby("feature")["abs_diff"]
            .agg(max_abs="max", mean_abs="mean", median_abs="median"))
    # شمارش مواردی که اختلاف از 1e-9 بزرگ‌تر است
    mis = (detailed
           .assign(mis=(detailed["abs_diff"].abs() > 1e-9).astype(int))
           .groupby("feature")["mis"].sum()
           .rename("mismatch_cnt"))
    summary = summ.join(mis, how="left").sort_values("mismatch_cnt", ascending=False)
    summary.to_csv("features_compare_summary.csv", index=True)

    # ---------- 2) مقایسهٔ خروجی/تارگت ----------
    if os.path.exists(args.sim_preds) and os.path.exists(args.dep_preds):
        sp = pd.read_csv(args.sim_preds)
        dp = pd.read_csv(args.dep_preds)
        if "timestamp" in sp.columns and "timestamp" in dp.columns:
            sp["timestamp"] = to_dt(sp["timestamp"])
            dp["timestamp"] = to_dt(dp["timestamp"])
            cp = sp.merge(dp, on="timestamp", suffixes=("_sim","_dep"), how="inner")
            # تبدیل اکشن به کد
            def map_act(s):
                m = {"BUY":1, "SELL":0, "NONE":-1}
                return s.map(m).astype("int64", errors="ignore") if isinstance(s, pd.Series) else m.get(s, -1)
            cp["action_code_sim"] = map_act(cp["action_sim"].astype(str))
            cp["action_code_dep"] = map_act(cp["action_dep"].astype(str))
            cp["y_prob_diff"] = (cp["y_prob_sim"] - cp["y_prob_dep"]).abs()
            cp["y_true_mis"]  = (cp["y_true"] if "y_true" in cp.columns else np.nan)
            cp["action_mismatch"] = (cp["action_code_sim"] != cp["action_code_dep"]).astype(int)
            cp.to_csv("predictions_compare.csv", index=False)

            # گزارش کوتاه
            act_mis = int(cp["action_mismatch"].sum())
            both = len(cp)
            L.info("[Pred-Compare] rows=%d action_mismatch=%d (%.1f%%)",
                   both, act_mis, 100.0*act_mis/max(1,both))
        else:
            L.warning("Missing 'timestamp' in predictions CSVs; skipped preds compare.")

if __name__ == "__main__":
    main()
