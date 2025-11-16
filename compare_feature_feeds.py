# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

def load_meta_cols(meta_path="best_model.meta.json") -> list[str]:
    cols = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cols = meta.get("train_window_cols") or meta.get("feats") or []
    return list(cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_X_feed_tail200.csv")
    ap.add_argument("--dep", type=str, default="deploy_X_feed_log.csv")
    ap.add_argument("--out-summary", type=str, default="features_compare_summary.csv")
    ap.add_argument("--out-detailed", type=str, default="features_compare_detailed.csv")
    args = ap.parse_args()

    # read
    sim = pd.read_csv(args.sim)
    dep = pd.read_csv(args.dep)

    # normalize timestamp dtype
    sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")
    dep["timestamp"] = pd.to_datetime(dep["timestamp"], errors="coerce")
    sim = sim.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    dep = dep.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # columns intersection based on training window columns
    train_cols = load_meta_cols("best_model.meta.json")
    if not train_cols:
        # fallback: all except timestamp/score/decision
        train_cols = [c for c in sim.columns if c != "timestamp"]
    feat_cols = [c for c in train_cols if c in sim.columns and c in dep.columns]

    # inner-join on timestamp to get exactly overlapping rows
    joined = sim[["timestamp"] + feat_cols].merge(
        dep[["timestamp"] + feat_cols],
        on="timestamp", suffixes=("_sim","_dep"), how="inner"
    )
    if joined.empty:
        print("No overlapping timestamps; nothing to compare.")
        return

    # build detailed rows per feature per timestamp
    detailed_rows = []
    for feat in feat_cols:
        a = joined[f"{feat}_sim"].astype(float)
        b = joined[f"{feat}_dep"].astype(float)
        diff = a - b
        rel = diff / (np.where(np.abs(a)>1e-12, np.abs(a), 1.0))
        match = ( (a.isna() & b.isna()) | (np.isclose(a, b, atol=1e-9, rtol=1e-6)) )
        df_feat = pd.DataFrame({
            "timestamp": joined["timestamp"],
            "feature": feat,
            "sim": a, "dep": b,
            "diff": diff,
            "rel_diff": rel,
            "match": match
        })
        detailed_rows.append(df_feat)

    detailed = pd.concat(detailed_rows, ignore_index=True)
    detailed.sort_values(["feature","timestamp"], inplace=True)
    detailed.to_csv(args.out_detailed, index=False)

    # summary per feature
    summary_rows = []
    for feat in feat_cols:
        df_f = detailed[detailed["feature"]==feat]
        rows = len(df_f)
        mismatch_cnt = int((~df_f["match"]).sum())
        nan_sim = int(df_f["sim"].isna().sum())
        nan_dep = int(df_f["dep"].isna().sum())
        nan_both = int((df_f["sim"].isna() & df_f["dep"].isna()).sum())
        mad = float(np.nanmean(np.abs(df_f["diff"]))) if rows else 0.0
        med = float(np.nanmedian(np.abs(df_f["diff"]))) if rows else 0.0
        mx = float(np.nanmax(np.abs(df_f["diff"]))) if rows else 0.0
        try:
            corr = float(np.corrcoef(
                np.nan_to_num(df_f["sim"].values, nan=0.0),
                np.nan_to_num(df_f["dep"].values, nan=0.0)
            )[0,1])
        except Exception:
            corr = np.nan
        rmse = float(np.sqrt(np.nanmean((df_f["diff"])**2))) if rows else 0.0

        summary_rows.append({
            "feature": feat,
            "rows": rows,
            "mismatch_cnt": mismatch_cnt,
            "mismatch_rate": mismatch_cnt / max(1, rows),
            "na_sim": nan_sim,
            "na_dep": nan_dep,
            "na_both": nan_both,
            "mean_abs_diff": mad,
            "median_abs_diff": med,
            "max_abs_diff": mx,
            "corr": corr,
            "rmse": rmse,
            # heuristic flags
            "flag_time_offset_like": False,
            "flag_scale_like": False,
            "flag_lag_like": False,
            "flag_nan_mismatch": (nan_sim != nan_dep)
        })

    summary = pd.DataFrame(summary_rows).sort_values(["mismatch_rate","rmse","feature"], ascending=[False,False,True])
    summary.to_csv(args.out_summary, index=False)
    print(f"Wrote: {args.out_summary} & {args.out_detailed}")

if __name__ == "__main__":
    main()
