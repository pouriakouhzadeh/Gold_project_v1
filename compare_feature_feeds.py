# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
compare_feature_feeds.py
ورودی‌ها:
  - sim_X_feed_tail200.csv  (خروجی شبیه‌ساز)
  - deploy_X_feed_tail200.csv (خروجی دپلوی)
خروجی‌ها:
  - features_compare_summary.csv
  - features_compare_detailed.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np

SIM = Path("sim_X_feed_tail200.csv")
DEP = Path("deploy_X_feed_tail200.csv")
OUT_SUM = Path("features_compare_summary.csv")
OUT_DET = Path("features_compare_detailed.csv")

def main():
    if not SIM.exists() or not DEP.exists():
        print("missing input CSVs")
        return

    sim = pd.read_csv(SIM)
    dep = pd.read_csv(DEP)

    # نرمال‌سازی نام ستون timestamp
    for df in (sim, dep):
        if "timestamp" not in df.columns:
            # برخی نسخه‌ها نام ستون را time نوشته‌اند
            tcands = [c for c in df.columns if "time" in c.lower()]
            if tcands:
                df.rename(columns={tcands[0]: "timestamp"}, inplace=True)
    sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")
    dep["timestamp"] = pd.to_datetime(dep["timestamp"], errors="coerce")

    sim = sim.dropna(subset=["timestamp"]).sort_values("timestamp")
    dep = dep.dropna(subset=["timestamp"]).sort_values("timestamp")

    # ستون‌های مشترک (به‌جز timestamp، score، decision)
    skip = {"timestamp", "score", "decision"}
    common = [c for c in sim.columns if c in dep.columns and c not in skip]

    # join بر اساس timestamp برابر
    joined = sim.merge(dep, on="timestamp", suffixes=("_sim","_dep"), how="inner")

    # مقایسهٔ اختلاف‌ها
    rows = []
    for c in common:
        a = joined[f"{c}_sim"].astype(float)
        b = joined[f"{c}_dep"].astype(float)
        diff = (a - b).abs()
        mism = (diff > 1e-12).sum()  # آستانهٔ بسیار سخت
        rows.append({"feature": c, "mismatch_cnt": int(mism), "max_abs_diff": float(diff.max())})

    summary = pd.DataFrame(rows).sort_values(["mismatch_cnt","max_abs_diff"], ascending=[False, False])
    summary.to_csv(OUT_SUM, index=False)

    # گزارش تفصیلی (برای فقط ستون‌هایی که mismatch دارند)
    bads = summary[summary["mismatch_cnt"] > 0]["feature"].tolist()
    det_cols = ["timestamp"] + [f"{c}_sim" for c in bads] + [f"{c}_dep" for c in bads]
    detailed = joined[det_cols].copy()
    detailed.to_csv(OUT_DET, index=False)

    print(f"Wrote {OUT_SUM} and {OUT_DET}")

if __name__ == "__main__":
    main()
