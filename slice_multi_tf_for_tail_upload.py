
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slice 4x timeframe CSVs to the minimal tail needed for backtesting 4000 main (30T) predictions,
while keeping lower/higher TFs aligned and including warm-up for windowing & indicators.

- Reads `best_model.pkl` (ModelSaver payload) to get window_size and feature lists.
- Uses PREPARE_DATA_FOR_TRAIN to confirm shapes and, if requested, auto-tunes warm-up.
- Produces smaller CSVs you can ZIP and upload.

Usage:
  python slice_multi_tf_for_tail_upload.py \
      --root /path/to/project \
      --main-tf 30T \
      --N-main 4000 \
      --out out_sliced \
      --auto-warmup \
      --min-warmup-rows-5T 1000 --min-warmup-rows-15T 500 --min-warmup-rows-30T 250 --min-warmup-rows-1H 150

Notes:
- Default CSV names follow your project: XAUUSD_M5.csv, XAUUSD_M15.csv, XAUUSD_M30.csv, XAUUSD_H1.csv
- If your filenames differ, use --map-* flags to provide custom paths.
"""
from __future__ import annotations
import argparse, json, sys, math, logging
from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
import joblib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Project root where CSVs and prepare_data_for_train.py reside.")
    p.add_argument("--model", default="best_model.pkl", help="ModelSaver artifact to infer window_size etc.")
    p.add_argument("--main-tf", default="30T", choices=["5T","15T","30T","1H"])
    p.add_argument("--N-main", type=int, default=4000, help="How many main-TF predictions needed (default: 4000).")
    p.add_argument("--out", default="out_sliced", help="Output folder for sliced CSVs.")
    p.add_argument("--auto-warmup", action="store_true", help="Auto-adjust start to avoid NaNs in final tail.")
    # custom file mappings
    p.add_argument("--map-5T", default="XAUUSD_M5.csv")
    p.add_argument("--map-15T", default="XAUUSD_M15.csv")
    p.add_argument("--map-30T", default="XAUUSD_M30.csv")
    p.add_argument("--map-1H", default="XAUUSD_H1.csv")
    # minimum warmup (rows) per TF (sensible defaults; tune if needed)
    p.add_argument("--min-warmup-rows-5T", type=int, default=1200)
    p.add_argument("--min-warmup-rows-15T", type=int, default=600)
    p.add_argument("--min-warmup-rows-30T", type=int, default=300)
    p.add_argument("--min-warmup-rows-1H", type=int, default=180)
    return p.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    root = Path(args.root)
    out  = Path(args.out); ensure_dir(out)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("slice")

    # Load model for window_size, feats, cols (if exists)
    model_path = root / args.model
    if not model_path.exists():
        log.warning("Model artifact not found at %s — proceeding without it (window_size=1).", model_path)
        saved = {}
        window_size = 1
    else:
        saved = joblib.load(model_path)
        window_size = int(saved.get("window_size", 1))
        log.info("Loaded %s (window_size=%d)", model_path.name, window_size)

    # Load raw CSVs
    paths = {
        "5T":  root / args.map_5T,
        "15T": root / args.map_15T,
        "30T": root / args.map_30T,
        "1H":  root / args.map_1H,
    }
    for tf, pth in paths.items():
        if not pth.exists():
            log.error("CSV missing for TF %s: %s", tf, pth)
            sys.exit(1)

    dfs = {}
    for tf, pth in paths.items():
        df = pd.read_csv(pth)
        if "time" not in df.columns:
            log.error("Column 'time' missing in %s", pth)
            sys.exit(1)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        dfs[tf] = df

    # Determine per-TF counts matching N_main (assuming main=30T by default ratios)
    # Ratios: relative to 30T → 5T=6x, 15T=2x, 30T=1x, 1H=0.5x (ceil for safety)
    ratios = {"5T": 6.0, "15T": 2.0, "30T": 1.0, "1H": 0.5}
    N = {}
    for tf in ["30T","15T","5T","1H"]:
        if args.main_tf == "30T":
            N[tf] = math.ceil(args.N_main * ratios[tf])
        elif args.main_tf == "15T":
            factor = args.N_main / 2.0
            N[tf] = math.ceil(factor * ratios[tf])
        elif args.main_tf == "5T":
            factor = args.N_main / 6.0
            N[tf] = math.ceil(factor * ratios[tf])
        elif args.main_tf == "1H":
            factor = args.N_main / 0.5
            N[tf] = math.ceil(factor * ratios[tf])

    # Warm-up rows per TF (minimums + window_size)
    min_warm = {
        "5T":  max(args.min_warmup_rows_5T, window_size * 6),
        "15T": max(args.min_warmup_rows_15T, window_size * 2),
        "30T": max(args.min_warmup_rows_30T, window_size * 1),
        "1H":  max(args.min_warmup_rows_1H,  window_size // 2 or 1),
    }

    log.info("Target rows: 30T=%d, 15T=%d, 5T=%d, 1H=%d (main=%s, window=%d)",
             N["30T"], N["15T"], N["5T"], N["1H"], args.main_tf, window_size)
    log.info("Warm-up minima: %s", min_warm)

    # Compute time window based on main TF tail
    main_df = dfs[args.main_tf if args.main_tf in dfs else "30T"]
    if len(main_df) < (N[args.main_tf] + min_warm[args.main_tf] + 10):
        log.warning("Main TF has fewer rows than requested; using entire available tail.")
    end_time = main_df["time"].iloc[-1]
    start_idx = max(0, len(main_df) - (N[args.main_tf] + min_warm[args.main_tf]))
    start_time = main_df["time"].iloc[start_idx]

    # Slice each TF by time range, but extend warm-up per TF
    sliced = {}
    for tf, df in dfs.items():
        if tf == args.main_tf:
            tf_start_time = start_time
        else:
            # conservative extension for faster/slower TFs
            tf_start_time = start_time - pd.Timedelta(minutes=30 if tf=="5T" else 15 if tf=="15T" else 60)
        mask = (df["time"] >= tf_start_time) & (df["time"] <= end_time)
        sub = df.loc[mask].copy()
        # ensure at least N[tf] + min_warm[tf]
        need = N[tf] + min_warm[tf]
        if len(sub) < need:
            sub = df.tail(need).copy()
        sliced[tf] = sub.reset_index(drop=True)
        log.info("TF %s sliced to %d rows (need %d)", tf, len(sub), need)

    # Optional: auto-warmup adjust — run a quick PREP pass on temp slices
    if args.auto_warmup:
        sys.path.insert(0, str(root))
        try:
            from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
            prep = PREPARE_DATA_FOR_TRAIN(
                filepaths={
                    "30T": str(out / "XAUUSD_M30_tmp.csv"),
                    "1H":  str(out / "XAUUSD_H1_tmp.csv"),
                    "15T": str(out / "XAUUSD_M15_tmp.csv"),
                    "5T":  str(out / "XAUUSD_M5_tmp.csv"),
                },
                main_timeframe=args.main_tf,
                verbose=False
            )
            # write temporary files
            for tf, name in [("30T","XAUUSD_M30_tmp.csv"),("1H","XAUUSD_H1_tmp.csv"),
                             ("15T","XAUUSD_M15_tmp.csv"),("5T","XAUUSD_M5_tmp.csv")]:
                sliced[tf].to_csv(out / name, index=False)
            raw = prep.load_data()
            X_all, y_all, _, _ = prep.ready(raw, window=window_size, selected_features=None, mode="train")
            X_all = X_all.replace([np.inf,-np.inf], np.nan)
            if X_all.tail(N[args.main_tf if args.main_tf in N else "30T"]).isna().any().any():
                logging.warning("Auto-warmup detected NaNs on last segment; expanding tails conservatively.")
                for tf in ["5T","15T","30T","1H"]:
                    df = dfs[tf]
                    need = len(sliced[tf]) + min_warm[tf]//2
                    sliced[tf] = df.tail(need).reset_index(drop=True)
        except Exception as e:
            logging.warning("Auto-warmup failed (%s). Proceeding with conservative slices.", e)

    # Save final slices with project filenames
    out_map = {
        "5T":  "XAUUSD_M5.csv",
        "15T": "XAUUSD_M15.csv",
        "30T": "XAUUSD_M30.csv",
        "1H":  "XAUUSD_H1.csv",
    }
    for tf, name in out_map.items():
        path = out / name
        sliced[tf].to_csv(path, index=False)
        log.info("Saved %s (%d rows)", path.name, len(sliced[tf]))

    print("Done. Upload the folder:", out)

if __name__ == "__main__":
    main()

