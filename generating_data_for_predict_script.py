#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generating_data_for_predict_script.py  —  ژنراتور هماهنگ با دپلوی

- ساخت ts_list و y_true با استفاده از PREPARE_DATA_FOR_TRAIN.ready
- نوشتن XAUUSD_*_live.csv تا هر timestamp
- هندشیک فقط از طریق deploy_predictions.csv
- لاگ کامل cover و accuracy و ذخیرهٔ generator_predictions.csv
"""

from __future__ import annotations
import json, time, argparse, logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore

LOG = logging.getLogger("generator_parity")


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity > 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_meta(base_dir: Path) -> Tuple[dict, list, int]:
    meta_path = base_dir / "best_model.meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"best_model.meta.json not found in {base_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    window = int(meta.get("window_size", 1))
    train_cols = (
        meta.get("train_window_cols")
        or meta.get("feats")
        or meta.get("feature_names")
        or []
    )
    return meta, list(train_cols), window


def write_live_csvs(
    base_dir: Path,
    symbol: str,
    ts: pd.Timestamp,
    raw_paths: Dict[str, Path],
) -> None:
    """برای هر TF تا لحظهٔ ts فایل *_live.csv می‌نویسد."""
    tf_map = {"30T": "M30", "15T": "M15", "5T": "M5", "1H": "H1"}
    for tf, fp in raw_paths.items():
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            raise ValueError(f"'time' column not found in {fp}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[df["time"] <= ts].copy()
        out = base_dir / f"{symbol}_{tf_map[tf]}_live.csv"
        df.to_csv(out, index=False)


def wait_for_deploy_row(
    base_dir: Path,
    ts: pd.Timestamp,
    poll_sec: float = 0.5,
) -> Tuple[str, float, float]:
    """
    منتظر می‌ماند تا در deploy_predictions.csv ردیفی با timestamp == ts پیدا شود.
    """
    pred_path = base_dir / "deploy_predictions.csv"
    while True:
        if pred_path.is_file():
            try:
                df = pd.read_csv(pred_path)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], errors="coerce"
                    )
                    hit = df.loc[df["timestamp"] == ts]
                    if not hit.empty:
                        last = hit.iloc[-1]
                        act = str(last["action"]).upper()
                        y_prob = float(last.get("y_prob", np.nan))
                        cover_cum = float(last.get("cover_cum", np.nan))
                        return act, y_prob, cover_cum
            except Exception:
                pass
        time.sleep(poll_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--last-n", default=200, type=int)
    ap.add_argument("--verbosity", default=1, type=int)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base_dir = Path(args.base_dir).resolve()

    LOG.info("=== Generator started ===")

    meta, train_cols, window = load_meta(base_dir)
    LOG.info(
        "Meta loaded | window=%d | train_cols=%d",
        window,
        len(train_cols),
    )

    raw_paths: Dict[str, Path] = {
        "30T": base_dir / f"{args.symbol}_M30.csv",
        "15T": base_dir / f"{args.symbol}_M15.csv",
        "5T":  base_dir / f"{args.symbol}_M5.csv",
        "1H":  base_dir / f"{args.symbol}_H1.csv",
    }
    for tf, fp in raw_paths.items():
        LOG.info("[raw] %s -> %s", tf, fp)

    # --- PREPARE_DATA_FOR_TRAIN برای ساخت mapping زمان → y_true ---
    prep = PREPARE_DATA_FOR_TRAIN(
        filepaths={k: str(v) for k, v in raw_paths.items()},
        main_timeframe="30T",
        verbose=(args.verbosity > 0),
    )
    merged = prep.load_data()
    LOG.info("[load_data] merged shape=%s", merged.shape)

    X_all, y_all, _, price_ser, t_idx = prep.ready(
        merged,
        window=window,
        selected_features=train_cols,
        mode="train",
        with_times=True,
        predict_drop_last=False,
        train_drop_last=True,
    )
    if X_all.empty:
        LOG.error("X_all is empty; cannot run generator test.")
        return

    time_series = pd.to_datetime(pd.Series(t_idx).reset_index(drop=True))
    y_series = pd.Series(y_all).reset_index(drop=True).astype("int64")

    if len(time_series) < args.last_n:
        args.last_n = len(time_series)
    ts_list = time_series.tail(args.last_n).reset_index(drop=True)

    LOG.info("Testing last %d steps (timestamps) ...", len(ts_list))

    gen_pred_path = base_dir / "generator_predictions.csv"
    if gen_pred_path.is_file():
        gen_pred_path.unlink()

    wins = loses = none = 0
    total_steps = len(ts_list)

    for i, ts in enumerate(ts_list, start=1):
        # 1) نوشتن فایل‌های live تا این timestamp
        write_live_csvs(base_dir, args.symbol, ts, raw_paths)
        LOG.info(
            "[Step %d/%d] Live CSVs written for ts=%s — waiting for deploy ...",
            i,
            total_steps,
            ts,
        )

        # 2) انتظار تا دپلوی جواب این timestamp را در deploy_predictions.csv بنویسد
        act, y_prob, cover_cum = wait_for_deploy_row(base_dir, ts, poll_sec=0.5)

        # 3) y_true را از mapping زمان → y می‌گیریم
        mask = (time_series == ts)
        if mask.any():
            y_true = int(y_series[mask].iloc[-1])
        else:
            y_true = np.nan

        # 4) به‌روزرسانی آمار
        if act == "NONE":
            none += 1
        elif not pd.isna(y_true):
            if (act == "BUY" and y_true == 1) or (act == "SELL" and y_true == 0):
                wins += 1
            else:
                loses += 1

        cover = (wins + loses) / max(1, i)
        acc = wins / max(1, wins + loses) if (wins + loses) > 0 else 0.0

        LOG.info(
            "[Step %d] ts=%s act=%s y_true=%s | cover=%.3f acc=%.3f wins=%d loses=%d none=%d",
            i,
            ts,
            act,
            str(y_true),
            cover,
            acc,
            wins,
            loses,
            none,
        )

        # 5) ذخیرهٔ رکورد این استپ
        row = {
            "timestamp": ts,
            "action": act,
            "y_true": y_true,
            "y_prob": y_prob,
            "cover_cum": cover_cum,
        }
        hdr = not gen_pred_path.is_file()
        pd.DataFrame([row]).to_csv(
            gen_pred_path,
            mode="a",
            header=hdr,
            index=False,
        )

    LOG.info(
        "[Final] acc=%.3f cover=%.3f wins=%d loses=%d none=%d",
        wins / max(1, wins + loses),
        (wins + loses) / max(1, total_steps),
        wins,
        loses,
        none,
    )


if __name__ == "__main__":
    main()
