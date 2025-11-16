# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
generating_data_for_predict_script.py

این اسکریپت نقش MT4 را بازی می‌کند: در هر گام داده‌های live را می‌نویسد و
منتظر جواب dploy می‌ماند. سپس برای محاسبه‌ی برچسب، زمان دقیق پیش‌بینی (ts_feat)
را از deploy_X_feed_log.csv می‌خواند و جهت واقعی بازار را بر همان اساس می‌سنجد.
"""

from __future__ import annotations
import os, sys, time, logging, argparse
from pathlib import Path
import pandas as pd

LOG_FILE = "generation_data_for_predict.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger

def resolve_raw_paths(base: Path, symbol: str) -> dict[str,str]:
    cands = {
        "30T":[f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T":[f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" :[f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" :[f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    out = {}
    for tf,names in cands.items():
        found = None
        for nm in names:
            p = base / nm
            if p.exists(): found = str(p); break
        if found is None:
            for ch in base.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found is None:
            raise FileNotFoundError(f"[{tf}] not found in {base}")
        out[tf] = found
        logging.info("[raw] %s -> %s", tf, found)
    return out

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=200)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()
    setup_logging(args.verbosity)
    logging.info("=== Generator started ===")

    base = Path(args.base_dir).resolve()
    paths = resolve_raw_paths(base, args.symbol)
    raw = {tf: pd.read_csv(p) for tf,p in paths.items()}
    for tf, df in raw.items():
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        raw[tf] = df

    df30 = raw["30T"]
    total = len(df30)
    n = min(args.last_n, total-2)
    start_idx = max(0, total - (n + 1))  # need t and t+1

    # buffer sizes for writing live CSVs
    SL = {"30T":1000,"15T":2000,"5T":5000,"1H":1000}
    ans_path = Path("answer.txt")
    feed_log = Path("deploy_X_feed_log.csv")

    wins=loses=none=0
    audit_rows = []

    for step, idx in enumerate(range(start_idx, start_idx+n), start=1):
        ts_now = df30.loc[idx, "time"]
        # write live CSVs up to ts_now
        for tf, df in raw.items():
            cut = df[df["time"] <= ts_now].tail(SL.get(tf,500)).copy()
            Path(live_name(paths[tf])).write_text("")  # ensure file is cleared
            cut.to_csv(live_name(paths[tf]), index=False)

        logging.info(f"[Step {step}/{n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        # wait until answer.txt appears
        while not ans_path.exists():
            time.sleep(args.sleep)
        try:
            ans = ans_path.read_text(encoding="utf-8").strip().upper() or "NONE"
        except Exception:
            ans = "NONE"
        # remove answer.txt for next iteration
        ans_path.unlink(missing_ok=True)

        # read last timestamp from deploy_X_feed_log.csv (ts_feat)
        try:
            dflog = pd.read_csv(feed_log, parse_dates=["timestamp"])
            ts_feat = pd.to_datetime(dflog["timestamp"].iloc[-1])
        except Exception:
            ts_feat = ts_now  # fallback

        # compute real_up based on ts_feat (not ts_now)
        try:
            # find row index in 30T where time == ts_feat
            ridx = df30.index[df30["time"] == ts_feat].tolist()
            if not ridx:
                real_up = None
            else:
                ridx = ridx[0]
                c0 = float(df30.loc[ridx, "close"])
                c1 = float(df30.loc[ridx+1, "close"])
                real_up = (c1 > c0)
        except Exception:
            real_up = None

        pred_up = None
        if ans == "BUY": pred_up = True
        elif ans == "SELL": pred_up = False

        if pred_up is None or real_up is None:
            none += 1
            hit = None
        else:
            hit = (pred_up == real_up)
            if hit: wins += 1
            else: loses += 1

        pred_n = wins + loses
        cover = pred_n / float(step)
        acc = (wins / pred_n) if pred_n > 0 else 0.0
        logging.info(f"[Step {step}] cover={cover:.3f} acc={acc:.3f} wins={wins} loses={loses} none={none}")

        audit_rows.append({
            "timestamp_sim": ts_now,
            "ts_pred": ts_feat,
            "answer": ans,
            "real_up": real_up,
            "hit": hit,
            "acc_cum": acc,
            "cover_cum": cover
        })

    pred_n = wins + loses
    cover = pred_n / max(1, n)
    acc = (wins / pred_n) if pred_n > 0 else 0.0
    logging.info(f"[Final] acc={acc:.3f} cover={cover:.3f} wins={wins} loses={loses} none={none}")
    pd.DataFrame(audit_rows).to_csv("generator_audit.csv", index=False)

if __name__ == "__main__":
    main()
