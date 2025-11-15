# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, logging, argparse
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

def resolve_raw_paths(base_dir: Path, symbol: str) -> dict[str,str]:
    cands = {
        "30T": [f"{symbol}_30T.csv", f"{symbol}_M30.csv"],
        "15T": [f"{symbol}_15T.csv", f"{symbol}_M15.csv"],
        "5T" : [f"{symbol}_5T.csv",  f"{symbol}_M5.csv" ],
        "1H" : [f"{symbol}_1H.csv",  f"{symbol}_H1.csv" ],
    }
    res = {}
    for tf, names in cands.items():
        found = None
        for nm in names:
            p = base_dir / nm
            if p.exists(): found = str(p); break
        if not found:
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    found = str(ch); break
        if found:
            logging.info(f"[raw] {tf} -> {found}")
            res[tf] = found
    if "30T" not in res:
        raise FileNotFoundError("30T file not found.")
    return res

def pick_slice_sizes() -> dict[str,int]:
    # حدودی که MT4 به‌صورت رایج در اختیار قرار می‌دهد
    return {"30T": 500, "15T": 1000, "5T": 3000, "1H": 300}

def live_name(path: str) -> str:
    p = Path(path)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=200, help="تعداد رکوردهای انتهایی برای سنجش (پیشنهادی: 200)")
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.25, help="وقفه‌ی polling (ثانیه)")
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== Generator started ===")

    base = Path(args.base_dir).resolve()
    paths = resolve_raw_paths(base, args.symbol)
    raw = {tf: pd.read_csv(paths[tf]) for tf in paths}
    for tf, df in raw.items():
        if "time" not in df.columns:
            raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce"); df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True); df.reset_index(drop=True, inplace=True)

    df30 = raw["30T"]
    total = len(df30)
    if total < args.last_n + 2:
        logging.warning("Dataset is small vs last_n; proceeding anyway.")

    SL = pick_slice_sizes()
    wins = loses = none = 0
    audit_rows = []
    start_idx = max(1, total - args.last_n)
    ans_path = base / "answer.txt"

    for i, idx in enumerate(range(start_idx, total-1), start=1):
        ts_now = pd.Timestamp(df30.loc[idx, "time"])
        # 1) write live CSV slices (up to ts_now)
        for tf in raw.keys():
            k = SL.get(tf, 500)
            df = raw[tf]
            df_cut = df[df["time"] <= ts_now].tail(k).copy()
            out_path = live_name(paths[tf])
            df_cut.to_csv(out_path, index=False)

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        # 2) wait for prediction then read and DELETE
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"
        try:
            ans_path.unlink(missing_ok=True)  # مهم: حتماً حذف شود تا مرحله بعد گیر نکند
        except Exception:
            pass

        # 3) compute ground-truth from 30T (direction t→t+1)
        c0 = float(df30.loc[idx, "close"]); c1 = float(df30.loc[idx+1, "close"])
        real_up = (c1 > c0)  # True=BUY, False=SELL

        pred_up = None
        if ans == "BUY":
            pred_up = True
        elif ans == "SELL":
            pred_up = False

        if pred_up is None:
            none += 1
            hit = None
        else:
            hit = (pred_up == real_up)
            if hit: wins += 1
            else:   loses += 1

        audit_rows.append({
            "timestamp": ts_now,
            "answer": ans,
            "real_up": real_up,
            "hit": hit,
            "close_t": c0,
            "close_t1": c1
        })

    # 4) final metrics
    pred_n = wins + loses
    cover = pred_n / max(1, args.last_n)
    acc = (wins / pred_n) if pred_n > 0 else 0.0
    logging.info(f"[Final] acc={acc:.3f} cover={cover:.3f} wins={wins} loses={loses} none={none}")
    pd.DataFrame(audit_rows).to_csv("generator_audit.csv", index=False)

if __name__ == "__main__":
    main()
