# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
generating_data_for_predict_script.py

ژنراتور نقش MT4 را بازی می‌کند: در هر استپ CSVهای زنده را تا ts_now می‌نویسد
و بی‌نهایت منتظر answer.txt می‌ماند، سپس با استفاده از timestamp فیدلاگ
(= ts_feat واقعی) برچسب را محاسبه و آمار تجمعی ارائه می‌دهد.
"""

from __future__ import annotations
import time, logging, argparse
from pathlib import Path
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def resolve_raw_paths(base: Path, symbol: str) -> dict[str,str]:
    return {
        "30T": str(base / f"{symbol}_M30.csv"),
        "15T": str(base / f"{symbol}_M15.csv"),
        "5T" : str(base / f"{symbol}_M5.csv"),
        "1H" : str(base / f"{symbol}_H1.csv"),
    }

def live_name(path_str: str) -> str:
    p = Path(path_str)
    return str(p.with_name(p.stem + "_live" + p.suffix))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--symbol", default="XAUUSD", type=str)
    ap.add_argument("--last-n", default=200, type=int)
    ap.add_argument("--sleep", default=0.5, type=float)
    args = ap.parse_args()

    setup_logging()
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
    start_idx = max(0, total - (n + 1))  # نیاز به t و t+1

    SL = {"30T":1000,"15T":2000,"5T":5000,"1H":1000}
    ans_path = Path("answer.txt")
    feed_log = Path("deploy_X_feed_log.csv")

    wins=loses=none=0

    for step, idx in enumerate(range(start_idx, start_idx+n), start=1):
        ts_now = df30.loc[idx, "time"]
        # نوشتن CSVهای زنده تا ts_now
        for tf, df in raw.items():
            cut = df[df["time"] <= ts_now].tail(SL.get(tf,500)).copy()
            Path(live_name(paths[tf])).write_text("")  # پاکسازی فایل خروجی
            cut.to_csv(live_name(paths[tf]), index=False)

        logging.info(f"[Step {step}/{n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        # بی‌نهایت صبر برای answer.txt
        while not ans_path.exists():
            time.sleep(args.sleep)
        try:
            ans = ans_path.read_text(encoding="utf-8").strip().upper() or "NONE"
        except Exception:
            ans = "NONE"
        ans_path.unlink(missing_ok=True)

        # خواندن آخرین timestamp از فیدلاگ (ts_feat واقعی)
        try:
            dflog = pd.read_csv(feed_log, parse_dates=["timestamp"])
            ts_feat = pd.to_datetime(dflog["timestamp"].iloc[-1])
        except Exception:
            ts_feat = ts_now  # fallback

        # محاسبهٔ برچسب حقیقی بر اساس ts_feat
        try:
            ridx_list = df30.index[df30["time"] == ts_feat].tolist()
            if not ridx_list:
                real_up = None
            else:
                ridx = ridx_list[0]
                c0 = float(df30.loc[ridx, "close"])
                c1 = float(df30.loc[ridx+1, "close"])
                real_up = (c1 > c0)
        except Exception:
            real_up = None

        pred_up = None
        if ans == "BUY":
            pred_up = True
        elif ans == "SELL":
            pred_up = False

        if pred_up is None or real_up is None:
            none += 1
        else:
            if pred_up == real_up:
                wins += 1
            else:
                loses += 1

        pred_n = wins + loses
        cover = pred_n / float(step)
        acc = (wins / pred_n) if pred_n > 0 else 0.0
        logging.info(f"[Step {step}] cover={cover:.3f} acc={acc:.3f} wins={wins} loses={loses} none={none}")

    logging.info(f"[Final] acc={acc:.3f} cover={cover:.3f} wins={wins} loses={loses} none={none}")

if __name__ == "__main__":
    main()
