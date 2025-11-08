# generate_data_for_pridict_script.py
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, pickle, logging, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

LOG_FILE = "prediction.log"

def setup_logging(verbosity:int=1):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbosity<=1 else logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt); logger.addHandler(ch)
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
        for nm in names:
            p = base_dir / nm
            if p.exists():
                res[tf] = str(p); break
        if tf not in res:
            # case-insensitive fallback
            for ch in base_dir.iterdir():
                if ch.is_file() and any(ch.name.lower()==nm.lower() for nm in names):
                    res[tf] = str(ch); break
        if tf in res:
            logging.info(f"[raw] {tf} -> {res[tf]}")
    if "30T" not in res:
        raise FileNotFoundError("30T file not found. This generator MUST have 30T as base.")
    return res

def pick_slice_sizes() -> dict[str,int]:
    return {"30T": 500, "15T": 1000, "5T": 3000, "1H": 300}

def live_name(path: str, out_dir: Path|None=None) -> str:
    p = Path(path)
    nm = p.stem + "_live" + p.suffix
    return str((out_dir if out_dir else p.parent) / nm)

def parse_required_tfs_from_model(model_pkl: Path) -> set[str]:
    if not model_pkl.exists(): return set()
    try:
        import joblib
        raw = joblib.load(model_pkl)
        cols = raw.get("train_window_cols") or raw.get("feats") or []
        tfs = set()
        for c in cols:
            m = re.match(r"^(30T|15T|5T|1H)_", c)
            if m: tfs.add(m.group(1))
        return tfs
    except Exception:
        return set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=2000)
    ap.add_argument("--base-dir", type=str, default=".")
    ap.add_argument("--symbol", type=str, default="XAUUSD")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--require-tfs-from-model", action="store_true")
    ap.add_argument("--model-path", type=str, default="best_model.pkl")
    args = ap.parse_args()

    setup_logging(1)
    logging.info("=== Generator started ===")

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_raw_paths(base, args.symbol)

    required_tfs = set(["30T"])
    if args.require_tfs_from_model:
        need = parse_required_tfs_from_model(Path(args.model_path))
        if need:
            required_tfs |= need
    missing_tfs = [tf for tf in required_tfs if tf not in paths]
    if missing_tfs:
        logging.error(f"Required TFs missing: {missing_tfs}. Provide their CSVs or disable --require-tfs-from-model.")
        sys.exit(2)

    raw = {tf: pd.read_csv(paths[tf]) for tf in paths}
    for tf, df in raw.items():
        if "time" not in df.columns: raise ValueError(f"[{tf}] 'time' column missing.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df.dropna(subset=["time"], inplace=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    df30 = raw["30T"]
    total = len(df30)
    if total < args.last_n + 2:
        logging.warning("Dataset is small vs last_n; proceeding anyway.")

    SL = pick_slice_sizes()
    wins = loses = none = 0
    start_idx = max(1, total - args.last_n)

    for i, idx in enumerate(range(start_idx, total), start=1):
        ts_now = df30.loc[idx, "time"]

        for tf, df in raw.items():
            k = SL.get(tf, 500)
            df_cut = df[df["time"] <= ts_now].tail(k).copy()
            out_path = live_name(paths[tf], out_dir)
            df_cut.to_csv(out_path, index=False)

        logging.info(f"[Step {i}/{args.last_n}] Live CSVs written at {ts_now} — waiting for answer.txt …")

        ans_path = out_dir / "answer.txt"
        while not ans_path.exists():
            time.sleep(args.sleep)

        try:
            ans = (ans_path.read_text(encoding="utf-8").strip() or "NONE").upper()
        except Exception:
            ans = "NONE"

        j = df30.index[df30["time"] == ts_now]
        real_up = None
        if len(j) > 0 and j[0] < len(df30) - 1:
            j = int(j[0]); c0 = float(df30.loc[j, "close"]); c1 = float(df30.loc[j+1, "close"])
            real_up = (c1 > c0)

        if ans == "NONE" or real_up is None:
            none += 1
        elif ans == "BUY":
            wins += 1 if real_up else 0; loses += 0 if real_up else 1
        elif ans == "SELL":
            wins += 1 if (not real_up) else 0; loses += 0 if (not real_up) else 1
        else:
            none += 1

        acc = wins / max(1, wins + loses)
        logging.info(f"[Result {i}] ans={ans} | real_up={real_up} | WINS={wins} LOSES={loses} NONE={none} ACC={acc:.3f}")

        try: ans_path.unlink(missing_ok=True)
        except Exception: pass

        if i >= args.last_n: break

if __name__ == "__main__":
    main()
