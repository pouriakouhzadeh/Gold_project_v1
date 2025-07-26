#!/usr/bin/env python3
"""
simulation_with_snapshot.py
---------------------------
• Re‑plays the *last N rows* of historical candles (default N=50 000).  
• Writes live CSVs, builds X_live snapshot, runs the stored model.  
• Logs everything to stdout **and** simulation_debug.log – with ZERO
  extra prints from prepare_data_for_train.
"""

# ───────────── imports
import argparse, logging, os, sys, time, multiprocessing as mp, io, contextlib
import pandas as pd, joblib
from tqdm import tqdm
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ───────────── constants & paths
MAX_ROWS = 4_999
SLEEP    = 1

LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1 = (
    "XAUUSD.F_M30_live.csv", "XAUUSD.F_M15_live.csv",
    "XAUUSD.F_M5_live.csv",  "XAUUSD.F_H1_live.csv"
)
SNAPSHOT_CSV = "X_live_snapshot.csv"
LOG_FILE     = "simulation_debug.log"

# ╔════════════════════════════════════════════════════════════╗
#  helper utilities
# ╚════════════════════════════════════════════════════════════╝
def rm(path):                       # remove if exists
    if os.path.exists(path): os.remove(path)

def purge():
    for fp in (LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1, SNAPSHOT_CSV):
        rm(fp)

class TqdmHandler(logging.Handler):     # keeps tqdm bar intact
    def emit(self, record):
        from tqdm import tqdm
        tqdm.write(self.format(record))

def setup_log(level=logging.INFO):
    fmt_file = "%(asctime)s [%(levelname)s] %(message)s"
    fmt_cli  = "%(asctime)s [%(levelname)s] %(message)s"
    fh = logging.FileHandler(LOG_FILE, mode="w")
    fh.setFormatter(logging.Formatter(fmt_file, "%Y-%m-%d %H:%M:%S"))
    sh = TqdmHandler()
    sh.setFormatter(logging.Formatter(fmt_cli, "%H:%M:%S"))
    logging.basicConfig(level=level, handlers=[fh, sh])
    logging.info("=== Simulation started ===")

def load_sorted(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True, ignore_index=True)
    return df

def tf_idx(df, ts, tf):
    if tf == "H1":
        ts = ts.replace(minute=0, second=0, microsecond=0)
    elif tf == "M15":
        ts = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
    elif tf == "M5":
        ts = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
    i = df["time"].searchsorted(ts, "left")
    return i if i < len(df) and df.iloc[i]["time"] == ts else -1

def slice_to(df, i, out_path):
    df.iloc[:i + 1].tail(MAX_ROWS).to_csv(out_path, index=False)

def write_csvs(ts, d5, d15, d30, dH1):
    jobs = [
        (d5, ts, "M5", LIVE_M5),
        (d15, ts, "M15", LIVE_M15),
        (d30, ts, "M30", LIVE_M30),
        (dH1, ts, "H1", LIVE_H1)
    ]
    with mp.Pool(4) as pool:
        results = [pool.apply_async(_job, j) for j in jobs]
        pool.close(); pool.join()
        return all(r.get() for r in results)

def _job(df, ts, tf, outp):
    i = tf_idx(df, ts, tf)
    if i == -1:
        logging.warning(f"{tf}: {ts} missing"); return False
    slice_to(df, i, outp); return True

# ── snapshot creator – stdout muted ───────────────────────────
def build_snapshot(saved) -> bool:
    window, feats, cols = saved["window_size"], saved["feats"], saved["train_window_cols"]
    raw_win = saved.get("train_raw_window")

    fp = {"30T": LIVE_M30, "1H": LIVE_H1, "15T": LIVE_M15, "5T": LIVE_M5}

    prep = PREPARE_DATA_FOR_TRAIN(fp, "30T")

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            merged = prep.load_data()
    except Exception as e:
        logging.error(f"[snapshot] load_data failed: {e}"); return False

    need = window + 1
    if raw_win is not None:
        merged = pd.concat([raw_win, merged], ignore_index=True).tail(need)
    else:
        merged = merged.tail(need)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            X,_ = (prep.ready(merged, 1, feats, "predict") if window == 1
                   else prep.ready_incremental(merged, window, feats))
    except Exception as e:
        logging.error(f"[snapshot] prepare failed: {e}"); return False

    if X.empty:
        logging.warning("[snapshot] X_live empty"); return False

    X = X.fillna(0)
    X.columns = [str(c) for c in X.columns]
    X = X.reindex(columns=cols, fill_value=0).astype(float)
    X.tail(1).to_csv(SNAPSHOT_CSV, index=False)
    return True

# ╔════════════════════════════════════════════════════════════╗
#  main loop
# ╚════════════════════════════════════════════════════════════╝
def main():
    ap = argparse.ArgumentParser("Sim‑Snapshot‑Predict (last N rows)")
    ap.add_argument("--model", default="best_model.pkl", help="trained model pickle")
    ap.add_argument("--rows", type=int, default=50_000,
                    help="how many last rows of M30 to replay (default 50k)")
    args = ap.parse_args()

    setup_log()
    purge()

    try:
        saved   = joblib.load(args.model)
        pipe    = saved["pipeline"]
        scaler  = saved["scaler"]
    except Exception as e:
        logging.critical(f"Cannot load model: {e}"); sys.exit(1)

    d5 = load_sorted("XAUUSD_M5.csv")
    d15 = load_sorted("XAUUSD_M15.csv")
    d30 = load_sorted("XAUUSD_M30.csv")
    dH1 = load_sorted("XAUUSD_H1.csv")

    total_rows = min(args.rows, len(d30) - 1)
    start_idx  = len(d30) - total_rows - 1   # first index to process

    stats = dict(w=0, l=0, nan=0, profit=0.0)

    with tqdm(total=total_rows, unit="bar", desc="Simulation") as bar:
        for offset in range(total_rows):
            idx = len(d30) - 1 - offset
            if idx <= start_idx: break

            while any(os.path.exists(fp) for fp in (LIVE_M30, LIVE_M15, LIVE_M5, LIVE_H1)):
                time.sleep(SLEEP)

            row, nxt = d30.iloc[idx], d30.iloc[idx - 1]
            ts  = row["time"]
            Δ   = row["close"] - nxt["close"]
            true = "SEL" if Δ >= 0 else "BUY"

            if not write_csvs(ts, d5, d15, d30, dH1):
                stats["nan"] += 1; purge(); bar.update(); continue

            if not build_snapshot(saved):
                stats["nan"] += 1; purge(); bar.update(); continue

            df = pd.read_csv(SNAPSHOT_CSV).fillna(0)
            X  = scaler.transform(df.values) if scaler is not None else df.values
            prob = pipe.predict_proba(X)[0, 1]
            pred = "BUY" if pipe.predict(X)[0] == 1 else "SEL"

            if pred == true:
                stats["w"] += 1; stats["profit"] += abs(Δ)
            else:
                stats["l"] += 1; stats["profit"] -= abs(Δ)

            acc = stats["w"] / (stats["w"] + stats["l"]) if (stats["w"] + stats["l"]) else 0
            bar.set_postfix(acc=f"{acc:.3f}", pred=pred, prob=f"{prob:.3f}")

            logging.info(f"[{offset+1}/{total_rows}] {ts} pred={pred} prob={prob:.3f} "
                         f"true={true} Δ={Δ:.2f} acc={acc:.3f}")

            purge(); bar.update()

    pd.DataFrame([stats]).to_csv("simulation_summary.csv", index=False)
    logging.info("Finished – summary ➜ simulation_summary.csv")

# ───────── entry
if __name__ == "__main__":
    main()
