#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_last_row_drift.py
────────────────────────────────────────────────────────

⏱️  Purpose
    Detect if the **very last feature‑row** changes once a new candle/bar is
    added.  We examine the most‑recent N rows (default 1000) **without any
    stepping**, write each result *immediately* to a CSV (append mode) **and**
    echo the same line to the console.  This way you can interrupt the script
    at any moment and still keep all logged data up to that point.

🔑  Key features (v4)
    • ``--last``  (=1000) – how many tail rows to scan.
    • ``--workers`` – optional parallelism; on error automatically falls back
      to sequential execution.
    • "What you see is what is saved": every CSV line is printed with
      ``tqdm.write`` right after being flushed to disk.

📦  Example
    $ python check_last_row_drift.py --model best_model.pkl --workers 4 --last 1000

    # Sequential (safe) run
    $ python check_last_row_drift.py --model best_model.pkl --workers 1

"""
from __future__ import annotations

import argparse
import csv
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from tqdm import tqdm

from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

# ─────────────────────────────── CLI ────────────────────────────────
parser = argparse.ArgumentParser(
    description="Detect per‑candle drift in the last feature row; "
                "store results incrementally so the script can be stopped/resumed.")

parser.add_argument('-m', '--model', default='best_model.pkl',
                    help='Path to the pickled model (must contain window_size & feats).')
parser.add_argument('-l', '--last', type=int, default=1000,
                    help='Number of most‑recent rows to evaluate (default 1000).')
parser.add_argument('-w', '--workers', type=int, default=1,
                    help='Parallel workers. 1 = sequential (default).')
parser.add_argument('-o', '--out', default='feature_shift_report.csv',
                    help='Output CSV file (append mode).')

ARGS = parser.parse_args()

# ─────────────────────────── Logging setup ──────────────────────────
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)

# ───────────────────── Load model and data once ─────────────────────
logging.info("Loading model from %s", ARGS.model)
MODEL_OBJ = joblib.load(ARGS.model)
WINDOW_SIZE: int = MODEL_OBJ['window_size']
SELECTED_FEATURES: List[str] = MODEL_OBJ['feats']

logging.info("Initialising data‑prep class …")
PREP = PREPARE_DATA_FOR_TRAIN(main_timeframe='30T', verbose=False)
logging.info("Loading merged/raw data … (this may take a moment)")
MERGED_DF: pd.DataFrame = PREP.load_data()
logging.info("Merged data shape: %s", MERGED_DF.shape)

# Tail selection boundaries
END_IDX = len(MERGED_DF) - 2          # need i+2 in loop
START_IDX = max(WINDOW_SIZE, END_IDX - ARGS.last)
INDICES = list(range(START_IDX, END_IDX))
logging.info("Scanning last %d rows (idx %d → %d)", len(INDICES), START_IDX, END_IDX - 1)

# ──────────────────────── CSV initialisation ────────────────────────
CSV_PATH = Path(ARGS.out)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
FIELDNAMES = ['idx', 'time', 'changed', 'num_changed', 'changed_cols']
FIRST_WRITE = not CSV_PATH.exists()
if FIRST_WRITE:
    with CSV_PATH.open('w', newline='') as f:
        csv.DictWriter(f, FIELDNAMES).writeheader()

# ───────────────────────── Worker function ──────────────────────────

def _process_index(i: int) -> Optional[Dict[str, str]]:
    """Return row‑dict for drift report or None to skip (insufficient data)."""
    # slices
    slice1 = MERGED_DF.iloc[: i + 1].copy()
    slice2 = MERGED_DF.iloc[: i + 2].copy()

    # Feature preparation depending on window
    if WINDOW_SIZE == 1:
        X1, _ = PREP.ready(slice1, window=1, selected_features=SELECTED_FEATURES, mode='predict')
        X2, _ = PREP.ready(slice2, window=1, selected_features=SELECTED_FEATURES, mode='predict')
        if X1.empty or len(X2) < 2:
            return None
        row_prev = X1.iloc[-1]
        row_prev_after = X2.iloc[-2]  # same candle after adding new row
    else:
        # Incremental mode does not provide comparable previous row; skip.
        X1, _ = PREP.ready_incremental(slice1, window=WINDOW_SIZE, selected_features=SELECTED_FEATURES)
        X2, _ = PREP.ready_incremental(slice2, window=WINDOW_SIZE, selected_features=SELECTED_FEATURES)
        if X1.empty or X2.empty:
            return None
        return None  # cannot compare reliably in incremental window>1

    diff_mask = row_prev.ne(row_prev_after)
    changed_cols: List[str] = row_prev.index[diff_mask].tolist()

    return {
        'idx': i,
        'time': str(slice1.index[-1]) if not slice1.index.empty else '',
        'changed': bool(changed_cols),
        'num_changed': len(changed_cols),
        'changed_cols': ';'.join(changed_cols)
    }

# ─────────────────────────── Main routine ───────────────────────────

def _write_row(row: Dict[str, str]):
    """Append single row to CSV & echo to console."""
    with CSV_PATH.open('a', newline='') as f:
        csv.DictWriter(f, FIELDNAMES).writerow(row)
    tqdm.write(str(row))


def run_sequential():
    for i in tqdm(INDICES, desc="Drift", unit="row"):
        row = _process_index(i)
        if row:
            _write_row(row)


def run_parallel(workers: int):
    logging.info("Running with %d workers", workers)
    with Pool(workers) as pool, tqdm(total=len(INDICES), desc="Drift", unit="row") as pbar:
        for row in pool.imap_unordered(_process_index, INDICES, chunksize=10):
            if row:
                _write_row(row)
            pbar.update()


if __name__ == '__main__':
    # first try parallel if requested and reasonable
    if ARGS.workers > 1:
        try:
            run_parallel(min(ARGS.workers, cpu_count()))
        except Exception as exc:
            logging.warning("Parallel execution failed (%s). Falling back to sequential.", exc)
            run_sequential()
    else:
        run_sequential()

    logging.info("Finished → %s", CSV_PATH.resolve())
