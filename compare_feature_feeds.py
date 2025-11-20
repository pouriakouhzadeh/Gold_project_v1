#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_parity_reports.py  —  مقایسهٔ فیچرها و پیش‌بینی‌ها بین:

1) live_like_sim_v3/v4:
    - sim_X_feed_tail200.csv
    - sim_predictions.csv

2) دپلوی + ژنراتور:
    - deploy_X_feed_tail200.csv
    - deploy_predictions.csv
    - generator_predictions.csv
"""

from __future__ import annotations
import argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("compare_parity")


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity > 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------- Feature comparison ----------
def compare_features(base_dir: Path) -> None:
    sim_feat_path = base_dir / "sim_X_feed_tail200.csv"
    dep_feat_path = base_dir / "deploy_X_feed_tail200.csv"

    if not sim_feat_path.is_file() or not dep_feat_path.is_file():
        LOG.warning(
            "Feature CSVs not found (sim_X_feed_tail200.csv / deploy_X_feed_tail200.csv), skipping feature comparison."
        )
        return

    sim = pd.read_csv(sim_feat_path)
    dep = pd.read_csv(dep_feat_path)

    # ستون‌های زمان
    if "timestamp" in sim.columns:
        sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")
        tcol_sim = "timestamp"
    else:
        LOG.warning("No 'timestamp' column in sim_X_feed_tail200.csv; using row index.")
        sim["timestamp"] = range(len(sim))
        tcol_sim = "timestamp"

    if "timestamp_feature" in dep.columns:
        dep["timestamp_feature"] = pd.to_datetime(
            dep["timestamp_feature"], errors="coerce"
        )
        tcol_dep = "timestamp_feature"
    else:
        LOG.warning(
            "No 'timestamp_feature' column in deploy_X_feed_tail200.csv; using row index."
        )
        dep["timestamp_feature"] = range(len(dep))
        tcol_dep = "timestamp_feature"

    merged = sim.merge(
        dep,
        left_on=tcol_sim,
        right_on=tcol_dep,
        how="inner",
        suffixes=("_sim", "_dep"),
    )
    if merged.empty:
        LOG.warning("No overlapping timestamps between sim and deploy features.")
        return

    # فیچرهای مشترک
    exclude_sim = {tcol_sim, "y_true"}
    exclude_dep = {tcol_dep, "timestamp_trigger"}
    feat_cols = sorted(
        set(c for c in sim.columns if c not in exclude_sim)
        & set(c for c in dep.columns if c not in exclude_dep)
    )

    if not feat_cols:
        LOG.warning("No common feature columns between sim and deploy.")
        return

    # محاسبهٔ diff برای هر فیچر
    for c in feat_cols:
        merged[f"{c}_diff"] = (
            merged[f"{c}_dep"].astype(float) - merged[f"{c}_sim"].astype(float)
        )

    # detailed
    detailed_path = base_dir / "features_compare_detailed.csv"
    merged.to_csv(detailed_path, index=False)
    LOG.info("features_compare_detailed.csv written (%s)", detailed_path)

    # summary
    rows = []
    for c in feat_cols:
        diff = merged[f"{c}_diff"].astype(float)
        rows.append(
            {
                "feature": c,
                "max_abs_diff": float(diff.abs().max()),
                "mean_abs_diff": float(diff.abs().mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)
    summary_path = base_dir / "features_compare_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOG.info("features_compare_summary.csv written (%s)", summary_path)


# ---------- Prediction comparison ----------
def compare_predictions(base_dir: Path, prob_tol: float = 1e-9) -> None:
    sim_path = base_dir / "sim_predictions.csv"
    gen_path = base_dir / "generator_predictions.csv"
    dep_path = base_dir / "deploy_predictions.csv"

    if not sim_path.is_file() or not gen_path.is_file() or not dep_path.is_file():
        LOG.warning(
            "Prediction CSVs not found (sim_predictions / generator_predictions / deploy_predictions), skipping prediction comparison."
        )
        return

    sim = pd.read_csv(sim_path)
    gen = pd.read_csv(gen_path)
    dep = pd.read_csv(dep_path)

    for df in (sim, gen, dep):
        if "timestamp" not in df.columns:
            raise ValueError("All prediction files must have 'timestamp' column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # rename deploy columns to جلوگیری از تداخل نام‌ها
    dep = dep.rename(
        columns={
            "action": "action_dep",
            "y_prob": "y_prob_dep",
            "cover_cum": "cover_cum_dep",
            "y_true": "y_true_dep",
        }
    )

    merged = (
        sim.merge(gen, on="timestamp", suffixes=("_sim", "_gen"))
        .merge(dep, on="timestamp", how="left")
        .sort_values("timestamp")
    )

    if merged.empty:
        LOG.warning("No overlapping timestamps between sim and generator.")
        return

    # فلگ‌های برابری / اختلاف
    if "y_true_sim" in merged.columns and "y_true_gen" in merged.columns:
        merged["y_true_equal_sim_gen"] = merged["y_true_sim"] == merged["y_true_gen"]
    else:
        merged["y_true_equal_sim_gen"] = np.nan

    if "action_sim" in merged.columns and "action_gen" in merged.columns:
        merged["action_equal_sim_gen"] = merged["action_sim"] == merged["action_gen"]
    else:
        merged["action_equal_sim_gen"] = np.nan

    if "y_prob_sim" in merged.columns and "y_prob_gen" in merged.columns:
        merged["prob_diff_sim_gen"] = (
            merged["y_prob_sim"].astype(float) - merged["y_prob_gen"].astype(float)
        ).abs()
        merged["prob_equal_sim_gen"] = merged["prob_diff_sim_gen"] <= prob_tol
    else:
        merged["prob_diff_sim_gen"] = np.nan
        merged["prob_equal_sim_gen"] = np.nan

    if "y_prob_sim" in merged.columns and "y_prob_dep" in merged.columns:
        merged["prob_diff_sim_dep"] = (
            merged["y_prob_sim"].astype(float) - merged["y_prob_dep"].astype(float)
        ).abs()
    else:
        merged["prob_diff_sim_dep"] = np.nan

    # ذخیره
    out_path = base_dir / "predictions_compare.csv"
    merged.to_csv(out_path, index=False)
    LOG.info("predictions_compare.csv written (%s)", out_path)

    # خلاصهٔ آماری
    total = len(merged)
    y_equal = merged["y_true_equal_sim_gen"].sum()
    act_equal = merged["action_equal_sim_gen"].sum()
    prob_close = merged["prob_equal_sim_gen"].sum()

    LOG.info("---- Prediction parity summary ----")
    LOG.info("Total rows (joined) = %d", total)
    LOG.info(
        "y_true equal (sim vs gen)    : %d (%.2f%%)",
        y_equal,
        100.0 * y_equal / max(1, total),
    )
    LOG.info(
        "action equal (sim vs gen)    : %d (%.2f%%)",
        act_equal,
        100.0 * act_equal / max(1, total),
    )
    LOG.info(
        "prob |sim-gen| <= %.1e       : %d (%.2f%%)",
        prob_tol,
        prob_close,
        100.0 * prob_close / max(1, total),
    )

    # دقت و cover برای هر سه
    def _acc_cover(df: pd.DataFrame, src: str) -> None:
        if "action" not in df.columns or "y_true" not in df.columns:
            LOG.info("[%s] no 'action'/'y_true' columns, skip", src)
            return
        act = df["action"]
        y_t = df["y_true"]
        mask = act != "NONE"
        cover = float(mask.mean()) if len(mask) else 0.0
        if mask.any():
            correct = ((act[mask] == "BUY") & (y_t[mask] == 1)) | (
                (act[mask] == "SELL") & (y_t[mask] == 0)
            )
            acc = float(correct.mean())
        else:
            acc = 0.0
        LOG.info("[%s] acc=%.3f cover=%.3f", src, acc, cover)

    _acc_cover(sim.rename(columns={"action": "action", "y_true": "y_true"}), "sim")
    _acc_cover(gen, "generator")
    _acc_cover(
        dep.rename(
            columns={
                "action_dep": "action",
                "y_true_dep": "y_true",
            }
        ),
        "deploy",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--prob-tol", default=1e-9, type=float)
    ap.add_argument("--verbosity", default=1, type=int)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base_dir = Path(args.base_dir).resolve()

    LOG.info("=== Comparing features ===")
    compare_features(base_dir)

    LOG.info("=== Comparing predictions ===")
    compare_predictions(base_dir, prob_tol=args.prob_tol)


if __name__ == "__main__":
    main()
