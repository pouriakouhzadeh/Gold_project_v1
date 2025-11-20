#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_parity_reports.py  —  مقایسهٔ کامل بین:

1) live_like_sim_v3:
    - sim_X_feed_tail200.csv
    - sim_predictions.csv

2) دپلوی + ژنراتور:
    - deploy_X_feed_tail200.csv
    - deploy_predictions.csv
    - generator_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
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
        LOG.warning("Feature CSVs not found, skipping feature comparison.")
        return

    sim = pd.read_csv(sim_feat_path)
    dep = pd.read_csv(dep_feat_path)

    if "timestamp" in sim.columns:
        sim["timestamp"] = pd.to_datetime(sim["timestamp"], errors="coerce")
    if "timestamp" in dep.columns:
        dep["timestamp"] = pd.to_datetime(dep["timestamp"], errors="coerce")

    merged = sim.merge(dep, on="timestamp", how="inner", suffixes=("_sim", "_dep"))
    if merged.empty:
        LOG.warning("No overlapping timestamps between sim and deploy feature feeds.")
        return

    # ستون‌هایی که واقعاً فیچر هستند (نه y_true و نه ستون‌های زمانی / متادیتا)
    exclude_sim = {"timestamp", "y_true"}
    exclude_dep = {
        "timestamp",
        "timestamp_trigger",
        "y_true",
        "y_prob",
        "action",
        "cover_cum",
        "neg_thr",
        "pos_thr",
    }
    feat_cols = sorted(
        set(c.replace("_sim", "") for c in merged.columns if c.endswith("_sim"))
        & set(c.replace("_dep", "") for c in merged.columns if c.endswith("_dep"))
    )
    feat_cols = [c for c in feat_cols if c not in exclude_sim and c not in exclude_dep]

    if not feat_cols:
        LOG.warning("No common feature columns to compare.")
        return

    rows = []
    for c in feat_cols:
        diff = (
            merged[f"{c}_dep"].astype(float) - merged[f"{c}_sim"].astype(float)
        ).abs()
        rows.append(
            {
                "feature": c,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values("max_abs_diff", ascending=False)
    out_summary = base_dir / "features_compare_summary.csv"
    summary.to_csv(out_summary, index=False)
    LOG.info("Feature comparison summary written to %s", out_summary)


# ---------- Prediction comparison ----------
def _acc_cover(df: pd.DataFrame, src: str) -> None:
    if "action" not in df.columns or "y_true" not in df.columns:
        LOG.info("[%s] no 'action'/'y_true' columns, skipped", src)
        return
    act = df["action"].astype(str)
    y_t = df["y_true"]
    mask = act != "NONE"
    cover = float(mask.mean()) if len(mask) else 0.0
    if mask.any():
        correct = (
            ((act[mask] == "BUY") & (y_t[mask] == 1))
            | ((act[mask] == "SELL") & (y_t[mask] == 0))
        )
        acc = float(correct.mean())
    else:
        acc = 0.0
    LOG.info("[%s] acc=%.3f cover=%.3f", src, acc, cover)


def compare_predictions(base_dir: Path, prob_tol: float = 1e-9) -> None:
    sim_path = base_dir / "sim_predictions.csv"
    dep_path = base_dir / "deploy_predictions.csv"
    gen_path = base_dir / "generator_predictions.csv"

    if not sim_path.is_file() or not dep_path.is_file() or not gen_path.is_file():
        LOG.warning("Prediction CSVs missing, skipping prediction comparison.")
        return

    sim = pd.read_csv(sim_path)
    dep = pd.read_csv(dep_path)
    gen = pd.read_csv(gen_path)

    for df in (sim, dep, gen):
        if "timestamp" not in df.columns:
            raise ValueError(f"{df} must have 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # نام‌گذاری ستون‌ها برای جلوگیری از تداخل
    sim = sim.rename(
        columns={
            "action": "action_sim",
            "y_true": "y_true_sim",
            "y_prob": "y_prob_sim",
        }
    )
    dep = dep.rename(
        columns={
            "action": "action_dep",
            "y_true": "y_true_dep",
            "y_prob": "y_prob_dep",
        }
    )
    gen = gen.rename(
        columns={
            "action": "action_gen",
            "y_true": "y_true_gen",
            "y_prob": "y_prob_gen",
        }
    )

    merged = (
        sim.merge(dep, on="timestamp", how="inner")
        .merge(gen, on="timestamp", how="inner")
        .sort_values("timestamp")
    )

    if merged.empty:
        LOG.warning("No overlapping timestamps across sim/deploy/generator.")
        return

    # اختلاف احتمالات
    merged["prob_diff_sim_dep"] = (
        merged["y_prob_sim"].astype(float) - merged["y_prob_dep"].astype(float)
    ).abs()
    merged["prob_diff_sim_gen"] = (
        merged["y_prob_sim"].astype(float) - merged["y_prob_gen"].astype(float)
    ).abs()

    merged["prob_equal_sim_dep"] = merged["prob_diff_sim_dep"] <= prob_tol
    merged["prob_equal_sim_gen"] = merged["prob_diff_sim_gen"] <= prob_tol

    # برابری اکشن‌ها
    merged["action_equal_sim_dep"] = (
        merged["action_sim"].astype(str) == merged["action_dep"].astype(str)
    )
    merged["action_equal_sim_gen"] = (
        merged["action_sim"].astype(str) == merged["action_gen"].astype(str)
    )

    out_path = base_dir / "predictions_compare.csv"
    merged.to_csv(out_path, index=False)
    LOG.info("predictions_compare.csv written to %s", out_path)

    total = len(merged)
    LOG.info("---- Prediction parity summary (inner-joined timestamps) ----")
    LOG.info("Total rows = %d", total)

    for name, col in [
        ("action_equal_sim_dep", "action_equal_sim_dep"),
        ("action_equal_sim_gen", "action_equal_sim_gen"),
        ("prob_equal_sim_dep", "prob_equal_sim_dep"),
        ("prob_equal_sim_gen", "prob_equal_sim_gen"),
    ]:
        if col in merged.columns:
            val = float(merged[col].mean())
            LOG.info("%s: %.2f%%", name, 100.0 * val)

    # دقت/کاور هر کدام جداگانه
    _acc_cover(
        sim.rename(columns={"action_sim": "action", "y_true_sim": "y_true"}), "sim"
    )
    _acc_cover(
        dep.rename(columns={"action_dep": "action", "y_true_dep": "y_true"}), "deploy"
    )
    _acc_cover(
        gen.rename(columns={"action_gen": "action", "y_true_gen": "y_true"}),
        "generator",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", type=str)
    ap.add_argument("--prob-tol", default=1e-9, type=float)
    ap.add_argument("--verbosity", default=1, type=int)
    args = ap.parse_args()

    setup_logging(args.verbosity)
    base_dir = Path(args.base_dir).resolve()

    LOG.info("=== Feature comparison ===")
    compare_features(base_dir)

    LOG.info("=== Prediction comparison ===")
    compare_predictions(base_dir, prob_tol=args.prob_tol)


if __name__ == "__main__":
    main()
