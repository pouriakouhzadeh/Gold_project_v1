#!/usr/bin/env python3
"""Genetic‑Algorithm optimisation with automatic checkpoint/restart.

This script is an **augmented** version of your original GA code.  The only
functional additions are:

1.  Portable **checkpointing** of the random states, population, and fitness
    after every generation (and on graceful SIGINT/SIGTERM)
2.  **Automatic resume**: if the checkpoint file exists the run continues from
    the next generation instead of starting from scratch.
3.  Removal of the checkpoint once training finishes successfully.

All other logic (model evaluation, fitness definition, thresholds, etc.) is
unchanged so that training results remain reproducible.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#                              Standard imports
# ---------------------------------------------------------------------------
import argparse
import gc
import logging
import multiprocessing as mp
import os
import pickle
import random
import signal
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning
import warnings

# ---------------------------------------------------------------------------
#                         Numerical‑lib thread limits
# ---------------------------------------------------------------------------
cores = str(mp.cpu_count())
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.pop(var, None)  # remove any previous limitation

# ---------------------------------------------------------------------------
#                       Optional Intel® oneAPI patch
# ---------------------------------------------------------------------------
from sklearnex import patch_sklearn  # type: ignore

patch_sklearn(verbose=False)

# ---------------------------------------------------------------------------
#                       Local project‑specific imports
# ---------------------------------------------------------------------------
from model_pipeline import ModelPipeline  # type: ignore
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore
from ModelSaver import ModelSaver  # type: ignore
from threshold_finder import ThresholdFinder  # type: ignore
from drift_checker import DriftChecker  # type: ignore

# ---------------------------------------------------------------------------
#                                Global setup
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = int(os.environ.get("GA_GLOBAL_SEED", 2025))
random.seed(SEED)
np.random.seed(SEED)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_fmt = "%(asctime)s %(levelname)s: %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"
file_hdl = RotatingFileHandler("genetic_algorithm.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
file_hdl.setFormatter(logging.Formatter(_fmt, _datefmt))
console_hdl = logging.StreamHandler()
console_hdl.setFormatter(logging.Formatter(_fmt, _datefmt))
LOGGER.addHandler(file_hdl)
LOGGER.addHandler(console_hdl)
for _mod in ("sklearnex", "sklearn", "daal4py"):
    logging.getLogger(_mod).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
#                     GA hyper‑parameters & checkpoint
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    population_size: int = 4
    n_generations: int = 2
    cx_pb: float = 0.8
    mut_pb: float = 0.4
    early_stopping_threshold: float = 0.85
    possible_window_sizes: Tuple[int, ...] = tuple(range(1, 11))


CFG = GAConfig()
CHECKPOINT_FILE = "ga_checkpoint.pkl"

# ---------------------------------------------------------------------------
#                               DEAP boilerplate
# ---------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
TOOLBOX = base.Toolbox()

# --------------------- stochastic hyper‑parameter samplers ------------------
rand_C = lambda: 10 ** random.uniform(-4, 2)
rand_max_iter = lambda: random.randint(8000, 10_000)
rand_tol = lambda: 10 ** random.uniform(-5, -2)
rand_penalty = lambda: random.choice(["l1", "l2"])
rand_solver = lambda: random.choice(["sag", "saga"])
rand_calib = lambda: random.choice(["sigmoid", "isotonic"])
rand_fit_intercept = lambda: random.choice([True, False])
rand_class_weight = lambda: random.choice([None, "balanced"])
rand_multi_class = lambda: random.choice(["auto", "ovr", "multinomial"])
rand_window_size = lambda: random.choice(CFG.possible_window_sizes)


# ---------------------------------------------------------------------------
#                     Shared helpers and multiprocessing pool
# ---------------------------------------------------------------------------
DATA_TRAIN_SHARED: pd.DataFrame | None = None
PREP_SHARED: PREPARE_DATA_FOR_TRAIN | None = None


def _suppress_warnings() -> None:
    """Suppress sklearn convergence warnings in child processes."""
    import warnings as _w

    from sklearn.exceptions import ConvergenceWarning as _CW

    _w.filterwarnings("ignore", category=_CW)
    _w.filterwarnings("ignore", message=".*max_iter.*")


def pool_init(data_train: pd.DataFrame, prep: PREPARE_DATA_FOR_TRAIN) -> None:
    _suppress_warnings()
    global DATA_TRAIN_SHARED, PREP_SHARED
    DATA_TRAIN_SHARED = data_train
    PREP_SHARED = prep
    print("[Pool] Shared globals initialised in worker")


# ---------------------------------------------------------------------------
#                        Individual creation & mutation
# ---------------------------------------------------------------------------

def create_individual() -> creator.Individual:
    penalty = rand_penalty()
    solver = rand_solver() if penalty == "l2" else "saga"
    multi = rand_multi_class()
    if solver == "liblinear" and multi == "multinomial":
        multi = random.choice(["auto", "ovr"])
    return creator.Individual(
        [
            rand_C(),
            rand_max_iter(),
            rand_tol(),
            penalty,
            solver,
            rand_fit_intercept(),
            rand_class_weight(),
            multi,
            rand_window_size(),  # gene 8 - look‑back window
            rand_calib(),  # gene 9 - calibration method
        ]
    )


def mutate_ind(ind: creator.Individual, indpb: float = 0.2):
    for i in range(len(ind)):
        if random.random() >= indpb:
            continue
        if i == 0:
            ind[0] = rand_C()
        elif i == 1:
            ind[1] = rand_max_iter()
        elif i == 2:
            ind[2] = rand_tol()
        elif i == 3:
            ind[3] = rand_penalty()
        elif i == 4:
            ind[4] = rand_solver()
        elif i == 5:
            ind[5] = rand_fit_intercept()
        elif i == 6:
            ind[6] = rand_class_weight()
        elif i == 7:
            ind[7] = rand_multi_class()
        elif i == 8:
            ind[8] = rand_window_size()
        elif i == 9:
            ind[9] = rand_calib()

    # simple compatibility fixes
    penalty, solver, multi = ind[3], ind[4], ind[7]
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        ind[4] = random.choice(["liblinear", "saga"])
    if penalty == "l2" and solver not in ["lbfgs", "liblinear", "sag", "saga"]:
        ind[4] = random.choice(["lbfgs", "liblinear", "sag", "saga"])
    if ind[4] == "liblinear" and multi == "multinomial":
        ind[7] = random.choice(["auto", "ovr"])
    return (ind,)


# ---------------------------------------------------------------------------
#                         Fold evaluation (unchanged)
# ---------------------------------------------------------------------------

def _fit_and_score_fold(tr_idx, ts_idx, X_full, y_full, price_series, hyper, calib_method):
    # identical implementation – omitted for brevity in this header comment
    import numpy as _np
    import pandas as _pd

    X_tr_raw, y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
    X_ts_raw, y_ts = X_full.iloc[ts_idx], y_full.iloc[ts_idx]

    feats = PREP_SHARED.select_features(X_tr_raw, y_tr)
    if not feats:
        return 0.0

    X_tr, X_ts = X_tr_raw[feats], X_ts_raw[feats]

    pipe = ModelPipeline(hyper, calibrate=True, calib_method=calib_method).fit(X_tr, y_tr)
    y_pred = pipe.pipeline.predict(X_ts)

    prices = price_series.iloc[ts_idx].astype(float).values
    if len(prices) < 2:
        return 0.0

    ret_mkt = _np.diff(prices) / prices[:-1]
    pos_shift = _np.roll(y_pred, 1)
    pos_shift[0] = 0
    pos_shift = pos_shift[:-1]
    ret_str = _np.where(pos_shift == 1, ret_mkt, 0.0)

    ret_ser = _pd.Series(ret_str)
    std = ret_ser.std()
    if std is None or std == 0 or _np.isnan(std):
        return 0.0

    sharpe = ret_ser.mean() / std * _np.sqrt(252 * 48)
    cum_equity = ret_ser.add(1).cumprod()
    max_dd = (cum_equity.cummax() - cum_equity).max()

    bal_acc = balanced_accuracy_score(y_ts, y_pred)
    if bal_acc < 0.55:
        return 0.0

    norm_sharpe = (np.tanh(sharpe / 5) + 1) / 2
    norm_dd = 1.0 - min(max_dd, 1.0)
    return 0.4 * norm_sharpe + 0.4 * norm_dd + 0.2 * bal_acc


# ---------------------------------------------------------------------------
#                               Fitness wrapper
# ---------------------------------------------------------------------------

def evaluate_cv(ind):
    try:
        if any(v is None for v in (DATA_TRAIN_SHARED, PREP_SHARED)):
            raise RuntimeError("Shared globals not ready!")

        (
            C,
            max_iter,
            tol,
            penalty,
            solver,
            fit_intercept,
            class_weight,
            multi_class,
            window,
            calib_method,
        ) = ind

        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            return (0.0,)
        if penalty == "l2" and solver not in ["lbfgs", "liblinear", "sag", "saga"]:
            return (0.0,)
        if solver == "liblinear" and multi_class == "multinomial":
            return (0.0,)

        X, y, _, price_ser = PREP_SHARED.ready(DATA_TRAIN_SHARED, window=window, selected_features=[], mode="train")
        if X.empty:
            return (0.0,)

        hyper = {
            "C": C,
            "max_iter": max_iter,
            "tol": tol,
            "penalty": penalty,
            "solver": solver,
            "fit_intercept": fit_intercept,
            "class_weight": class_weight,
            "multi_class": multi_class,
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_and_score_fold)(tr, ts, X, y, price_ser, hyper, calib_method) for tr, ts in tscv.split(X, y)
        )
        return (float(np.mean(scores)),)
    except Exception as e:
        LOGGER.error("evaluate_cv failed: %s", e)
        return (0.0,)


# ---------------------------------------------------------------------------
#                            Toolbox registrations
# ---------------------------------------------------------------------------
TOOLBOX.register("mate", tools.cxTwoPoint)
TOOLBOX.register("mutate", mutate_ind, indpb=0.2)
TOOLBOX.register("select", tools.selTournament, tournsize=3)


# ---------------------------------------------------------------------------
#                     Checkpoint helper functions
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, fname: str = CHECKPOINT_FILE):
    try:
        with open(fname, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Checkpoint saved → %s", fname)
    except Exception as exc:
        LOGGER.error("Could not save checkpoint: %s", exc)


def load_checkpoint(fname: str = CHECKPOINT_FILE):
    if not os.path.exists(fname):
        return None
    try:
        with open(fname, "rb") as f:
            state = pickle.load(f)
        LOGGER.info("Checkpoint loaded ← %s", fname)
        return state
    except Exception as exc:
        LOGGER.error("Failed to load checkpoint: %s", exc)
        return None


# ---------------------------------------------------------------------------
#                           Main GA runner class
# ---------------------------------------------------------------------------
class GeneticAlgorithmRunner:
    def __init__(self, checkpoint_path: str = CHECKPOINT_FILE):
        self.neg_thr = 0.5
        self.pos_thr = 0.5
        self.final_cols: list[str] = []
        self.checkpoint_path = checkpoint_path
        self.current_gen = 0  # will be updated while running
        self.best_overall = 0.0
        self.population: list[creator.Individual] = []

    # ------------------------------------------------------------------
    #                           Public entry‑point
    # ------------------------------------------------------------------
    def main(self):
        # -------------------------------------------------------------
        # 0)  Attempt resume
        # -------------------------------------------------------------
        state = load_checkpoint(self.checkpoint_path)
        if state is not None:
            LOGGER.info("⇢ Resuming from generation %d", state["gen"])
            self.current_gen = state["gen"]
            self.best_overall = state["best_overall"]
            self.population = state["population"]
            random.setstate(state["rand_state"])
            np.random.set_state(state["np_state"])
        else:
            LOGGER.info("⇢ Starting a *new* optimisation run")

        # -------------------------------------------------------------
        # 1)  Prepare data & pool (always redone – cheap / stateless)
        # -------------------------------------------------------------
        prep = PREPARE_DATA_FOR_TRAIN(main_timeframe="30T")
        raw = prep.load_data()
        LOGGER.info("Raw data loaded → rows=%d, shape=%s", len(raw), raw.shape)

        time_col = f"{prep.main_timeframe}_time"
        raw[time_col] = pd.to_datetime(raw[time_col])
        raw.sort_values(time_col, inplace=True)
        total = len(raw)
        train_end, thresh_end = int(total * 0.85), int(total * 0.90)
        data_tr = raw.iloc[:train_end].copy()
        data_thr = raw.iloc[train_end:thresh_end].copy()
        data_te = raw.iloc[thresh_end:].copy()
        LOGGER.info("Split → train=%d, thresh=%d, test=%d", len(data_tr), len(data_thr), len(data_te))

        n_proc = min(mp.cpu_count(), 8)
        pool = mp.Pool(n_proc, initializer=pool_init, initargs=(data_tr, prep))
        LOGGER.info("Multiprocessing pool with %d workers created", n_proc)

        # Replace the default map with an imap_unordered backed by the pool
        if "map" in TOOLBOX.__dict__:
            TOOLBOX.unregister("map")
        TOOLBOX.register("map", lambda f, it: list(pool.imap_unordered(f, it, chunksize=1)))

        TOOLBOX.register("init_individual", create_individual)

        # -------------------------------------------------------------
        # 2)  Initialise or restore the population
        # -------------------------------------------------------------
        if not self.population:  # a resume had empty population OR fresh run
            self.population = [TOOLBOX.init_individual() for _ in range(CFG.population_size)]
            LOGGER.info("Initial population generated")
            # Evaluate fitness for the first time
            for ind, fit in zip(self.population, TOOLBOX.map(evaluate_cv, self.population)):
                ind.fitness.values = fit
            LOGGER.info("Initial fitnesses computed")
            self.best_overall = max(ind.fitness.values[0] for ind in self.population)
            self._save_checkpoint()  # save after first evaluation
            self.current_gen = 0

        # -------------------------------------------------------------
        # 3)  Define signal handlers for graceful termination
        # -------------------------------------------------------------
        def _sig_handler(signum, _frame):
            LOGGER.warning("Signal %s received – storing checkpoint and exiting …", signum)
            self._save_checkpoint()
            pool.close(); pool.join()
            sys.exit(0)

        for _sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(_sig, _sig_handler)

        # -------------------------------------------------------------
        # 4)  Main evolutionary loop (resumes correctly)
        # -------------------------------------------------------------
        for gen in range(self.current_gen + 1, CFG.n_generations + 1):
            self.current_gen = gen
            print(f"[GA] Generation {gen}/{CFG.n_generations} …")
            offspring = [tool for tool in TOOLBOX.select(self.population, len(self.population))]
            offspring = [creator.Individual(ind) for ind in offspring]  # deep copy lists

            # Crossover & mutation
            for i1, i2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CFG.cx_pb:
                    TOOLBOX.mate(i1, i2)
                    del i1.fitness.values, i2.fitness.values
            for mut in offspring:
                if random.random() < CFG.mut_pb:
                    TOOLBOX.mutate(mut)
                    del mut.fitness.values

            # Evaluate invalid ones
            invalid = [i for i in offspring if not i.fitness.valid]
            for ind, fit in zip(invalid, TOOLBOX.map(evaluate_cv, invalid)):
                ind.fitness.values = fit

            self.population[:] = offspring
            gc.collect()

            best_gen = tools.selBest(self.population, 1)[0]
            self.best_overall = max(self.best_overall, best_gen.fitness.values[0])
            print(f"[GA] Gen best = {best_gen.fitness.values[0]:.4f} · overall = {self.best_overall:.4f}")

            # Early stopping
            if best_gen.fitness.values[0] >= CFG.early_stopping_threshold:
                print("[GA] Early stopping reached!")
                self._save_checkpoint()  # ensure last state saved before break
                break

            # Persist after every generation
            self._save_checkpoint()

        best_ind = tools.selBest(self.population, 1)[0]
        print("[GA] Finished optimisation → best_score =", best_ind.fitness.values[0])

        # -------------------------------------------------------------
        # 5)  Final model training & evaluation (unchanged logic)
        # -------------------------------------------------------------
        final_model, feats = self._build_final_model(best_ind, data_tr, prep)
        if final_model is None:
            LOGGER.error("Final model could not be built – exiting")
            pool.close(); pool.join()
            return best_ind, best_ind.fitness.values[0]

        self._run_thresholds(final_model, data_thr, prep, best_ind, feats)
        self._eval(final_model, data_te, prep, best_ind, feats, label="Test")
        self._save(final_model, best_ind, feats)
        print("[main] Model & thresholds saved 🎉")

        pool.close(); pool.join()

        # Clean‑up: remove checkpoint when successful
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            LOGGER.info("Checkpoint removed – run completed successfully")

        return best_ind, best_ind.fitness.values[0]

    # ------------------------------------------------------------------
    #                       Convenience helper methods
    # ------------------------------------------------------------------
    def _save_checkpoint(self):
        state = {
            "gen": self.current_gen,
            "population": self.population,
            "best_overall": self.best_overall,
            "rand_state": random.getstate(),
            "np_state": np.random.get_state(),
        }
        save_checkpoint(state, self.checkpoint_path)

    # ---------------- helpers copied verbatim from original script ----------
    # _build_final_model, _run_thresholds, _eval, _save  (unchanged)

    # … (for brevity, these methods are identical to your original file; you can
    #    paste them here without any modifications.)


# ---------------------------------------------------------------------------
#                            CLI / entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic‑Algorithm optimiser with resume support")
    parser.add_argument("--checkpoint", "-ckpt", default=CHECKPOINT_FILE, help="Path to checkpoint file (default: ga_checkpoint.pkl)")
    parser.add_argument("--reset", action="store_true", help="Remove checkpoint and start fresh")
    args = parser.parse_args()

    # ✅ این خط باعث حذف چک‌پوینت می‌شود اگر --reset داده شده باشد
    if args.reset and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print("🗑️  Checkpoint removed – starting fresh!")

    runner = GeneticAlgorithmRunner(checkpoint_path=args.checkpoint)
    best_ind, best_score = runner.main()
    print("[MAIN] GA done → best_score =", best_score)
