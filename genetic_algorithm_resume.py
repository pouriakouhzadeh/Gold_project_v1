#!/usr/bin/env python3
# ---------- Thread limits for numeric libs ----------
from __future__ import annotations
# ---------- Enronviment variable --------------------
import os, multiprocessing as mp
cores = str(mp.cpu_count())          # ← تعداد واقعی هسته‌ها
# اگر می‌خواهید کاملاً آزاد باشد «پاک» کنید؛ در غیر این صورت مساوی cores
for var in ("OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TBB_NUM_THREADS"):
    os.environ.pop(var, None)        # ➊ پاک‌کردن محدودیت
    # یا: os.environ[var] = cores    # ➋ تنظیم روی حداکثر


# ---------------------------------------------------------------------------
# Intel® oneAPI acceleration (اختیاری)
# ---------------------------------------------------------------------------
from sklearnex import patch_sklearn  # type: ignore
patch_sklearn(verbose=False)

# ---------------------------------------------------------------------------
# کتابخانه‌های استاندارد و لاگ‌گیری
# ---------------------------------------------------------------------------
import logging, gc, os, random, copy, multiprocessing as mp
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
from model_pipeline import ModelPipeline          # type: ignore
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore
from ModelSaver import ModelSaver                 # type: ignore
from threshold_finder import ThresholdFinder      # type: ignore
from drift_checker import DriftChecker            # type: ignore
from sklearn.exceptions import ConvergenceWarning
import pickle, sys, signal
import warnings; 
warnings.filterwarnings("ignore",
                category=ConvergenceWarning)
# ---------------------------------------------------------------------------
# ثبات تصادفی
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("GA_GLOBAL_SEED", 2025))
random.seed(SEED)
np.random.seed(SEED)
CHECKPOINT = "ga_checkpoint.pkl"
# ---------------------------------------------------------------------------
# راه‌اندازی لاگر
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_fmt = "%(asctime)s %(levelname)s: %(message)s"
_date = "%Y-%m-%d %H:%M:%S"
file_hdl = RotatingFileHandler("genetic_algorithm.log", maxBytes=2_000_000,
                               backupCount=3, encoding="utf-8")
file_hdl.setFormatter(logging.Formatter(_fmt, _date))
console_hdl = logging.StreamHandler()
console_hdl.setFormatter(logging.Formatter(_fmt, _date))
LOGGER.addHandler(file_hdl)
LOGGER.addHandler(console_hdl)

logging.getLogger("sklearnex").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("daal4py").setLevel(logging.WARNING)

def _install_sig_handlers(cur_gen_fn, pop_fn, best_fn):

    def _handler(_sig, _frm):
        save_checkpoint(cur_gen_fn(), pop_fn(), best_fn())
        LOGGER.warning("Interrupted – checkpoint written; exiting …")
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def save_checkpoint(gen, population, best_overall):
    """تمام اطلاعات لازم برای ادامهٔ بی‌نقص را ذخیره می‌کند."""
    state = {
        "gen": gen,                               # نسل فعلی
        "population": population,                 # جمعیت با fitness
        "best_overall": best_overall,             # بهترین امتیاز تا این لحظه
        "rng_py": random.getstate(),              # وضعیت RNG پایتون
        "rng_np": np.random.get_state(),          # وضعیت RNG NumPy
    }
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Checkpoint saved (gen=%d)", gen)


def load_checkpoint() -> tuple[int, list, float] | None:
    """اگر فایل وجود داشته باشد state را برمی‌گرداند."""
    if not os.path.isfile(CHECKPOINT):
        return None
    with open(CHECKPOINT, "rb") as f:
        state = pickle.load(f)
    random.setstate(state["rng_py"])
    np.random.set_state(state["rng_np"])
    LOGGER.info("Checkpoint loaded (gen=%d)", state["gen"])
    return state["gen"], state["population"], state["best_overall"]



# ---------------------------------------------------------------------------
# پیکربندی GA
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    population_size: int = 4
    n_generations: int = 2
    cx_pb: float = 0.8
    mut_pb: float = 0.4
    early_stopping_threshold: float = 0.85
    possible_window_sizes: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

CFG = GAConfig()

# ---------------------------------------------------------------------------
# DEAP primitives
# ---------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
TOOLBOX = base.Toolbox()

# -------------------- توابع تصادفی --------------------
rand_C             = lambda: 10 ** random.uniform(-4, 2)
rand_max_iter      = lambda: random.randint(8000, 10_000)
rand_tol           = lambda: 10 ** random.uniform(-5, -2)
rand_penalty       = lambda: random.choice(["l1", "l2"])
rand_solver        = lambda: random.choice(["sag", "saga"])     # ← چندترد
rand_calib         = lambda: random.choice(["sigmoid", "isotonic"])
rand_fit_intercept = lambda: random.choice([True, False])
rand_class_weight  = lambda: random.choice([None, "balanced"])
rand_multi_class   = lambda: random.choice(["auto", "ovr", "multinomial"])
rand_window_size   = lambda: random.choice(CFG.possible_window_sizes)

# -------------------- ساخت فرد --------------------
# ---------------------------------------------------------------------------
# ۱) تابع مشترک برای خاموش‌کردن هشدارهای اسکیک‌لرن
# ---------------------------------------------------------------------------
def _suppress_warnings() -> None:
    """
    Silence sklearn’s ConvergenceWarning everywhere (main + workers).
    فراخوانی این تابع را هر جا که پروسهٔ جدید ساخته می‌شود تکرار کنید.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # هشدارهای مربوط به نرسیدن به همگرایی
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # اگر پیغام دیگری با عبارت max_iter بود (برخی نسخه‌ها UserWarning می‌دهند)
    warnings.filterwarnings("ignore", message=".*max_iter.*")


def create_individual() -> creator.Individual:
    """یک فرد تصادفی می‌سازد."""
    penalty = rand_penalty()
    solver = rand_solver() if penalty == "l2" else "saga"   # l1 ⇒ فقط saga

    multi   = rand_multi_class()
    if solver == "liblinear" and multi == "multinomial":
        multi = random.choice(["auto", "ovr"])

    return creator.Individual([
        rand_C(), rand_max_iter(), rand_tol(),
        penalty, solver,
        rand_fit_intercept(), rand_class_weight(),
        multi,
        rand_window_size(),         # ژن ۸
        rand_calib()                # ژن ۹ : calib_method
])


# -------------------- جهش --------------------

def mutate_ind(ind: creator.Individual, indpb: float = 0.2):
    for i in range(len(ind)):
        if random.random() >= indpb:
            continue
        if   i == 0: ind[0] = rand_C()
        elif i == 1: ind[1] = rand_max_iter()
        elif i == 2: ind[2] = rand_tol()
        elif i == 3: ind[3] = rand_penalty()
        elif i == 4: ind[4] = rand_solver()
        elif i == 5: ind[5] = rand_fit_intercept()
        elif i == 6: ind[6] = rand_class_weight()
        elif i == 7: ind[7] = rand_multi_class()
        elif i == 8: ind[8] = rand_window_size()
        elif i == 9: ind[9] = rand_calib()
    
    # سازگاری‌ها
    penalty, solver, multi = ind[3], ind[4], ind[7]
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        ind[4] = random.choice(["liblinear", "saga"])
    if penalty == "l2" and solver not in ["lbfgs", "liblinear", "sag", "saga"]:
        ind[4] = random.choice(["lbfgs", "liblinear", "sag", "saga"])
    if ind[4] == "liblinear" and multi == "multinomial":
        ind[7] = random.choice(["auto", "ovr"])
    return (ind,)

# ---------------------------------------------------------------------------
# اشتراک داده بین پروسس‌ها
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ۲) راه‌اندازی Pool با اعمال همان فیلتر در هر worker
# ---------------------------------------------------------------------------
DATA_TRAIN_SHARED: pd.DataFrame | None = None
PREP_SHARED: PREPARE_DATA_FOR_TRAIN | None = None

def pool_init(data_train: pd.DataFrame,
              prep: PREPARE_DATA_FOR_TRAIN) -> None:
    # خفه کردن هشدارها داخل این worker
    _suppress_warnings()

    # به‌اشتراک‌گذاری داده‌‌ها
    global DATA_TRAIN_SHARED, PREP_SHARED
    DATA_TRAIN_SHARED = data_train
    PREP_SHARED       = prep
    print("[Pool] Initialised shared globals in worker process")


# ---------------------------------------------------------------------------
# ارزیابی یک فولد (انتخاب ویژگی درون‑فولد)
# ---------------------------------------------------------------------------

def _fit_and_score_fold(tr_idx, ts_idx, X_full, y_full, price_series, hyper, calib_method):
   
    X_tr_raw, y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
    X_ts_raw, y_ts = X_full.iloc[ts_idx], y_full.iloc[ts_idx]

    feats = PREP_SHARED.select_features(X_tr_raw, y_tr)
    if not feats:
        return 0.0

    X_tr, X_ts = X_tr_raw[feats], X_ts_raw[feats]

    pipe = ModelPipeline(hyper, calibrate=True,
                      calib_method=calib_method).fit(X_tr, y_tr)
    y_pred = pipe.pipeline.predict(X_ts)       # 1 = لانگ، 0 = نقد

    # ── استخراج قیمت خام (ستون 30T_close) ──
    prices = price_series.iloc[ts_idx].astype(float).values
    
    if len(prices) < 2:               # داده‌ی خیلی کم
        return 0.0

    # ── بازده بازار و بازده استراتژی ──
    ret_mkt = np.diff(prices) / prices[:-1]

    pos_shift = np.roll(y_pred, 1)    # پوزیشنِ کندل قبلی
    pos_shift[0] = 0                  # اولین کندل → No-Trade
    pos_shift = pos_shift[:-1]        # طول = len(ret_mkt)

    ret_str = np.where(pos_shift == 1, ret_mkt, 0.0)


    ret_ser = pd.Series(ret_str)
    std = ret_ser.std()
    if (std is None) or (std == 0) or np.isnan(std):
        return 0.0                    # استراتژی بی‌تحرک یا انحراف صفر

    # Sharpe Ratio سالانه‌شده (برای 48 کندل ۳۰-دقیقه‌ای در روز)
    periods_per_year = 252 * 48
    sharpe = ret_ser.mean() / std * np.sqrt(periods_per_year)

    # Max-DrawDown
    cum_equity = ret_ser.add(1).cumprod()
    max_dd = (cum_equity.cummax() - cum_equity).max()

    # امتیاز نهایی = Sharpe – DrawDown (می‌توانی فرمول دیگری بگذاری)
    # --- Sharpe و Max-DD محاسبه شدند ---

    # ➊ Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_ts, y_pred)
    
    if bal_acc < 0.55:
        return 0.0
    # ➋ ترکیب دو معیار در یک Fitness
    #    ضریبِ 1 برای Score و 1 برای BalAcc → می‌توان تغییر داد
    norm_sharpe = (np.tanh(sharpe / 5) + 1) / 2       # ۰…۱
    norm_dd     = 1.0 - min(max_dd, 1.0)              # ۰…۱
    fitness     = 0.4 * norm_sharpe + 0.4 * norm_dd + 0.2 * bal_acc

    return fitness



# ---------------------------------------------------------------------------
# تابع fitness
# ---------------------------------------------------------------------------

def evaluate_cv(ind):
    try:
        if any(v is None for v in (DATA_TRAIN_SHARED, PREP_SHARED)):
            raise RuntimeError("Shared globals not ready!")

        (C, max_iter, tol, penalty, solver,
         fit_intercept, class_weight, multi_class,
         window, calib_method) = ind

        # ناسازگاری‌های سریع
        if penalty == "l1" and solver not in ["liblinear", "saga"]:
            return (0.0,)
        if penalty == "l2" and solver not in ["lbfgs", "liblinear", "sag", "saga"]:
            return (0.0,)
        if solver == "liblinear" and multi_class == "multinomial":
            return (0.0,)
        X, y, _, price_ser = PREP_SHARED.ready(DATA_TRAIN_SHARED,
                                            window=window,
                                            selected_features=[],
                                            mode="train")
        if X.empty:
            return (0.0,)

        hyper = {
            "C":C, "max_iter":max_iter, "tol":tol, "penalty":penalty,
            "solver":solver, "fit_intercept":fit_intercept,
            "class_weight":class_weight, "multi_class":multi_class
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_and_score_fold)(tr, ts, X, y, price_ser, hyper, calib_method)
            for tr, ts in tscv.split(X, y)
        )


        return (float(np.mean(scores)),)
    except Exception as e:
        LOGGER.error("evaluate_cv failed: %s", e)
        return (0.0,)

# ---------------------------------------------------------------------------
# ثبت در TOOLBOX
# ---------------------------------------------------------------------------
TOOLBOX.register("mate", tools.cxTwoPoint)
TOOLBOX.register("mutate", mutate_ind, indpb=0.2)
TOOLBOX.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------------------------
# کلاس Runner
# ---------------------------------------------------------------------------
class GeneticAlgorithmRunner:
    def __init__(self):
        self.neg_thr = 0.5
        self.pos_thr = 0.5
        self.final_cols: list[str] = []

    # ----------------------- main -----------------------
    def main(self):
        chk = load_checkpoint()
        if chk is not None:
            chk_gen, population, best_overall = chk
            start_gen = chk_gen + 1
            LOGGER.info("Resuming GA from generation %d …", start_gen)
        else:
            start_gen = 1
            best_overall = 0.0
            population = None 

        print("[main] Initialising PREPARE_DATA_FOR_TRAIN …")
        prep = PREPARE_DATA_FOR_TRAIN(main_timeframe="30T")
        raw  = prep.load_data()
        print("[main] Raw data loaded → rows =", len(raw), ", shape =", raw.shape)
        # -------------------- save tain raw with def ready for test-------------
        X_tail, _, _, _ = prep.ready(raw.tail(2001),
                                    selected_features=self.final_cols,
                                    mode="predict")
        X_tail.to_csv("raw_tail2000_clean.csv", index=False)
        print(f"[main] Saved cleaned tail to raw_tail2000_clean.csv, Number of cols = {X_tail.shape[1]}")
        # ------ sort & split ------
        time_col = f"{prep.main_timeframe}_time"
        raw[time_col] = pd.to_datetime(raw[time_col]); raw.sort_values(time_col, inplace=True)
        total = len(raw)
        train_end, thresh_end = int(total*0.85), int(total*0.90)
        data_tr  = raw.iloc[:train_end].copy()
        data_thr = raw.iloc[train_end:thresh_end].copy()
        data_te  = raw.iloc[thresh_end:].copy()
        print(f"[main] Split → train={len(data_tr)}, thresh={len(data_thr)}, test={len(data_te)}")

        # ------ pool ------
        n_proc = min(mp.cpu_count(), 8)
        pool = mp.Pool(n_proc, initializer=pool_init, initargs=(data_tr, prep))
        print(f"[main] Multiprocessing pool with {n_proc} workers created")
        # نصب هندلر قطع ناگهانی
        current_gen = {"val": start_gen - 1}          # برای closure
        _install_sig_handlers(lambda: current_gen["val"],
                            lambda: population,
                            lambda: best_overall)

        if "map" in TOOLBOX.__dict__:
            TOOLBOX.unregister("map")
        # ch = min(max(4, CFG.population_size // (2*mp.cpu_count())), 16)
        TOOLBOX.register("map", lambda f, it: list(pool.imap_unordered(f, it, chunksize=1)))

        TOOLBOX.register("init_individual", create_individual)
        if population is None:                       # اجرای تازه
            TOOLBOX.register("init_individual", create_individual)
            population = [TOOLBOX.init_individual() for _ in range(CFG.population_size)]
            LOGGER.info("Initial population generated")
            invalid = [ind for ind in population if not ind.fitness.valid]
            for ind, fit in zip(invalid, TOOLBOX.map(evaluate_cv, invalid)):
                ind.fitness.values = fit
            save_checkpoint(0, population, best_overall)      # ← پس از Gen-0
            logging.info("Checkpoint Gen-0 saved")
            print("Checkpoint Gen-0 saved")


        for ind, fit in zip(population, TOOLBOX.map(evaluate_cv, population)):
            ind.fitness.values = fit
        logging.info("[main] Initial fitnesses computed")
        LOGGER.info("[main] Initial fitnesses computed")

        save_checkpoint(gen - 1, population, best_overall)   # چک‌پوینت اضطراری
        LOGGER.info("Checkpoint auto-saved before starting gen %d", gen)

        for gen in range(start_gen, CFG.n_generations + 1):
            current_gen["val"] = gen
            print(f"[GA] Generation {gen}/{CFG.n_generations} …")
            logging.info(f"[GA] Generation {gen}/{CFG.n_generations} …")
           
            offspring = [copy.deepcopy(i) for i in TOOLBOX.select(population, len(population))]

            # crossover
            for i1, i2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CFG.cx_pb:
                    TOOLBOX.mate(i1, i2); del i1.fitness.values, i2.fitness.values
            # mutation
            for mut in offspring:
                if random.random() < CFG.mut_pb:
                    TOOLBOX.mutate(mut); del mut.fitness.values
            # evaluation
            invalid = [i for i in offspring if not i.fitness.valid]
            for ind, fit in zip(invalid, TOOLBOX.map(evaluate_cv, invalid)):
                ind.fitness.values = fit
            population[:] = offspring; gc.collect()
            
            save_checkpoint(gen, population, best_overall)
            logging.info(f"Checkpoint Gen-{gen} saved")
            print(f"Checkpoint Gen-{gen} saved")
            best_gen = tools.selBest(population, 1)[0]
            best_overall = max(best_overall, best_gen.fitness.values[0])
            print(f"[GA] Gen best = {best_gen.fitness.values[0]:.4f} · overall = {best_overall:.4f}")
            if best_gen.fitness.values[0] >= CFG.early_stopping_threshold:
                print("[GA] Early stopping reached!"); break

        best_ind = tools.selBest(population, 1)[0]
        print("[GA] Finished optimisation → best_score =", best_ind.fitness.values[0])

        # ------ final model ------
        final_model, feats = self._build_final_model(best_ind, data_tr, prep)
        if final_model is None:
            print("[ERROR] Final model could not be built!")
            pool.close(); pool.join(); return best_ind, best_ind.fitness.values[0]
        print("[main] Final model trained")

        self._run_thresholds(final_model, data_thr, prep, best_ind, feats)
        self._eval(final_model, data_te, prep, best_ind, feats, label="Test")
        self._save(final_model, best_ind, feats)
        print("[main] Model & thresholds saved 🎉")
        
        if os.path.isfile(CHECKPOINT):
            os.remove(CHECKPOINT)
            LOGGER.info("Checkpoint file removed – run completed")

        pool.close(); pool.join()
        return best_ind, best_ind.fitness.values[0]

    # ---------------- helpers ----------------
    def _build_final_model(self, ind, data_tr, prep):
        (C, max_iter, tol, penalty, solver,
        fit_intercept, class_weight, multi_class,
        window, calib_method) = ind

        X, y, feats, _ = prep.ready(data_tr,
                            window=window,
                            selected_features=None,
                            mode="train")

        if X.empty:
            return None, []

        hyper = {
            "C":C, "max_iter":max_iter, "tol":tol, "penalty":penalty,
            "solver":solver, "fit_intercept":fit_intercept,
            "class_weight":class_weight, "multi_class":multi_class
        }
        model = ModelPipeline(
            hyper,
            calibrate=True,
            calib_method=calib_method           # ← پاس ژن
        ).fit(X, y)
        
        self.final_cols = list(X.columns)
        print("[helper] Final model built with", len(self.final_cols), "features")

        scaler = model.pipeline.named_steps.get("scaler")
        X_proc = scaler.transform(X) if scaler is not None else X.values
        DriftChecker(verbose=False).fit_on_train(
            pd.DataFrame(X_proc, columns=X.columns),
            bins=10, quantile=False
        ).save_train_distribution("train_distribution.json")
        LOGGER.info("train_distribution.json saved (window=%d)", window)
        return model, feats

    def _run_thresholds(self, model, data_thr, prep, ind, feats):
        if data_thr.empty:
            print("[threshold] No threshold data … skipped"); return
        window = ind[8]
        
        X_thr, y_thr, _, _ = prep.ready(
                data_thr,
                window=window,
                selected_features=self.final_cols,   # ⬅️ دقیقاً ستون‌های Train
                mode="train",
        )

        X_thr = X_thr[self.final_cols]
        if X_thr.empty:                       # ← جلوگیری از خطا
            print("[threshold] X_thr empty – skipped")
            return
        y_prob = model.predict_proba(X_thr)[:, 1]

        tf = ThresholdFinder(steps=200, min_predictions_ratio=2/3)
        self.neg_thr, self.pos_thr, best_acc, *_ = tf.find_best_thresholds(y_prob, y_thr.values)
        print(f"[threshold] neg={self.neg_thr:.3f} · pos={self.pos_thr:.3f} · acc={best_acc:.4f}")

    def _eval(self, model, data_part, prep, ind, feats, label="Test"):
        """
        گزارش نهایی:
        • معیار اصلی GA  (Sharpe-Ratio − MaxDD)
        • Balanced-Accuracy
        • تعداد درست، غلط و بدون پیش‌بینی (∅)
        """
        if data_part.empty:
            print(f"[{label}] Empty dataset")
            return

        window = ind[8]                       # ژنِ اندازهٔ پنجره
        # ❶ چهار خروجی می‌گیریم؛ price_ser همان سریِ 30T_close است
        X, y, _, price_ser = prep.ready(
            data_part,
            window=window,
            selected_features=feats,
            mode="train",
        )
        X = X[self.final_cols]                # ستون‌های نهایی مدل
        if X.empty:                           # ← جلوگیری از خطا
            print(f"[{label}] X empty – skipped")
            return
        
        # ── پیش‌بینی با دو آستانه ─────────────────────────────
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = np.full_like(y, -1, dtype=int)   # -1 ⇒ No-Trade
        y_pred[y_prob <= self.neg_thr] = 0
        y_pred[y_prob >= self.pos_thr] = 1

        mask = y_pred != -1                    # تنها نمونه‌های معامله‌شده
        conf = float(mask.mean())              # نسبتِ دارای پیش‌بینی
        # ───────── Balanced-Accuracy ─────────
        if mask.any():
            bal_acc = balanced_accuracy_score(y[mask], y_pred[mask])
        else:
            bal_acc = 0.0
        # ───────── معیار مالی (Sharpe − MaxDD) ─────────
        if mask.any():
            prices = price_ser[mask].astype(float).values
            if len(prices) >= 2:
                ret_mkt = np.diff(prices) / prices[:-1]
                pos_shift = np.roll(y_pred[mask], 1)
                pos_shift[0] = 0
                pos_shift = pos_shift[:-1]                  # طول = len(ret_mkt)
                ret_str = np.where(pos_shift == 1, ret_mkt, 0.0)

                ret_ser = pd.Series(ret_str)
                if ret_ser.std() > 0:
                    periods_per_year = 252 * 48             # کندل ۳۰دقیقه‌ای
                    sharpe = (ret_ser.mean() / ret_ser.std()
                            * np.sqrt(periods_per_year))
                else:
                    sharpe = 0.0

                cum_eq = ret_ser.add(1).cumprod()
                maxdd = (cum_eq.cummax() - cum_eq).max()
                norm_sharpe = (np.tanh(sharpe / 5) + 1) / 2
                norm_dd     = 1.0 - min(maxdd, 1.0)
                score       = 0.4 * norm_sharpe + 0.4 * norm_dd + 0.2 * bal_acc
                     # معیار اصلی
            else:
                score = sharpe = maxdd = 0.0
        else:
            score = sharpe = maxdd = 0.0



        # ───────── شمارش درست / غلط / ∅ ─────────
        total       = len(y_pred)
        unpred_n    = int((y_pred == -1).sum())              # ∅
        pred_n      = total - unpred_n
        correct_n   = int(((y_pred == y) & (y_pred != -1)).sum())
        incorrect_n = pred_n - correct_n

        print(
            f"[{label}] size={total} · conf={conf:.2f} · "
            f"Score={score:.4f} (Sharpe-DD) · BalAcc={bal_acc:.4f} · "
            f"Correct ={correct_n} Incorrect ={incorrect_n} Unpredict ={unpred_n}"
        )

    def _save(self, model, ind, feats):
        (C, max_iter, tol, penalty, solver,
         fit_intercept, class_weight, multi_class,
         window, calib_method) = ind
        hyper = {
            "C":C,"max_iter":max_iter,"tol":tol,"penalty":penalty,
            "solver":solver,"fit_intercept":fit_intercept,
            "class_weight":class_weight,"multi_class":multi_class
        }
        scaler = model.pipeline.named_steps.get("scaler")
        ModelSaver().save_full(
            pipeline=model.pipeline, hyperparams=hyper, scaler=scaler,
            window_size=window, neg_thr=self.neg_thr, pos_thr=self.pos_thr,
            feats=feats, feat_mask=None, train_window_cols=self.final_cols,
        )
        print("[save] All artefacts persisted to disk successfull")

# ---------------------------------------------------------------------------
# اجرای مستقیم
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runner = GeneticAlgorithmRunner()
    best_ind, best_score = runner.main()
    print("[MAIN] GA done → best_score =", best_score)
