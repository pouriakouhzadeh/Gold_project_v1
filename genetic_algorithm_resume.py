#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genetic_algorithm_resume.py – نسخهٔ اصلی + قابلیت checkpoint/resume
-------------------------------------------------------------------
برای شروع تازه:
    python genetic_algorithm_resume.py --reset
برای ادامهٔ خودکار:
    python genetic_algorithm_resume.py
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
#                   آزادسازی هسته‌های عددی (همانند نسخهٔ اصلی)
# ---------------------------------------------------------------------------
import os, multiprocessing as mp
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","TBB_NUM_THREADS"):
    os.environ.pop(_v, None)

# ---------------------------------------------------------------------------
#                   شتاب oneAPI (اختیاری)
# ---------------------------------------------------------------------------
from sklearnex import patch_sklearn  # type: ignore
patch_sklearn(verbose=False)

# ---------------------------------------------------------------------------
#                   کتابخانه‌های استاندارد و ثالث
# ---------------------------------------------------------------------------
import argparse, copy, gc, logging, pickle, random, signal, sys, warnings
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Tuple, List

import numpy as np
import pandas as pd
from deap import base, creator, tools
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from model_pipeline import ModelPipeline          # type: ignore
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN  # type: ignore
from ModelSaver import ModelSaver                 # type: ignore
from threshold_finder import ThresholdFinder      # type: ignore
from drift_checker import DriftChecker            # type: ignore

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = int(os.environ.get("GA_GLOBAL_SEED", 2025))
random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
#                   لاگ‌گیری – بدون تغییر
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
_fmt = "%(asctime)s %(levelname)s: %(message)s"
_date = "%Y-%m-%d %H:%M:%S"
_fhdl = RotatingFileHandler("genetic_algorithm.log", maxBytes=2_000_000,
                            backupCount=3, encoding="utf-8")
_fhdl.setFormatter(logging.Formatter(_fmt, _date))
_chdl = logging.StreamHandler(); _chdl.setFormatter(logging.Formatter(_fmt, _date))
LOGGER.addHandler(_fhdl); LOGGER.addHandler(_chdl)
for _m in ("sklearnex","sklearn","daal4py"):
    logging.getLogger(_m).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
#                   پیکربندی GA
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
#                   DEAP primitives (بدون تغییر)
# ---------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
TOOLBOX = base.Toolbox()

# -------------------- توابع تصادفی (همان نسخهٔ اصلی) --------------------
rand_C             = lambda: 10 ** random.uniform(-4, 2)
rand_max_iter      = lambda: random.randint(8_000, 10_000)
rand_tol           = lambda: 10 ** random.uniform(-5, -2)
rand_penalty       = lambda: random.choice(["l1", "l2"])
rand_solver        = lambda: random.choice(["sag", "saga"])
rand_calib         = lambda: random.choice(["sigmoid", "isotonic"])
rand_fit_intercept = lambda: random.choice([True, False])
rand_class_weight  = lambda: random.choice([None, "balanced"])
rand_multi_class   = lambda: random.choice(["auto", "ovr", "multinomial"])
rand_window_size   = lambda: random.choice(CFG.possible_window_sizes)

# ---------------------------------------------------------------------------
#                   suppress warnings helper
# ---------------------------------------------------------------------------
def _suppress_warnings():
    import warnings as _w; from sklearn.exceptions import ConvergenceWarning as _CW
    _w.filterwarnings("ignore", category=_CW)
    _w.filterwarnings("ignore", message=".*max_iter.*")

# ---------------------------------------------------------------------------
#                   Pool initialiser و دادهٔ مشترک
# ---------------------------------------------------------------------------
DATA_TRAIN_SHARED: pd.DataFrame | None = None
PREP_SHARED: PREPARE_DATA_FOR_TRAIN | None = None

def pool_init(data_train: pd.DataFrame, prep: PREPARE_DATA_FOR_TRAIN):
    _suppress_warnings()
    global DATA_TRAIN_SHARED, PREP_SHARED
    DATA_TRAIN_SHARED, PREP_SHARED = data_train, prep
    print("[Pool] Shared globals initialised")

# ---------------------------------------------------------------------------
#                   ایجاد فرد و جهش
# ---------------------------------------------------------------------------
def create_individual() -> creator.Individual:
    p = rand_penalty(); solver = rand_solver() if p=="l2" else "saga"
    multi = rand_multi_class()
    if solver == "liblinear" and multi == "multinomial": multi = "ovr"
    return creator.Individual([rand_C(),rand_max_iter(),rand_tol(),p,solver,
                               rand_fit_intercept(),rand_class_weight(),multi,
                               rand_window_size(),rand_calib()])

def mutate_ind(ind: creator.Individual, indpb: float = 0.2):
    for i in range(len(ind)):
        if random.random() >= indpb: continue
        if   i==0: ind[0]=rand_C()
        elif i==1: ind[1]=rand_max_iter()
        elif i==2: ind[2]=rand_tol()
        elif i==3: ind[3]=rand_penalty()
        elif i==4: ind[4]=rand_solver()
        elif i==5: ind[5]=rand_fit_intercept()
        elif i==6: ind[6]=rand_class_weight()
        elif i==7: ind[7]=rand_multi_class()
        elif i==8: ind[8]=rand_window_size()
        elif i==9: ind[9]=rand_calib()
    p,solver,multi = ind[3], ind[4], ind[7]
    if p=="l1" and solver not in ["liblinear","saga"]: ind[4]="saga"
    if p=="l2" and solver not in ["lbfgs","liblinear","sag","saga"]: ind[4]="sag"
    if solver=="liblinear" and multi=="multinomial": ind[7]="ovr"
    return (ind,)

# ---------------------------------------------------------------------------
#                   ارزیابی فولد و fitness
# ---------------------------------------------------------------------------
def _fit_and_score_fold(tr_idx, ts_idx, X_full, y_full, price_series, hyper, calib_method):
    import numpy as _np, pandas as _pd
    X_tr_raw,y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
    X_ts_raw,y_ts = X_full.iloc[ts_idx], y_full.iloc[ts_idx]
    feats = PREP_SHARED.select_features(X_tr_raw, y_tr)
    if not feats: return 0.0
    X_tr,X_ts = X_tr_raw[feats], X_ts_raw[feats]
    pipe = ModelPipeline(hyper, calibrate=True, calib_method=calib_method).fit(X_tr,y_tr)
    y_pred = pipe.pipeline.predict(X_ts)

    prices = price_series.iloc[ts_idx].astype(float).values
    if len(prices)<2: return 0.0
    ret_mkt = _np.diff(prices)/prices[:-1]
    pos_shift = _np.roll(y_pred,1); pos_shift[0]=0; pos_shift=pos_shift[:-1]
    ret_str = _np.where(pos_shift==1, ret_mkt, 0.0)

    ret_ser = _pd.Series(ret_str); std = ret_ser.std()
    if std is None or std==0 or _np.isnan(std): return 0.0
    sharpe = ret_ser.mean()/std*_np.sqrt(252*48)
    cum_eq = ret_ser.add(1).cumprod(); max_dd = (cum_eq.cummax()-cum_eq).max()
    bal_acc = balanced_accuracy_score(y_ts, y_pred)
    if bal_acc<0.55: return 0.0
    norm_sharpe=(np.tanh(sharpe/5)+1)/2; norm_dd=1.0-min(max_dd,1.0)
    return 0.4*norm_sharpe + 0.4*norm_dd + 0.2*bal_acc

def evaluate_cv(ind):
    try:
        if DATA_TRAIN_SHARED is None or PREP_SHARED is None:
            raise RuntimeError("Shared globals not ready!")
        (C,max_iter,tol,penalty,solver,fit_intercept,class_weight,
         multi_class,window,calib_method) = ind
        if penalty=="l1" and solver not in ["liblinear","saga"]: return (0.0,)
        if penalty=="l2" and solver not in ["lbfgs","liblinear","sag","saga"]: return (0.0,)
        if solver=="liblinear" and multi_class=="multinomial": return (0.0,)

        X,y,_,price_ser = PREP_SHARED.ready(DATA_TRAIN_SHARED, window=window,
                                            selected_features=[], mode="train")
        if X.empty: return (0.0,)
        hyper={"C":C,"max_iter":max_iter,"tol":tol,"penalty":penalty,"solver":solver,
               "fit_intercept":fit_intercept,"class_weight":class_weight,
               "multi_class":multi_class}
        tscv = TimeSeriesSplit(n_splits=3)
        scores = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_and_score_fold)(tr,ts,X,y,price_ser,hyper,calib_method)
            for tr,ts in tscv.split(X,y))
        return (float(np.mean(scores)),)
    except Exception as e:
        LOGGER.error("evaluate_cv failed: %s",e); return (0.0,)

TOOLBOX.register("mate", tools.cxTwoPoint)
TOOLBOX.register("mutate", mutate_ind, indpb=0.2)
TOOLBOX.register("select", tools.selTournament, tournsize=3)

# ---------------------------------------------------------------------------
#                   مدیریت چک‌پوینت
# ---------------------------------------------------------------------------
def _save_checkpoint(state: dict, path: str = CHECKPOINT_FILE):
    try:
        with open(path,"wb") as f: pickle.dump(state,f,pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Checkpoint saved → %s", path)
    except Exception as exc:
        LOGGER.error("Cannot save checkpoint: %s", exc)

def _load_checkpoint(path: str = CHECKPOINT_FILE):
    if not os.path.exists(path): return None
    try:
        with open(path,"rb") as f: st=pickle.load(f)
        LOGGER.info("Checkpoint loaded ← %s", path); return st
    except Exception as exc:
        LOGGER.error("Cannot load checkpoint: %s", exc); return None

# ---------------------------------------------------------------------------
#                   Helper‑های اصلی (بدون تغییر)
# ---------------------------------------------------------------------------
class GeneticAlgorithmRunner:
    def __init__(self, ckpt_path: str = CHECKPOINT_FILE):
        self.ckpt_path = ckpt_path
        self.current_gen = 0
        self.best_overall = 0.0
        self.population: List[creator.Individual] = []
        self.neg_thr = 0.5; self.pos_thr = 0.5
        self.final_cols: List[str] = []

    # ---------- helpers copied as‑is ----------
    def _build_final_model(self, ind, data_tr, prep):
        (C,max_iter,tol,penalty,solver,fit_intercept,class_weight,
         multi_class,window,calib_method)=ind
        X,y,feats,_ = prep.ready(data_tr, window=window,
                                 selected_features=None, mode="train")
        if X.empty: return None,[]
        hyper={"C":C,"max_iter":max_iter,"tol":tol,"penalty":penalty,"solver":solver,
               "fit_intercept":fit_intercept,"class_weight":class_weight,
               "multi_class":multi_class}
        model = ModelPipeline(hyper, calibrate=True,
                              calib_method=calib_method).fit(X,y)
        self.final_cols = list(X.columns)
        scaler = model.pipeline.named_steps.get("scaler")
        X_proc = scaler.transform(X) if scaler is not None else X.values
        DriftChecker(verbose=False).fit_on_train(
            pd.DataFrame(X_proc,columns=X.columns),
            bins=10,quantile=False).save_train_distribution("train_distribution.json")
        return model, feats

    def _run_thresholds(self, model, data_thr, prep, ind, feats):
        if data_thr.empty: return
        window = ind[8]
        X_thr,y_thr,_,_ = prep.ready(data_thr, window=window,
                                     selected_features=self.final_cols,
                                     mode="train")
        X_thr = X_thr[self.final_cols]
        if X_thr.empty: return
        y_prob = model.predict_proba(X_thr)[:,1]
        tf = ThresholdFinder(steps=200, min_predictions_ratio=2/3)
        self.neg_thr, self.pos_thr, *_ = tf.find_best_thresholds(y_prob, y_thr.values)

    def _eval(self, model, data_part, prep, ind, feats, label="Test"):
        if data_part.empty: return
        window = ind[8]
        X,y,_,price_ser = prep.ready(data_part, window=window,
                                     selected_features=feats, mode="train")
        X = X[self.final_cols]
        if X.empty: return
        y_prob = model.predict_proba(X)[:,1]
        y_pred = np.full_like(y,-1)
        y_pred[y_prob<=self.neg_thr]=0; y_pred[y_prob>=self.pos_thr]=1
        mask = y_pred!=-1
        if mask.any():
            bal_acc = balanced_accuracy_score(y[mask], y_pred[mask])
            prices = price_ser[mask].astype(float).values
            if len(prices)>=2:
                ret = np.diff(prices)/prices[:-1]
                pos = np.roll(y_pred[mask],1); pos[0]=0; pos=pos[:-1]
                ret_str = np.where(pos==1, ret, 0.0)
                ret_ser = pd.Series(ret_str)
                sharpe = (ret_ser.mean()/ret_ser.std())*np.sqrt(252*48) if ret_ser.std()>0 else 0.0
                cum = ret_ser.add(1).cumprod(); maxdd = (cum.cummax()-cum).max()
                norm_sh=(np.tanh(sharpe/5)+1)/2; norm_dd = 1.0-min(maxdd,1.0)
                score = 0.4*norm_sh + 0.4*norm_dd + 0.2*bal_acc
            else:
                score=sharpe=maxdd=0.0
        else:
            bal_acc=score=sharpe=maxdd=0.0
        total=len(y_pred); unpred=(y_pred==-1).sum()
        print(f"[{label}] size={total} conf={float(mask.mean()):.2f} "
              f"Score={score:.4f} BalAcc={bal_acc:.4f} unpred={unpred}")

    def _save(self, model, ind, feats):
        (C,max_iter,tol,penalty,solver,fit_intercept,class_weight,
         multi_class,window,_) = ind
        hyper={"C":C,"max_iter":max_iter,"tol":tol,"penalty":penalty,"solver":solver,
               "fit_intercept":fit_intercept,"class_weight":class_weight,
               "multi_class":multi_class}
        scaler = model.pipeline.named_steps.get("scaler")
        ModelSaver().save_full(pipeline=model.pipeline, hyperparams=hyper,
                               scaler=scaler, window_size=window,
                               neg_thr=self.neg_thr, pos_thr=self.pos_thr,
                               feats=feats, feat_mask=None,
                               train_window_cols=self.final_cols)

    # ---------- checkpoint util ----------
    def _save_ckpt(self):
        _save_checkpoint({"gen":self.current_gen,
                          "population":self.population,
                          "best_overall":self.best_overall,
                          "rand_state":random.getstate(),
                          "np_state":np.random.get_state()},
                         self.ckpt_path)

    # ---------- main ----------
    def main(self):
        # 0) load checkpoint
        st = _load_checkpoint(self.ckpt_path)
        if st:
            self.current_gen = st["gen"]; self.population = st["population"]
            self.best_overall = st["best_overall"]
            random.setstate(st["rand_state"]); np.random.set_state(st["np_state"])
            LOGGER.info("Resuming from generation %d", self.current_gen)
        else:
            LOGGER.info("Starting new GA run")

        # 1) data prep
        prep = PREPARE_DATA_FOR_TRAIN(main_timeframe="30T")
        raw  = prep.load_data()
        tcol = f"{prep.main_timeframe}_time"
        raw[tcol] = pd.to_datetime(raw[tcol]); raw.sort_values(tcol,inplace=True)
        total=len(raw); tr_end,th_end = int(total*0.85), int(total*0.90)
        data_tr, data_thr, data_te = raw.iloc[:tr_end], raw.iloc[tr_end:th_end], raw.iloc[th_end:]

        # 2) pool
        nproc=min(mp.cpu_count(),8)
        pool = mp.Pool(nproc, initializer=pool_init, initargs=(data_tr,prep))
        if "map" in TOOLBOX.__dict__: TOOLBOX.unregister("map")
        TOOLBOX.register("map",lambda f,it:list(pool.imap_unordered(f,it,chunksize=1)))
        TOOLBOX.register("init_individual", create_individual)

        # 3) initial population
        if not self.population:
            self.population=[TOOLBOX.init_individual() for _ in range(CFG.population_size)]
            for ind,fit in zip(self.population,TOOLBOX.map(evaluate_cv,self.population)):
                ind.fitness.values = fit
            self.best_overall = max(i.fitness.values[0] for i in self.population)
            self.current_gen  = 0
            self._save_ckpt()

        # 4) signal handler
        def _sig(s,_f):
            LOGGER.warning("Signal %s caught – checkpointing",s)
            self._save_ckpt(); pool.close(); pool.join(); sys.exit(0)
        for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s,_sig)

        # 5) evolutionary loop
        for gen in range(self.current_gen+1, CFG.n_generations+1):
            self.current_gen = gen
            print(f"[GA] Generation {gen}/{CFG.n_generations} …")
            offspring=[copy.deepcopy(i) for i in tools.selTournament(self.population,len(self.population),tournsize=3)]
            for i1,i2 in zip(offspring[::2],offspring[1::2]):
                if random.random()<CFG.cx_pb:
                    tools.cxTwoPoint(i1,i2); del i1.fitness.values,i2.fitness.values
            for m in offspring:
                if random.random()<CFG.mut_pb: mutate_ind(m); del m.fitness.values
            invalid=[i for i in offspring if not i.fitness.valid]
            for ind,fit in zip(invalid,TOOLBOX.map(evaluate_cv,invalid)):
                ind.fitness.values=fit
            self.population[:] = offspring; gc.collect()

            best_gen = tools.selBest(self.population,1)[0]
            self.best_overall = max(self.best_overall,best_gen.fitness.values[0])
            print(f"[GA] Gen best={best_gen.fitness.values[0]:.4f} overall={self.best_overall:.4f}")

            if best_gen.fitness.values[0] >= CFG.early_stopping_threshold:
                print("[GA] Early stopping reached")
                self._save_ckpt(); break
            self._save_ckpt()

        # 6) final model & evaluation
        best_ind = tools.selBest(self.population,1)[0]
        print("[GA] Finished optimisation → best_score =", best_ind.fitness.values[0])
        final_model, feats = self._build_final_model(best_ind, data_tr, prep)
        if final_model is None:
            LOGGER.error("Final model could not be built")
            pool.close(); pool.join(); return

        self._run_thresholds(final_model, data_thr, prep, best_ind, feats)
        self._eval(final_model, data_te, prep, best_ind, feats, label="Test")
        self._save(final_model, best_ind, feats)
        print("🎉 Model & thresholds saved")

        pool.close(); pool.join()
        if os.path.exists(self.ckpt_path):
            os.remove(self.ckpt_path)
            LOGGER.info("Checkpoint removed – run completed successfully")

# ---------------------------------------------------------------------------
#                               CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GA optimiser with resume")
    p.add_argument("--checkpoint","-ckpt",default=CHECKPOINT_FILE, help="checkpoint path")
    p.add_argument("--reset",action="store_true", help="delete checkpoint and start fresh")
    args = p.parse_args()

    if args.reset and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
        print("🗑️  Checkpoint removed – starting fresh")

    runner = GeneticAlgorithmRunner(ckpt_path=args.checkpoint)
    ind, score = runner.main()
    print("[MAIN] GA done → best_score =", score)
