#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-accelerated version of ensemble_deep.py

- منطق، اسپلـیت دیتاست، گرید سرچ، ThresholdFinder، گزارش‌ها، ensemble و ... 
  با اسکریپت CPU کاملاً یکسان است.
- تنها تفاوت: تمام مدل‌های deep روی GPU (RX580/ROCm) اجرا می‌شوند.
"""

# ---------- بخش 0: آرگومان‌ها و قفل‌کردن GPU قبل از import torch ----------
import os, sys, argparse, math, itertools, logging
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0,
                    help="ایندکس GPU داخل کانتینر (معمولاً 0)")
parser.add_argument("--batch", type=int, default=512,
                    help="اندازه batch برای آموزش")
parser.add_argument("--grid-shard", type=str, default="1/1",
                    help="برای شارد کردن گرید سرچ، مثل 1/2 یا 2/2 (فعلاً معمولاً 1/1)")
args, _ = parser.parse_known_args()

# این متغیرها باید قبل از import torch ست شوند
os.environ.setdefault("ROCR_VISIBLE_DEVICES", str(args.gpu))
os.environ.setdefault("HIP_VISIBLE_DEVICES",  str(args.gpu))
# برای Polaris لازم است
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "8.0.3")
# پایدارسازی حافظه HIP
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:64")
# پرهیز از SDMA برای پایداری بیشتر
os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("HSA_ENABLE_PEER_SDMA", "0")

# ---------- بقیه import ها (بعد از تنظیم env) ----------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from threshold_finder import ThresholdFinder

# ==============================
# Logging (console + file)
# ==============================
LOG_FILE = "ensemble_deep_gpu.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def stage(title: str):
    sep = "=" * 80
    log.info("\n%s\n🟢 STAGE: %s\n%s", sep, title, sep)

# ==============================
# Torch helpers
# ==============================
def seed_all(seed=2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# دستگاه به صورت global (بعداً در main مقداردهی می‌شود)
device = torch.device("cpu")

# ==============================
# مدل‌ها
# ==============================
class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.seq(x)

class MLPClassifier(nn.Module):
    """Simple strong MLP for tabular (همان مدل اسکریپت CPU)"""
    def __init__(self, in_dim, hidden=[512, 256], dropout=0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(MLPBlock(last, h, dropout))
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)

    def forward(self, x):
        z = self.backbone(x)
        logit = self.head(z)
        return logit.squeeze(1)  # (N,)

class ResNetMLP(nn.Module):
    """ResNet-style MLP block for deep tabular (همان CPU)"""
    def __init__(self, in_dim, width=512, depth=4, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                    nn.BatchNorm1d(width),
                )
                for _ in range(depth)
            ]
        )
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            res = x
            x = b(x)
            x = self.act(x + res)
        return self.head(x).squeeze(1)

class WideDropMLP(nn.Module):
    """Wider MLP with stronger dropout to fight overfit/bias (همان CPU)"""
    def __init__(self, in_dim, hidden=[768, 384], dropout=0.35):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(MLPBlock(last, h, dropout))
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

# ==============================
# Training / Evaluation helpers
# ==============================
def make_loader(X, y, batch=512, shuffle=False):
    X_t = torch.from_numpy(X.astype(np.float32))
    if y is None:
        ds = TensorDataset(X_t)
    else:
        y_t = torch.from_numpy(y.astype(np.float32))
        ds = TensorDataset(X_t, y_t)
    # num_workers=0 برای پایداری روی ROCm/داکر،
    # pin_memory=False چون داده‌ها روی CPU می‌مانند و فقط batchها به GPU منتقل می‌شوند.
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

@torch.no_grad()
def predict_proba(model, X_np, batch=2048):
    """دقیقاً همان منطق اسکریپت CPU؛ فقط device=GPU"""
    model.eval()
    dl = make_loader(X_np, None, batch=batch, shuffle=False)
    outs = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=False)
        logit = model(xb)
        prob = torch.sigmoid(logit)
        outs.append(prob.detach().cpu().numpy())
    p = np.concatenate(outs, axis=0)
    # برگرداندن (N,2) مثل sklearn
    return np.stack([1 - p, p], axis=1)

def train_one(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    lr=1e-3,
    wd=1e-4,
    epochs=30,
    batch=512,
    pos_weight=None,
    patience=6,
):
    """کپی تابع CPU، فقط با device=GPU"""
    global device
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if pos_weight is not None:
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
    else:
        crit = nn.BCEWithLogitsLoss()

    best = {"f1": -1, "state": None, "epoch": -1}
    dl_tr = make_loader(X_tr, y_tr, batch=batch, shuffle=True)

    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            logit = model(xb)
            loss = crit(logit, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(xb)
        train_loss = loss_sum / len(X_tr)

        # eval روی validation
        prob_val = predict_proba(model, X_val)[:, 1]
        y_hat = (prob_val >= 0.5).astype(int)

        acc = accuracy_score(y_val, y_hat)
        bal = balanced_accuracy_score(y_val, y_hat)
        f1 = f1_score(y_val, y_hat)

        log.info(
            f"      ep={ep:02d} | train_loss={train_loss:.4f} | "
            f"val_acc={acc:.4f} | val_bal={bal:.4f} | val_f1={f1:.4f}"
        )

        # bias penalty (prevent always-one-class)
        buy_ratio = float(np.mean(y_hat == 1)) if len(y_hat) else 0.0
        sell_ratio = float(np.mean(y_hat == 0)) if len(y_hat) else 0.0
        penalty = 0.1 if (buy_ratio > 0.9 or sell_ratio > 0.9) else 0.0
        f1_eff = max(0.0, f1 - penalty)

        if f1_eff > best["f1"]:
            best = {
                "f1": f1_eff,
                "state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "epoch": ep,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    f"      early-stop at ep={ep} "
                    f"(best@{best['epoch']}, f1_eff={best['f1']:.4f})"
                )
                break

    # load best
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        model = model.to(device)
    return model

# ------------------------------
# Grid search (با قابلیت shard مثل نسخه GPU قبلی)
# ------------------------------
def all_param_combos(grid: dict):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def shard_combos(combos, shard_str: str):
    """
    shard_str مثل "1/2" یا "2/3"
    فقط برای این است که اگر خواستی چند کانتینر موازی اجرا کنی،
    هر کدام بخشی از گرید را آموزش دهند.
    برای 1/1 همه‌ی کامبوها استفاده می‌شوند (مثل CPU).
    """
    p = shard_str.strip().split("/")
    shard_idx = int(p[0]) - 1
    shard_mod = int(p[1])
    combos = list(combos)
    return [c for i, c in enumerate(combos) if i % shard_mod == shard_idx]

def grid_search_torch(
    model_name,
    model_ctor,
    grid,
    X_tr,
    y_tr,
    X_val,
    y_val,
    class_pos_weight,
    shard="1/1",
):
    """
    همان منطق grid_search اسکریپت CPU
    فقط قابلیت shard اضافه شده (اگر shard="1/1" باشد دقیقاً معادل CPU است).
    """
    best = None
    all_combos = list(all_param_combos(grid))
    shard_list = shard_combos(all_combos, shard)

    for ci, params in enumerate(shard_list, 1):
        log.info(
            f"   ▶ {model_name} try params={params} "
            f"[combo {ci}/{len(shard_list)} | shard {shard}]"
        )
        model = model_ctor(**params)
        model = train_one(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            lr=params.get("lr", 1e-3),
            wd=params.get("weight_decay", 1e-4),
            epochs=params.get("epochs", 30),
            batch=params.get("batch_size", 512),
            pos_weight=class_pos_weight,
            patience=params.get("patience", 6),
        )
        # evaluate on val
        prob = predict_proba(model, X_val)[:, 1]
        y_hat = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_val, y_hat)
        bal = balanced_accuracy_score(y_val, y_hat)
        f1 = f1_score(y_val, y_hat)
        buy_ratio = float(np.mean(y_hat == 1)) if len(y_hat) else 0.0
        sell_ratio = float(np.mean(y_hat == 0)) if len(y_hat) else 0.0
        penalty = 0.1 if (buy_ratio > 0.9 or sell_ratio > 0.9) else 0.0
        f1_eff = max(0.0, f1 - penalty)
        log.info(
            f"   ◀ {model_name} val_acc={acc:.4f} val_bal={bal:.4f} "
            f"val_f1={f1:.4f} penalty={penalty:.2f}"
        )

        entry = {
            "Model": model_name,
            "Params": params,
            "ModelObj": model,
            "val_acc": acc,
            "val_bal": bal,
            "val_f1": f1,
            "bias_penalty": penalty,
            "f1_eff": f1_eff,
        }
        if (best is None) or (entry["f1_eff"] > best["f1_eff"]):
            best = entry
    return best

# ==============================
# MAIN
# ==============================
def main():
    global device

    # انتخاب و چک GPU
    if not torch.cuda.is_available():
        log.info(
            "HIP ver: %s | cuda.is_available=%s | device=cpu",
            getattr(torch.version, "hip", None),
            torch.cuda.is_available(),
        )
        sys.exit(
            "❌ GPU در دسترس نیست. "
            "متغیرهای ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES و پاس‌دادن /dev/kfd و /dev/dri را بررسی کن."
        )

    device = torch.device(f"cuda:{args.gpu}")
    log.info(
        "HIP ver: %s | cuda.is_available=%s | device=%s",
        getattr(torch.version, "hip", None),
        torch.cuda.is_available(),
        device,
    )

    seed_all(2025)

    # 1) LOAD & SPLIT  (دقیقاً مثل اسکریپت CPU)
    stage("Loading and splitting dataset (Train / Threshold / Test / Live)")
    data_path = "prepared_train_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ File 'prepared_train_data.csv' not found.")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("❌ 'target' column missing.")

    X = df.drop(columns=["target"]).select_dtypes(include=[np.number]).copy()
    y = df["target"].astype(int).values

    # handle inf/nan
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    total_len = len(X)
    LIVE_SIZE = 3000
    THRESH_SIZE = int(total_len * 0.05)

    if total_len <= (LIVE_SIZE + THRESH_SIZE + 1000):
        log.warning(
            "⚠️ Dataset is small relative to LIVE/THRESH sizes; splits may be tight."
        )

    # split by time order (مثل CPU)
    X_train_full = X.iloc[: -(LIVE_SIZE + THRESH_SIZE)]
    y_train_full = y[: -(LIVE_SIZE + THRESH_SIZE)]

    X_thresh = X.iloc[-(LIVE_SIZE + THRESH_SIZE) : -LIVE_SIZE]
    y_thresh = y[-(LIVE_SIZE + THRESH_SIZE) : -LIVE_SIZE]

    X_live = X.iloc[-LIVE_SIZE:]
    y_live = y[-LIVE_SIZE:]

    # ستون قیمت برای گزارش سیگنال‌ها
    price_col = next((c for c in X.columns if "close" in c.lower()), None)
    if price_col is not None:
        price_live = X_live[price_col].values
    else:
        price_live = np.arange(len(X_live))

    # train/test split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=False
    )

    log.info(
        f"✅ Split summary: Train={len(X_train)}, Threshold={len(X_thresh)}, "
        f"Test={len(X_test)}, Live={len(X_live)}"
    )

    # 2) SCALING (مثل CPU)
    stage("Fitting StandardScaler on Train and transforming all parts")
    scaler = StandardScaler()
    scaler.fit(X_train.values)

    X_train_s = scaler.transform(X_train.values)
    X_test_s = scaler.transform(X_test.values)
    X_thresh_s = scaler.transform(X_thresh.values)
    X_live_s = scaler.transform(X_live.values)

    # validation slice از انتهای train (time-aware)
    val_size = max(2000, int(0.15 * len(X_train_s)))
    X_tr, X_val = X_train_s[:-val_size], X_train_s[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    # class weight برای BCE
    pos_ratio = float(np.mean(y_tr == 1)) if len(y_tr) else 0.5
    if 0 < pos_ratio < 1:
        pos_weight = float((1 - pos_ratio) / max(1e-9, pos_ratio))
    else:
        pos_weight = None
    log.info(f"Class balance on train: pos_ratio={pos_ratio:.3f} → pos_weight={pos_weight}")

    # 3) DEFINE MODELS & GRIDS (دقیقاً مثل CPU، فقط batch از args.batch می‌آید)
    stage("Defining Deep models & hyperparameter grids")
    in_dim = X_train_s.shape[1]

    MODELS = [
        (
            "MLP_Baseline",
            lambda lr, weight_decay, hidden_0, hidden_1, dropout, epochs, batch_size, patience: MLPClassifier(
                in_dim, hidden=[hidden_0, hidden_1], dropout=dropout
            ),
            {
                "lr": [1e-3, 5e-4],
                "weight_decay": [1e-4, 1e-5],
                "hidden_0": [512, 384],
                "hidden_1": [256, 192],
                "dropout": [0.2, 0.3],
                "epochs": [25],
                "batch_size": [args.batch],
                "patience": [6],
            },
        ),
        (
            "ResNet_MLP",
            lambda lr, weight_decay, width, depth, dropout, epochs, batch_size, patience: ResNetMLP(
                in_dim, width=width, depth=depth, dropout=dropout
            ),
            {
                "lr": [1e-3, 7e-4],
                "weight_decay": [1e-4, 5e-5],
                "width": [512],
                "depth": [3, 4],
                "dropout": [0.2, 0.3],
                "epochs": [28],
                "batch_size": [args.batch],
                "patience": [6],
            },
        ),
        (
            "WideDrop_MLP",
            lambda lr, weight_decay, hidden_0, hidden_1, dropout, epochs, batch_size, patience: WideDropMLP(
                in_dim, hidden=[hidden_0, hidden_1], dropout=dropout
            ),
            {
                "lr": [1e-3],
                "weight_decay": [1e-4, 5e-5],
                "hidden_0": [768],
                "hidden_1": [384, 320],
                "dropout": [0.35, 0.4],
                "epochs": [24],
                "batch_size": [args.batch],
                "patience": [6],
            },
        ),
    ]

    # 4) TRAIN + GRID + VALIDATION (مثل CPU، با grid-shard)
    stage("Training (grid-search per model) with bias control & early stopping")
    results = []
    for name, ctor, grid in MODELS:
        # تبدیل ctor به model_ctor(**params)
        def model_ctor(**p):
            return ctor(**p)

        best = grid_search_torch(
            name,
            model_ctor,
            grid,
            X_tr,
            y_tr,
            X_val,
            y_val,
            class_pos_weight=pos_weight,
            shard=args.grid_shard,
        )
        if best is None:
            log.error(f"❌ No best result for {name}")
            continue

        best_model = best["ModelObj"].to(device).eval()
        log.info(
            f"✔ Best {name} params: {best['Params']} | val_f1_eff={best['val_f1']:.4f}"
        )

        # Evaluate on TEST
        prob_test = predict_proba(best_model, X_test_s)[:, 1]
        y_pred_test = (prob_test >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred_test)
        bal = balanced_accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Threshold finding on THRESH
        tf = ThresholdFinder(steps=600, min_predictions_ratio=0.90)
        prob_thresh = predict_proba(best_model, X_thresh_s)[:, 1]
        neg_thr, pos_thr, thr_acc, *_ = tf.find_best_thresholds(prob_thresh, y_thresh)

        # LIVE confident predictions
        prob_live = predict_proba(best_model, X_live_s)[:, 1]
        y_live_pred = np.full(len(prob_live), -1, dtype=int)
        y_live_pred[prob_live <= neg_thr] = 0
        y_live_pred[prob_live >= pos_thr] = 1
        mask = y_live_pred != -1
        if mask.any():
            acc_live = accuracy_score(y_live[mask], y_live_pred[mask])
            bal_live = balanced_accuracy_score(y_live[mask], y_live_pred[mask])
            f1_live = f1_score(y_live[mask], y_live_pred[mask])
        else:
            acc_live = bal_live = f1_live = 0.0

        # wins/loses روی live_confident (مثل CPU)
        wins = int(np.sum((y_live_pred == y_live) & mask))
        loses = int(np.sum((y_live_pred != y_live) & mask))

        log.info("-" * 75)
        log.info(f"MODEL: {name}")
        log.info(f"  TEST   → acc={acc:.4f} bal_acc={bal:.4f} f1={f1:.4f}")
        log.info(
            f"  THRESH → neg_thr={neg_thr:.3f} pos_thr={pos_thr:.3f} (bal_acc~={thr_acc:.4f})"
        )
        log.info(
            f"  LIVE   → acc={acc_live:.4f} bal_acc={bal_live:.4f} "
            f"f1={f1_live:.4f}  wins={wins} loses={loses}"
        )
        log.info("-" * 75)

        results.append(
            {
                "Model": name,
                "Best_Params": best["Params"],
                "Bias_Penalty": best["bias_penalty"],
                "Test_Accuracy": acc,
                "Test_BalAcc": bal,
                "Test_F1": f1,
                "NegThr": neg_thr,
                "PosThr": pos_thr,
                "Live_Accuracy": acc_live,
                "Live_BalAcc": bal_live,
                "Live_F1": f1_live,
                "Best_Model": best_model,
                "Prob_Live": prob_live,
            }
        )

    # 5) EXPORT stability / test report برای deep models (مثل CPU)
    stage("Exporting deep model stability report")
    deep_report_path = "deep_stability_report_gpu.csv"
    try:
        if results:
            rep = pd.DataFrame(results)[
                [
                    "Model",
                    "Test_Accuracy",
                    "Test_BalAcc",
                    "Test_F1",
                    "Live_Accuracy",
                    "Live_BalAcc",
                    "Live_F1",
                    "NegThr",
                    "PosThr",
                    "Bias_Penalty",
                ]
            ]
            rep["Perf_Drift(F1)"] = (rep["Test_F1"] - rep["Live_F1"]).abs()
            rep.to_csv(deep_report_path, index=False)
            log.info(f"💾 {deep_report_path} saved.")
        else:
            pd.DataFrame(
                columns=[
                    "Model",
                    "Test_Accuracy",
                    "Test_BalAcc",
                    "Test_F1",
                    "Live_Accuracy",
                    "Live_BalAcc",
                    "Live_F1",
                    "NegThr",
                    "PosThr",
                    "Bias_Penalty",
                    "Perf_Drift(F1)",
                ]
            ).to_csv(deep_report_path, index=False)
            log.warning(f"⚠️ No results; empty {deep_report_path} created.")
    except Exception as e:
        log.error(f"❌ Failed to write {deep_report_path}: {e}")

    # 6) ENSEMBLE VOTING over deep models (مثل اسکریپت CPU)
    stage("Deep-ensemble voting on LIVE (balanced ratio logic with safe divide)")

    N = len(X_live_s)
    if results:
        votes = []
        probas = []
        for r in results:
            proba = r["Prob_Live"]
            y_c = np.full(N, -1, dtype=int)
            y_c[proba <= r["NegThr"]] = 0
            y_c[proba >= r["PosThr"]] = 1
            votes.append(y_c)
            probas.append(proba)
        votes = np.array(votes) if votes else np.empty((0, N))
        probas = np.array(probas) if probas else np.empty((0, N))
    else:
        votes = np.empty((0, N))
        probas = np.empty((0, N))

    vote_sum = np.sum(votes == 1, axis=0) if votes.size else np.zeros(N)
    vote_conf = np.sum(votes != -1, axis=0) if votes.size else np.zeros(N)
    mean_conf = (
        np.nanmean(np.where(votes != -1, probas, np.nan), axis=0)
        if votes.size
        else np.zeros(N)
    )

    ens_df = pd.DataFrame(
        {
            "Index": np.arange(N),
            "y_true": y_live,
            "Votes_BUY": vote_sum,
            "Confident_Models": vote_conf,
            "Mean_Confidence": mean_conf,
        }
    )
    deep_ens_path = "deep_ensemble_predictions_gpu.csv"
    ens_df.to_csv(deep_ens_path, index=False)
    log.info(f"💾 {deep_ens_path} saved.")

    # Balanced ratio voting (نیاز به حداقل 3 مدل با رأی confident)
    signals = np.full(N, "NONE", dtype=object)
    safe_conf = np.where(vote_conf == 0, np.nan, vote_conf)
    vote_ratio = np.divide(vote_sum, safe_conf)  # BUY ratio among confident votes

    buy_condition = (vote_ratio >= 0.7) & (vote_conf >= 3)
    sell_condition = (vote_ratio <= 0.3) & (vote_conf >= 3)

    signals[buy_condition] = "BUY"
    signals[sell_condition] = "SELL"
    signals[~(buy_condition | sell_condition)] = "NONE"

    sig_df = pd.DataFrame(
        {
            "Index": np.arange(N),
            "Signal": signals,
            "Price": price_live,
            "Confidence": mean_conf,
            "Votes_BUY": vote_sum,
            "Confident_Models": vote_conf,
        }
    )
    deep_sig_path = "deep_signals_gpu.csv"
    sig_df.to_csv(deep_sig_path, index=False)
    log.info(f"💾 {deep_sig_path} saved with {len(sig_df)} rows.")

    # 7) COVERAGE + FINAL LIVE ACCURACY (همان لاجیک CPU برای ensemble)
    stage("Final LIVE accuracy & coverage for deep-ensemble")
    buy_n = int(np.sum(signals == "BUY"))
    sell_n = int(np.sum(signals == "SELL"))
    none_n = int(np.sum(signals == "NONE"))
    total = len(signals)
    coverage = ((buy_n + sell_n) / total) * 100 if total > 0 else 0.0

    log.info(f"✅ BUY={buy_n}, SELL={sell_n}, NONE={none_n}")
    log.info(f"📈 COVERAGE (Predictable Ratio): {coverage:.2f}% of live data covered")

    mask_live = signals != "NONE"
    if np.any(mask_live):
        y_pred_final = np.where(signals[mask_live] == "BUY", 1, 0)
        acc_live_final = accuracy_score(y_live[mask_live], y_pred_final)
        bal_acc_live_final = balanced_accuracy_score(
            y_live[mask_live], y_pred_final
        )
        f1_live_final = f1_score(y_live[mask_live], y_pred_final)
        log.info("\n📊 FINAL DEEP-ENSEMBLE LIVE REPORT:")
        log.info(f"  Accuracy:           {acc_live_final:.4f}")
        log.info(f"  Balanced Accuracy:  {bal_acc_live_final:.4f}")
        log.info(f"  F1 Score:           {f1_live_final:.4f}")
        log.info(
            f"  Coverage:           {coverage:.2f}%  "
            f"(BUY={buy_n}, SELL={sell_n}, NONE={none_n})"
        )
    else:
        log.warning("⚠️ No confident live predictions for accuracy computation.")

    log.info("\n✅ Deep GPU pipeline finished. All CSV reports saved.")

if __name__ == "__main__":
    main()
