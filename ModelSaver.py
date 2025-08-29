# ModelSaver.py
from __future__ import annotations

import os
import sys
import json
import joblib
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# تلاش برای گرفتن نسخه‌ی کتابخانه‌ها (اختیاری)
def _safe_version(mod_name: str) -> Optional[str]:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


class ModelSaver:
    """
    ذخیره و بازیابی کامل مدل + متادیتا برای اسکریپت‌های live.

    نکات:
    - سازگار با live_like_sim.py و کلیدهایی که آن اسکریپت انتظار دارد:
        pipeline, window_size, neg_thr, pos_thr,
        train_window_cols (یا feats), train_distribution
    - ذخیره‌سازی اتمیک (tmp → rename) برای جلوگیری از فایل ناقص در صورت قطع‌شدن.
    - درج متادیتا (نسخه‌ی کتابخانه‌ها، زمان ساخت، …) برای بازتولید‌پذیری.
    """

    def __init__(self, filename: str = "best_model.pkl", model_dir: Optional[str] = None) -> None:
        """
        filename: نام فایل خروجی (پیش‌فرض: best_model.pkl)
        model_dir: اگر تعیین شود، فایل در این پوشه نوشته می‌شود؛ در غیر این صورت cwd.
        """
        self.filename = filename
        self.model_dir = Path(model_dir) if model_dir else Path.cwd()

    # ------------------------ public API ------------------------

    def load_full(self, model_dir: Optional[str] = None):
        """
        مدل را از مسیر مشخص‌شده بارگذاری می‌کند.
        اگر model_dir ندهید، از self.model_dir استفاده می‌شود.
        """
        base = Path(model_dir) if model_dir else self.model_dir
        path = base / self.filename
        if not path.is_file():
            raise FileNotFoundError(f"[ModelSaver] file not found: {path}")
        return joblib.load(path)

    def save_full(
        self,
        *,
        pipeline,
        hyperparams: Dict[str, Any],
        window_size: int,
        neg_thr: float,
        pos_thr: float,
        feats: List[str],
        feat_mask: List[int],
        train_window_cols: List[str],
        scaler: Optional[Any] = None,
        train_distribution_path: str = "train_distribution.json",
    ) -> Path:
        """
        ذخیرهٔ همهٔ اجزای لازم در یک pickle.

        پارامترهای مهم برای سازگاری با شبیه‌ساز:
        - train_window_cols: ترتیب دقیق ستون‌ها بعد از WINDOW در TRAIN
        - train_distribution_path: نام/مسیر فایل توزیع آموزش (در شبیه‌ساز با model_dir جوین می‌شود)

        خروجی: مسیر فایل ذخیره‌شده (Path)
        """
        out_dir = self.model_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # برای سازگاری با live_like_sim: اگر مسیر train_distribution نسبی باشد،
        # شبیه‌ساز خودش با model_dir جوین می‌کند. اگر مطلق بود، فقط basename را ذخیره کنیم
        # تا در سیستم‌های دیگر هم با قرار دادن فایل در کنار مدل، قابل استفاده باشد.
        train_distribution_rel = os.path.basename(train_distribution_path or "train_distribution.json")

        # ساخت payload
        payload: Dict[str, Any] = {
            # === اجزای اصلی ===
            "pipeline": pipeline,                 # wrapper سازگار (predict/predict_proba)
            "scaler": scaler,                     # جهت سازگاری نسخه‌های قدیمی (ممکن است None بماند)
            "hyperparams": hyperparams,
            "window_size": int(window_size),
            "neg_thr": float(neg_thr),
            "pos_thr": float(pos_thr),

            # === اطلاعات فیچرها ===
            # توجه: live_like_sim در اولویت از train_window_cols استفاده می‌کند
            "feats": list(feats or []),
            "feat_mask": list(feat_mask or []),
            "train_window_cols": list(train_window_cols or []),

            # === مسیر توزیع آموزشی برای پرکردن ستون‌های مفقود ===
            "train_distribution": train_distribution_rel,

            # === متادیتا برای بازتولیدپذیری ===
            "schema_version": 2,
            "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "libs": {
                "numpy": _safe_version("numpy"),
                "pandas": _safe_version("pandas"),
                "scikit_learn": _safe_version("sklearn"),
                "imblearn": _safe_version("imblearn"),
                "joblib": _safe_version("joblib"),
            },
        }

        # اعتبارسنجی حداقلی کلیدهای ضروری
        self._validate_payload(payload)

        # مسیر خروجی
        out_path = out_dir / self.filename
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        # اتمیک‌نویسی: ابتدا tmp سپس rename
        joblib.dump(payload, tmp_path, compress=3, protocol=4)
        tmp_path.replace(out_path)

        # گزارش کوچک
        size_kb = out_path.stat().st_size / 1024.0
        digest = self._sha1_of(out_path)
        print(f"[ModelSaver] Saved → {out_path} ({size_kb:.1f} KB) sha1={digest}")

        # اختیاری: متادیتا را کنار فایل pkl هم ذخیره کن (برای خواندن سریع انسانی)
        try:
            meta_path = out_path.with_suffix(".meta.json")
            meta = {k: v for k, v in payload.items() if k not in ("pipeline", "scaler")}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # خطای ذخیرهٔ متادیتا مانع ذخیرهٔ مدل نشود
            pass

        return out_path

    # ------------------------ helpers ------------------------

    @staticmethod
    def _validate_payload(p: Dict[str, Any]) -> None:
        """بررسیِ حضور کلیدهای حیاتی که live_like_sim به آن‌ها نیاز دارد."""
        required = ["pipeline", "window_size", "neg_thr", "pos_thr", "train_distribution"]
        missing = [k for k in required if k not in p]
        if missing:
            raise ValueError(f"[ModelSaver] payload missing required keys: {missing}")

        # تایپ‌های رایج
        if not isinstance(p["window_size"], int):
            raise TypeError("[ModelSaver] window_size must be int")
        for k in ("neg_thr", "pos_thr"):
            if not isinstance(p[k], (float, int)):
                raise TypeError(f"[ModelSaver] {k} must be float")

        # لیست ستون‌ها
        for k in ("feats", "train_window_cols"):
            if k in p and p[k] is not None and not isinstance(p[k], list):
                raise TypeError(f"[ModelSaver] {k} must be a list")

    @staticmethod
    def _sha1_of(path: Path) -> str:
        """هشِ SHA1 فایل (فقط برای گزارش)."""
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:10]
