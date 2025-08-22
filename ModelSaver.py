# ModelSaver.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib, os

class ModelSaver:
    """
    ذخیره و بازیابی کامل مدل + متادیتا برای اسکریپت‌های live.
    فقط یک «حقیقت منحصربه‌فرد» برای ساخت فایل best_model.pkl.
    """

    def __init__(self, filename: str = "best_model.pkl") -> None:
        self.filename = filename

    def load_full(self, model_dir: str):
        path = os.path.join(model_dir, self.filename)
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
    ) -> None:
        """ذخیرهٔ همهٔ اجزای لازم در یک pickle."""
        payload = {
            "pipeline": pipeline,
            "scaler": scaler,               # برای سازگاری قدیمی
            "hyperparams": hyperparams,
            "window_size": window_size,
            "neg_thr": neg_thr,
            "pos_thr": pos_thr,
            "feats": feats,
            "feat_mask": feat_mask,
            "train_window_cols": train_window_cols,
            "train_distribution": train_distribution_path,
        }
        joblib.dump(payload, self.filename, compress=3, protocol=4)
        size_kb = Path(self.filename).stat().st_size / 1024
        print(f"[ModelSaver] Saved → {self.filename} ({size_kb:.1f} KB)")
