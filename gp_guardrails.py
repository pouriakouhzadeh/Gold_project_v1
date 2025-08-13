# --- Guardrails for Live/Sim consistency ---
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class LiveGuardError(RuntimeError): ...
class WarmupNotEnough(LiveGuardError): ...
class ColumnsMismatch(LiveGuardError): ...
class BadValuesFound(LiveGuardError): ...
class TimestampNotAligned(LiveGuardError): ...

def assert_no_nan_inf(df: pd.DataFrame, where: str = "X") -> None:
    if not np.isfinite(df.values).all():
        bad_locs = np.argwhere(~np.isfinite(df.values))
        msg = f"[{where}] Found NaN/Inf at positions like {bad_locs[:3].tolist()} ..."
        raise BadValuesFound(msg)

def assert_columns_match_and_order(X: pd.DataFrame, train_window_cols: List[str]) -> pd.DataFrame:
    # همان ترتیب آموزش را اعمال می‌کنیم و همزمان بررسی برابری مجموعه ستون‌ها
    x_cols = list(X.columns)
    if set(x_cols) != set(train_window_cols):
        extra = sorted(set(x_cols) - set(train_window_cols))
        missing = sorted(set(train_window_cols) - set(x_cols))
        raise ColumnsMismatch(f"[cols] mismatch: extra={extra[:5]}, missing={missing[:5]}")
    X = X.reindex(columns=train_window_cols)
    return X

def assert_dtype_float32(X: pd.DataFrame) -> pd.DataFrame:
    # تبدیل امن به float32
    if any(dt.kind not in "fiu" for dt in X.dtypes):
        X = X.astype("float32")
    else:
        X = X.astype("float32")
    return X

def assert_warmup_covered(history_lengths: Dict[str, int], min_required: Dict[str, int]) -> None:
    """history_lengths: {'5T': N5, '15T': N15, ...}
       min_required:    {'5T': need5, ...}  -> بر اساس بزرگ‌ترین lookback اندیکاتورها تعیین کن
    """
    lacks = {tf: (history_lengths.get(tf, 0), need) for tf, need in min_required.items() if history_lengths.get(tf, 0) < need}
    if lacks:
        preview = {k: f"have={v[0]}, need={v[1]}" for k, v in lacks.items()}
        raise WarmupNotEnough(f"[warmup] not enough history: {preview}")

def latest_common_timestamp(ctx: Dict[str, pd.DataFrame]) -> pd.Timestamp:
    """ آخرین تایم‌استمپ مشترک بین تمام تایم‌فریم‌ها که در همه وجود دارد و کندل بسته شده است. """
    # فرض: هر df ستونی 'time' دارد
    sets = []
    for tf, df in ctx.items():
        if "time" not in df.columns:
            raise TimestampNotAligned(f"[{tf}] missing 'time' column")
        ts = pd.to_datetime(df["time"])
        sets.append(pd.Index(ts))
    inter = sets[0]
    for idx in sets[1:]:
        inter = inter.intersection(idx)
    if len(inter) == 0:
        raise TimestampNotAligned("[time] no common timestamps across timeframes")
    return inter.max()

def take_last_closed_rows(ctx: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """ هر تایم‌فریم را تا آخرین تایم‌استمپ مشترک trim می‌کند """
    ts_common = latest_common_timestamp(ctx)
    out = {}
    for tf, df in ctx.items():
        df2 = df.copy()
        df2["time"] = pd.to_datetime(df2["time"])
        out[tf] = df2[df2["time"] <= ts_common].reset_index(drop=True)
    return out

def guard_and_prepare_for_predict(
    X: pd.DataFrame,
    train_window_cols: List[str],
    min_required_history: Dict[str, int],
    ctx_history_lengths: Dict[str, int],
    where: str = "X"
) -> pd.DataFrame:
    # warmup
    assert_warmup_covered(ctx_history_lengths, min_required_history)
    # NaN/Inf
    assert_no_nan_inf(X, where=where)
    # columns & order
    X = assert_columns_match_and_order(X, train_window_cols)
    # dtype
    X = assert_dtype_float32(X)
    return X
