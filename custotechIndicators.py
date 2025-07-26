from __future__ import annotations
import numpy as np
import pandas as pd

# ───── helper: force numeric ─────
def _num(s: pd.Series) -> pd.Series:
    """Ensure numeric dtype; non‑convertible → NaN."""
    return pd.to_numeric(s, errors="coerce")

class CustomTechIndicators:
    """
    اندیکاتورهای متداول (بدون ta‑lib) با حذف کامل نگاه‌به‌آینده.
    تمام خروجی‌ها shift(1) می‌شوند.
    """
    # ---------- سازنده ----------
    def __init__(
        self,
        df: pd.DataFrame,
        o: str = "open", h: str = "high", l: str = "low",
        c: str = "close", v: str = "volume",
        prefix: str = "ti_", shift: int = 1
    ):
        self.df = df.copy()
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.prefix, self.shift = prefix, shift

    # ---------- EMA / RMA ----------
    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rma(s: pd.Series, n: int) -> pd.Series:
        alpha = 1 / n
        return s.ewm(alpha=alpha, adjust=False).mean()

    # ---------- اندیکاتورها ----------
    def _indicators(self) -> pd.DataFrame:
        hi, lo, cl, op, vol = (
            _num(self.df[self.h]), _num(self.df[self.l]),
            _num(self.df[self.c]), _num(self.df[self.o]),
            _num(self.df[self.v])
        )

        out: dict[str, pd.Series] = {}

        # 1) Parabolic SAR
        sar = pd.Series(index=cl.index, dtype=float)
        rising, af, max_af = True, 0.018, 0.2
        ep = lo.iloc[0]; sar.iloc[0] = lo.iloc[0]
        for i in range(1, len(cl)):
            prev = sar.iloc[i-1]
            if rising:
                sar.iloc[i] = prev + af * (ep - prev)
                if lo.iloc[i] < sar.iloc[i]:
                    rising, sar.iloc[i], ep, af = False, ep, lo.iloc[i], 0.018
                elif hi.iloc[i] > ep:
                    ep, af = hi.iloc[i], min(af + 0.018, max_af)
            else:
                sar.iloc[i] = prev + af * (ep - prev)
                if hi.iloc[i] > sar.iloc[i]:
                    rising, sar.iloc[i], ep, af = True, ep, hi.iloc[i], 0.018
                elif lo.iloc[i] < ep:
                    ep, af = lo.iloc[i], min(af + 0.018, max_af)
        out["parabolic_sar"] = sar

        # 2) Momentum‑14
        out["momentum_14"] = cl.diff(14)

        # 3) TRIX‑15
        ema3 = self._ema(self._ema(self._ema(cl, 15), 15), 15)
        out["trix_15"] = ema3.pct_change()

        # 4) Ultimate Oscillator
        bp = cl - np.minimum(lo, cl.shift())
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift()).abs(),
            (lo - cl.shift()).abs()
        ], axis=1).max(axis=1)
        out["ultimate_osc"] = _num(
            100 * (
                4 * bp.rolling(7).sum() / tr.rolling(7).sum() +
                2 * bp.rolling(14).sum() / tr.rolling(14).sum() +
                    bp.rolling(28).sum() / tr.rolling(28).sum()
            ) / 7
        )

        # 5) Daily Range
        out["daily_range"] = hi - lo

        # 6) HV‑20
        out["hv_20"] = _num(np.log(cl / cl.shift()).rolling(20).std() * np.sqrt(24*365))

        # 7) Garman‑Klass‑20
        hl = np.log(hi / lo); co = np.log(cl / op)
        gk = 0.5*hl**2 - (2*np.log(2)-1)*co**2
        out["garman_klass"] = gk.rolling(20).mean()

        # 8) Parkinson‑20
        ln_hl = np.log(hi / lo)
        out["parkinson_20"] = np.sqrt((ln_hl**2).rolling(20).sum() / (4*np.log(2)*20))

        # 9) Ulcer Index‑14
        dd = (cl / cl.cummax() - 1)*100
        out["ulcer_index_14"] = np.sqrt((dd**2).rolling(14).mean())

        # 10) MFI‑14
        tp = (hi + lo + cl) / 3
        mf = tp * vol
        mfr = mf.where(tp > tp.shift(), 0).rolling(14).sum() / mf.where(tp < tp.shift(), 0).rolling(14).sum()
        out["mfi"] = 100 - 100/(1 + mfr)

        # 11) EOM‑14
        emv = ((hi + lo)/2).diff() * (hi - lo) / vol.replace(0, np.nan)
        out["eom_14"] = emv.rolling(14).mean()

        # 12) DPO‑20
        out["dpo_20"] = cl.shift(11) - cl.rolling(20).mean()

        # 13) MACD line
        out["macd"] = self._ema(cl, 12) - self._ema(cl, 26)

        # 14) RSI‑14
        delta = cl.diff()
        rs = self._rma(delta.clip(lower=0), 14) / self._rma((-delta).clip(lower=0), 14)
        out["rsi_14"] = 100 - 100/(1 + rs)

        # 15) Bollinger bands & width
        sma20, std20 = cl.rolling(20).mean(), cl.rolling(20).std()
        out["bollinger_high"]  = sma20 + 2*std20
        out["bollinger_low"]   = sma20 - 2*std20
        bw = out["bollinger_high"] - out["bollinger_low"]
        out["bollinger_width"] = bw

        # 16) SMA/EMA pairs
        out["sma10"] = cl.rolling(10).mean()
        out["sma50"] = cl.rolling(50).mean()
        out["sma10_sma50_diff"] = out["sma10"] - out["sma50"]
        out["ema10"] = self._ema(cl, 10)
        out["ema50"] = self._ema(cl, 50)
        out["ema10_50_diff"] = out["ema10"] - out["ema50"]

        # 17) ATR‑14
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift()).abs(),
            (lo - cl.shift()).abs()
        ], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14).mean()

        # 18) Bollinger width ratio
        out["rolling_std_20"] = std20
        out["bollinger_width_ratio"] = bw / (std20 + 1e-9)

        # ---------- اجبار نهایی به numeric ----------
        for k in out:
            out[k] = pd.to_numeric(out[k], errors="coerce")

        feats = pd.DataFrame(out).shift(self.shift).add_prefix(self.prefix)
        return feats

    # ---------- رابط عمومی ----------
    def add_features(self, inplace: bool = True) -> pd.DataFrame:
        feats = self._indicators()
        if inplace:
            self.df = pd.concat([self.df, feats], axis=1)
            return self.df
        return feats
