from __future__ import annotations
import numpy as np
import pandas as pd

class CustomTechIndicators:
    """
    تولید مجموعه‌ای از اندیکاتورهای متداول بدون وابستگی به کتابخانهٔ ta.
    همهٔ خروجی‌ها shift(1) می‌شوند تا نگاه‌به‌آینده حذف شود.
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

    # ---------- توابع کمکی ----------
    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rma(s: pd.Series, n: int) -> pd.Series:
        alpha = 1 / n
        return s.ewm(alpha=alpha, adjust=False).mean()

    # ---------- محاسبهٔ اندیکاتورها ----------
    def _indicators(self) -> pd.DataFrame:
        df = self.df
        hi, lo, cl, op, vol = df[self.h], df[self.l], df[self.c], df[self.o], df[self.v]
        out = {}

        # 1) Parabolic SAR  -------------------------------
        sar = pd.Series(index=cl.index, dtype=float)
        rising, af, max_af = True, 0.018, 0.2
        ep = lo.iloc[0]; sar.iloc[0] = lo.iloc[0]
        for i in range(1, len(df)):
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

        # 2) Momentum-14 ---------------------------------
        out["momentum_14"] = cl.diff(14)

        # 3) TRIX-15 --------------------------------------
        ema1 = self._ema(cl, 15)
        ema2 = self._ema(ema1, 15)
        ema3 = self._ema(ema2, 15)
        out["trix_15"] = ema3.pct_change()

        # 4) Ultimate Oscillator --------------------------
        bp = cl - np.minimum(lo, cl.shift())
        tr = np.maximum.reduce([
            hi - lo,
            (hi - cl.shift()).abs(),
            (lo - cl.shift()).abs()
        ])
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        out["ultimate_osc"] = 100 * (4*avg7 + 2*avg14 + avg28) / 7

        # 5) Daily range ---------------------------------
        out["daily_range"] = hi - lo

        # 6) HV-20 ----------------------------------------
        log_ret = np.log(cl / cl.shift())
        out["hv_20"] = log_ret.rolling(20).std() * np.sqrt(24*365)

        # 7) Garman-Klass-20 ------------------------------
        hl = np.log(hi / lo)
        co = np.log(cl / op)
        gk = 0.5*hl**2 - (2*np.log(2)-1)*co**2
        out["garman_klass"] = gk.rolling(20).mean()

        # 8) Parkinson-20 --------------------------------
        ln_hl = np.log(hi / lo)
        out["parkinson_20"] = np.sqrt((ln_hl**2).rolling(20).sum() /
                                      (4*np.log(2)*20))

        # 9) Ulcer Index-14 -------------------------------
        dd = (cl / cl.cummax() - 1)*100
        out["ulcer_index_14"] = np.sqrt((dd**2).rolling(14).mean())

        # 10) Money Flow Index-14 -------------------------
        tp = (hi + lo + cl) / 3
        mf = tp * vol
        pos_mf = mf.where(tp > tp.shift(), 0)
        neg_mf = mf.where(tp < tp.shift(), 0)
        mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum()
        out["mfi"] = 100 - 100/(1 + mfr)

        # 11) Ease of Movement-14 -------------------------
        emv = ((hi + lo)/2).diff() * (hi - lo) / vol.replace(0, np.nan)
        out["eom_14"] = emv.rolling(14).mean()

        # 12) DPO-20 --------------------------------------
        out["dpo_20"] = cl.shift(int(20/2)+1) - cl.rolling(20).mean()

        # 13) MACD ----------------------------------------
        macd_line = self._ema(cl, 12) - self._ema(cl, 26)
        out["macd"] = macd_line

        # 14) RSI-14 --------------------------------------
        delta = cl.diff()
        gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
        rs = self._rma(gain, 14) / self._rma(loss, 14)
        out["rsi_14"] = 100 - 100/(1 + rs)

        # 15) Bollinger Bands & width ---------------------
        sma20 = cl.rolling(20).mean()
        std20 = cl.rolling(20).std()
        out["bollinger_high"] = sma20 + 2*std20
        out["bollinger_low"]  = sma20 - 2*std20
        bw = out["bollinger_high"] - out["bollinger_low"]
        out["bollinger_width"] = bw

        # 16) SMA & EMA pairs -----------------------------
        out["sma10"] = cl.rolling(10).mean()
        out["sma50"] = cl.rolling(50).mean()
        out["sma10_sma50_diff"] = out["sma10"] - out["sma50"]

        out["ema10"] = self._ema(cl, 10)
        out["ema50"] = self._ema(cl, 50)
        out["ema10_50_diff"] = out["ema10"] - out["ema50"]

        # 17) ATR-14 --------------------------------------
        tr = np.maximum.reduce([
            hi - lo,
            (hi - cl.shift()).abs(),
            (lo - cl.shift()).abs()
        ])
        out["atr_14"] = tr.rolling(14).mean()

        # 18) Bollinger width ratio -----------------------
        out["rolling_std_20"] = std20
        out["bollinger_width_ratio"] = bw / (std20 + 1e-9)

        # به DataFrame برگردانیم
        feats = pd.DataFrame(out)

        # ---------- شیفت برای حذف نشتی ----------
        feats = feats.shift(self.shift).add_prefix(self.prefix)
        return feats

    # ---------- رابط عمومی ----------
    def add_features(self, inplace: bool = True) -> pd.DataFrame:
        feats = self._indicators()
        if inplace:
            self.df = pd.concat([self.df, feats], axis=1)
            return self.df
        return feats
