# leakfree_indicators.py
from __future__ import annotations
import numpy as np, pandas as pd

class LeakFreeBatchLive:
    """
    بازنویسی اندیکاتورهای مناقشه‌برانگیز (KCP, DCP, BBP, CCI, FI, OBV, ADI,
    Stoch-RSI, Pivot, Momentum, MA20, ...)  بدون کتابخانه ta و بدون نشتی.
    """

    def __init__(self, df: pd.DataFrame, prefix: str = "30T_", *, 
                 o="open", h="high", l="low", c="close", v="volume",
                 shift: int = 1):
        self.df = df
        self.o, self.h, self.l, self.c, self.v = o, h, l, c, v
        self.p = prefix
        self.shift = shift            # همیشه 1 برای حذف look-ahead

    # ---------- helpers ----------
    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _percent(ch_price: pd.Series, top: pd.Series, bot: pd.Series) -> pd.Series:
        return (ch_price - bot) / (top - bot + 1e-9)

    def build(self) -> pd.DataFrame:
        hi, lo, cl, op, vol = (self.df[x] for x in (self.h, self.l, self.c, self.o, self.v))
        out = {}

        # === 1) Keltner Channel Percent (KC-P) ============
        ema20   = self._ema(cl.shift(1), 20)              # ← exclude bar t
        atr20   = (hi.shift(1) - lo.shift(1)).rolling(20).mean()
        kc_up   = ema20 + 2 * atr20
        kc_low  = ema20 - 2 * atr20
        out["volatility_kcp"] = self._percent(cl.shift(1), kc_up, kc_low)   # :contentReference[oaicite:1]{index=1}

        # === 2) Donchian Channel Percent (DC-P) ============
        don_hi  = hi.shift(1).rolling(20).max()           # exclude t
        don_lo  = lo.shift(1).rolling(20).min()
        out["volatility_dcp"] = self._percent(cl.shift(1), don_hi, don_lo)  # :contentReference[oaicite:2]{index=2}

        # === 3) Bollinger Percent-B (BBP) =================
        sma20   = cl.shift(1).rolling(20).mean()
        std20   = cl.shift(1).rolling(20).std()
        bb_up, bb_lo = sma20 + 2*std20, sma20 - 2*std20
        out["volatility_bbp"] = self._percent(cl.shift(1), bb_up, bb_lo)    # :contentReference[oaicite:3]{index=3}

        # === 4) CCI-20 (Trend) ============================
        tp      = (hi.shift(1) + lo.shift(1) + cl.shift(1)) / 3
        sma_tp  = tp.rolling(20).mean()
        md      = (tp - sma_tp).abs().rolling(20).mean()
        out["trend_cci"] = (tp - sma_tp) / (0.015*md + 1e-9)                # :contentReference[oaicite:4]{index=4}

        # === 5) Force Index (FI) =========================
        out["volume_fi"] = (cl - cl.shift(1)) * vol.shift(1)                # :contentReference[oaicite:5]{index=5}

        # === 6) OBV & ADI =================================
        direction = np.sign(cl.diff()).replace(0, 1)
        out["volume_obv"] = (vol * direction).cumsum().shift(1)             # :contentReference[oaicite:6]{index=6}

        clv = ((cl - lo) - (hi - cl)) / (hi - lo).replace(0, np.nan)
        out["volume_adi"] = (clv * vol).cumsum().shift(1)                   # :contentReference[oaicite:7]{index=7}

        # === 7) Stoch-RSI-14 (Momentum) ===================
        delta = cl.diff()
        gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
        rs   = gain.rolling(14).mean() / loss.rolling(14).mean()
        rsi  = 100 - 100/(1 + rs)
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        out["momentum_stoch_rsi"] = stoch_rsi.shift(1)                      # :contentReference[oaicite:8]{index=8}

        # === 8) Pivot (prev-5)  ===========================
        pivot_raw = (hi + lo + cl)/3
        out["pivot"] = pivot_raw.shift(5)                                   # :contentReference[oaicite:9]{index=9}

        # === 9) Simple rolling stats (MA20, return diff, ROC, momentum-14) =
        out["ma20"]               = cl.shift(1).rolling(20).mean()
        out["return_difference"]  = cl.diff().shift(1)
        out["momentum_roc"]       = cl.shift(1).pct_change() * 100
        out["momentum_14"]        = cl.diff(14).shift(1)

        # ---------- بسته‌بندی ----------
        feats = pd.DataFrame(out).add_prefix(self.p)
        return feats
