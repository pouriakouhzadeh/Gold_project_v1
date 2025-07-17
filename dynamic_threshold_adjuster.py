# dynamic_threshold_adjuster.py

class DynamicThresholdAdjuster:
    """
    یک تنظیم‌کنندهٔ داینامیک آستانه، با سه حالت:
      1) HIGH: وقتی ATR از حدی بالاتر برود => آستانه‌ها را دور می‌کنیم (ناحیهٔ نامطمئن بزرگ می‌شود)
      2) LOW: وقتی Volume از حدی پایین‌تر باشد => آستانه‌ها را نزدیک می‌کنیم (ناحیهٔ نامطمئن کوچک می‌شود)
      3) NORMAL: در غیر این دو حالت => از آستانهٔ اصلی (بدون تغییر) استفاده می‌کنیم

    منطق ساده (الویت):
      اگر ATR > atr_high  => state = HIGH
      else اگر Volume < vol_low => state = LOW
      در غیر این صورت => state = NORMAL
    """

    def __init__(self,
                 atr_high=2.0,
                 vol_low=500,
                 shift=0.03):
        """
        Parameters
        ----------
        atr_high : float
            اگر ATR آخرین کندل از این مقدار بزرگ‌تر باشد => می‌رویم به حالت HIGH.
        vol_low : float
            اگر حجم آخرین کندل از این مقدار کوچک‌تر باشد => می‌رویم به حالت LOW.
        shift : float
            میزانی که برای جابه‌جاکردن آستانه‌ها (دور یا نزدیک) استفاده می‌کنیم.
        """
        self.atr_high = atr_high
        self.vol_low = vol_low
        self.shift = shift

    def adjust(self, neg_thr, pos_thr, last_atr, last_volume):
        """
        براساس ATR و Volume آخرین کندل، حالت را تشخیص داده و آستانه‌های جدید را برمی‌گرداند.

        Parameters
        ----------
        neg_thr : float
            آستانه منفی پیش‌فرض (خروجی ThresholdFinder).
        pos_thr : float
            آستانه مثبت پیش‌فرض (خروجی ThresholdFinder).
        last_atr : float
            مقدار ATR آخرین کندل.
        last_volume : float
            مقدار Volume آخرین کندل.

        Returns
        -------
        new_neg : float
            آستانه منفی داینامیک
        new_pos : float
            آستانه مثبت داینامیک
        """

        # تعیین حالت
        # اولویت: اگر ATR خیلی بالا باشد => HIGH
        if last_atr > self.atr_high:
            state = "HIGH"
        # اگر در حالت بالا نبود، و Volume خیلی پایین است => LOW
        elif last_volume < self.vol_low:
            state = "LOW"
        else:
            state = "NORMAL"

        # براساس state مقدار new_neg و new_pos را تعیین می‌کنیم
        new_neg = neg_thr
        new_pos = pos_thr

        if state == "HIGH":
            # ناحیه عدم قطعیت بزرگ شود
            new_neg = max(neg_thr - self.shift, 0.0)
            new_pos = min(pos_thr + self.shift, 1.0)

        elif state == "LOW":
            # ناحیه عدم قطعیت کوچک شود
            new_neg = min(neg_thr + self.shift, 1.0)
            new_pos = max(pos_thr - self.shift, 0.0)

        # در پایان مطمئن می‌شویم new_neg < new_pos بماند
        if new_neg >= new_pos:
            midpoint = (new_neg + new_pos) / 2
            new_neg = max(midpoint - 1e-3, 0.0)
            new_pos = min(midpoint + 1e-3, 1.0)

        return new_neg, new_pos
