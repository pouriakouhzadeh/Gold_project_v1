import pandas as pd
import numpy as np

class ClearData:
    """
    کلاسی برای تمیز کردن داده‌های مربوط به جفت ارز XAUUSD.
    مقادیر صفر، تهی (NaN)، یا خارج از محدوده نرمال را با میانگین سطر قبل و بعد جایگزین می‌کند.
    """

    def __init__(self):
        pass

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        دیتافریمی شامل ستون‌های time, open, high, low, close, volume را گرفته
        و آن را تمیز می‌کند. ستون time تغییری نخواهد کرد مگر آنکه صفر یا NaN باشد
        (که در اینجا فیلتر نشده است). اگر تمایل دارید ستون زمان نیز
        پاکسازی شود، می‌توانید آن را به لیست ستون‌های عددی اضافه کنید یا منطق جداگانه‌ای پیاده‌سازی نمایید.

        پارامترها:
        ----------
        df : pd.DataFrame
            دیتافریمی که نیاز به پاکسازی دارد.

        خروجی:
        -------
        df_cleaned : pd.DataFrame
            دیتافریم پاکسازی شده (و با ایندکس ریست شده).
        """

        # لیست ستون‌های عددی که می‌خواهیم روی آن‌ها عملیات پاکسازی انجام دهیم
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']

        # برای هر ستون عددی، ابتدا میانگین و انحراف معیار را محاسبه کرده و سپس مقادیر
        # صفر، نال، یا خارج از محدوده نرمال را جایگزین می‌کنیم.
        for col in numeric_cols:
            mean_val = df[col].mean(skipna=True)
            std_val = df[col].std(skipna=True)

            # محدوده نرمال را میانگین ± 3*std در نظر می‌گیریم
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val

            # در طول سطرهای دیتافریم حرکت می‌کنیم
            for i in range(len(df)):
                val = df.loc[i, col]

                # بررسی شرایطی که داده باید پاکسازی شود
                if pd.isna(val) or val == 0 or val < lower_bound or val > upper_bound:
                    # اگر سطر اول است، از سطر بعد برای جایگزین استفاده می‌کنیم
                    if i == 0:
                        if len(df) > 1:
                            df.loc[i, col] = df.loc[i+1, col]
                        else:
                            # اگر کل دیتافریم فقط یک سطر داشت!
                            df.loc[i, col] = mean_val
                    # اگر سطر آخر است، از سطر قبل برای جایگزین استفاده می‌کنیم
                    elif i == len(df) - 1:
                        df.loc[i, col] = df.loc[i-1, col]
                    else:
                        # در حالت عادی، میانگین سطر قبل و بعد را می‌گیریم
                        val_before = df.loc[i-1, col]
                        val_after = df.loc[i+1, col]
                        df.loc[i, col] = (val_before + val_after) / 2

        # در انتهای عملیات، ایندکس دیتافریم را ریست می‌کنیم تا ایندکس قدیمی حذف شود
        df.reset_index(drop=True, inplace=True)

        return df
