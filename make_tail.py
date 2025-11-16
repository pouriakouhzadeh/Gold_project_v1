import pandas as pd

files_config = {
    "XAUUSD_M5.csv": 10000,
    "XAUUSD_M15.csv": 1000,
    "XAUUSD_M30.csv": 1000,
    "XAUUSD_H1.csv": 10000
}

for file_name, n_rows in files_config.items():
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"فایل {file_name} پیدا نشد.")
        continue

    # گرفتن n_rows ردیف آخر
    df_tail = df.tail(n_rows)

    # ساختن نام جدید
    new_file_name = file_name.replace(".csv", "_tail.csv")

    # ذخیره فایل
    df_tail.to_csv(new_file_name, index=False)

    print(f"{n_rows} ردیف آخر از {file_name} در فایل {new_file_name} ذخیره شد.")
