import pandas as pd
import os
import time
import multiprocessing as mp
import logging

wins = 0
loses = 0
profit_pips = 0

# بارگذاری دیتاست‌های اصلی
df_H1 = pd.read_csv("XAUUSD_H1_live.csv")
df_M30 = pd.read_csv("XAUUSD_M30_live.csv")
df_M15 = pd.read_csv("XAUUSD_M15_live.csv")
df_M5 = pd.read_csv("XAUUSD_M5_live.csv")

# تبدیل به DataFrame (در صورت نیاز)
df_H1 = pd.DataFrame(df_H1)
df_M30 = pd.DataFrame(df_M30)
df_M15 = pd.DataFrame(df_M15)
df_M5 = pd.DataFrame(df_M5)

# مطمئن شوید که ستون time به datetime و مرتب‌شده است:
df_M5['time'] = pd.to_datetime(df_M5['time'])
df_M5.sort_values('time', inplace=True, ignore_index=True)

df_M15['time'] = pd.to_datetime(df_M15['time'])
df_M15.sort_values('time', inplace=True, ignore_index=True)

df_H1['time'] = pd.to_datetime(df_H1['time'])
df_H1.sort_values('time', inplace=True, ignore_index=True)

df_M30['time'] = pd.to_datetime(df_M30['time'])
df_M30.sort_values('time', inplace=True, ignore_index=True)

# تنظیمات لاگ
logging.basicConfig(
    filename='simulation.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def find_index_by_time(df, target_time, freq='M5'):
    """
    df : دیتافریم حاوی ستون time (datetime مرتب‌شده)
    target_time : زمان هدف (datetime)
    freq : یکی از ['M5', 'M15', 'H1']
    
    خروجی: ایندکس ردیف موردنظر یا -1 اگر پیدا نشد.
    """
    # اگر H1 است، دقیقه و ثانیه را صفر کنیم تا مشابه کد اصلی فقط year, month, day, hour مهم باشند
    if freq == 'H1':
        target_time = target_time.replace(minute=0, second=0, microsecond=0)
    
    # جستجوی دودویی با استفاده از searchsorted
    idx = df['time'].searchsorted(target_time, side='left')
    
    # اگر idx برابر طول دیتافریم شد یعنی target_time بزرگتر از آخرین زمان دیتافریم است
    if idx == len(df):
        return -1
    
    # حالا بررسی کنیم که آیا رکورد در آن ایندکس از نظر سال/ماه/روز/ساعت/(دقیقه) مطابق است
    row_time = df.iloc[idx]['time']
    
    if freq in ['M5', 'M15']:
        # سال، ماه، روز، ساعت و دقیقه
        if (row_time.year   == target_time.year  and
            row_time.month  == target_time.month and
            row_time.day    == target_time.day   and
            row_time.hour   == target_time.hour  and
            row_time.minute == target_time.minute):
            return idx
        else:
            return -1
    else:  # H1
        # سال، ماه، روز، ساعت
        if (row_time.year  == target_time.year  and
            row_time.month == target_time.month and
            row_time.day   == target_time.day   and
            row_time.hour  == target_time.hour):
            return idx
        else:
            return -1

def make_csv_for_time(df, target_time, freq, output_csv):
    """
    این تابع با گرفتن دیتافریم، زمان موردنظر و نوع تایم‌فریم (فرکانس)، 
    رکورد مربوط به آن زمان را پیدا کرده و تا آن رکورد را بریده (Slice) و فقط 4999 سطر آخر را ذخیره می‌کند.
    سپس در فایل CSV می‌نویسد.
    """
    idx = find_index_by_time(df, target_time, freq=freq)
    if idx == -1:
        # در این تابع اگر زمان پیدا نشود، فایلی ساخته نمی‌شود.
        print(f"Time {target_time} not found in {freq}. CSV not generated.")
        return
    
    df_part = df.iloc[:idx+1]
    
    # فقط 4999 سطر آخر
    if len(df_part) > 4999:
        df_part = df_part.iloc[-4999:]
    
    df_part.to_csv(output_csv, index=False)
    print(f"{freq}: Made -> {output_csv}")

def create_all_timeframes(df_M5, df_M15, df_H1, target_time):
    """
    ابتدا بررسی می‌کند زمان target_time در هر سه دیتافریم وجود دارد یا نه.
    - اگر حتی در یکی نباشد، تولید فایل انجام نمی‌شود (برمی‌گردد False).
    - اگر همه موجود باشد، با استفاده از مالتی‌پروسسینگ سه فایل CSV ساخته شده و True برمی‌گرداند.
    """
    # ابتدا اندیس هر دیتافریم را بررسی می‌کنیم
    idx_m5  = find_index_by_time(df_M5,  target_time, freq='M5')
    idx_m15 = find_index_by_time(df_M15, target_time, freq='M15')
    idx_h1  = find_index_by_time(df_H1,  target_time, freq='H1')

    # اگر یکی از آن‌ها -1 بود، یعنی پیدا نشده
    if idx_m5 == -1 or idx_m15 == -1 or idx_h1 == -1:
        print("At least one timeframe missing the target time. Skipping CSV generation.")
        return False
    
    # اگر همه اندیس‌ها معتبرند، در یک استخر پردازشی موازی فایل‌ها را ایجاد می‌کنیم
    with mp.Pool(processes=3) as pool:
        results = []
        results.append(pool.apply_async(
            make_csv_for_time, 
            args=(df_M5,  target_time, 'M5',  "XAUUSD.F_M5_live.csv")
        ))
        results.append(pool.apply_async(
            make_csv_for_time, 
            args=(df_M15, target_time, 'M15', "XAUUSD.F_M15_live.csv")
        ))
        results.append(pool.apply_async(
            make_csv_for_time, 
            args=(df_H1,  target_time, 'H1',  "XAUUSD.F_H1_live.csv")
        ))
        
        # منتظر بمانیم تا همهٔ فرآیندها تمام شوند
        for r in results:
            r.get()
    
    return True


# --- حلقه اصلی شبیه کد شما ---
for i in range(5000, 1, -1):
    # آماده‌سازی df_M30_temp برای 30 دقیقه جاری
    df_M30_temp = df_M30[:(len(df_M30)+1)-i].copy()
    df_M30_temp = df_M30_temp[-4999:]  # حداکثر 4999 سطر آخر
    df_M30_temp.to_csv("XAUUSD.F_M30_live.csv", index=False)

    temp = df_M30_temp.iloc[-1, :]  # سطر آخر M30_TEMP
    time_ = pd.to_datetime(temp["time"])
    close_M30 = temp["close"]

    # گرفتن close کندل بعدی (فقط برای محاسبه جهت)
    if (len(df_M30) - i) >= len(df_M30):
        # اگر به هر دلیلی ایندکس خارج محدوده شد، رد شویم (ولی نباید معمولاً رخ دهد)
        continue
    temp_time = df_M30.iloc[(len(df_M30)+1)-i,:]      # = k
    close_M30_next = temp_time["close"]

    oriantation = (close_M30 - close_M30_next)
    if oriantation >= 0:
        position = "SEL"
    else:
        position = "BUY"
    print(f"Next candel {position}")
    
    # اینجا تلاش می‌کنیم سه فایل (M5, M15, H1) را بر اساس time_ بسازیم.
    # تنها در صورت وجود زمانِ time_ در همه تایم‌فریم‌ها فایل‌ها ساخته می‌شوند.
    three_files_created = create_all_timeframes(df_M5, df_M15, df_H1, time_)

    # اگر فایل‌ها ساخته نشدند، یعنی در یکی از تایم‌فریم‌ها زمان پیدا نشد؛ برو به تکرار بعدی (نیم ساعت بعد)
    if not three_files_created:
        print("Skipping this half-hour because one or more timeframes did not have the specified time.")
        continue

    # حالا که سه فایل ساخته شده‌اند، منتظر پاسخ AI می‌شویم.
    print("Waiting for AI answer ...")
    flag = False

    while not os.path.isfile("Answer.txt"):
        time.sleep(1)
        if os.path.isfile("Answer.txt"):
            flag = True

    print("answer found")
    if flag:
        with open("Answer.txt", "r") as file:
            first_three_chars = file.read(3)
            os.remove("Answer.txt")
            
            # بررسی نتیجه و محاسبه برد/باخت
            if first_three_chars == "BUY" and position == "BUY":
                wins += 1
                profit_pips += abs(oriantation)
            elif first_three_chars == "SEL" and position == "SEL":
                wins += 1
                profit_pips += abs(oriantation)
            elif (first_three_chars == "BUY" and position == "SEL") or (first_three_chars == "SEL" and position == "BUY"):
                loses += 1
                profit_pips -= abs(oriantation)
            elif first_three_chars == "NAN":
                if wins > 0 and loses > 0:
                    logging.info(f"AI answer is NAN , Wins = {wins} --- Loses = {loses} --- ACC = {wins*100/(wins+loses)} --- Profit = {profit_pips} pips")
                    print(f"AI answer is NAN , Wins = {wins} --- Loses = {loses} --- ACC = {wins*100/(wins+loses)} --- Profit = {profit_pips} pips")
                else:
                    logging.info(f"AI answer is NAN , Wins = {wins} --- Loses = {loses} --- ACC = 0 --- Profit = {profit_pips} pips")
                    print(f"AI answer is NAN , Wins = {wins} --- Loses = {loses} --- ACC = 0 --- Profit = {profit_pips} pips")

        # گزارش وضعیت برد/باخت
        if first_three_chars != "NAN":
            if wins > 0:
                logging.info(f"Wins = {wins} --- Loses = {loses} --- ACC = {wins*100/(wins+loses)} --- Profit = {profit_pips} pips")
                print(f"Wins = {wins} --- Loses = {loses} --- ACC = {wins*100/(wins+loses)} --- Profit = {profit_pips} pips")
            else:
                logging.info(f"Wins = {wins} --- Loses = {loses} --- ACC = 0 --- Profit = {profit_pips} pips")
                print(f"Wins = {wins} --- Loses = {loses} --- ACC = 0 --- Profit = {profit_pips} pips")
