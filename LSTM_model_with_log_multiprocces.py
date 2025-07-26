############################################################
# main_lstm_project_with_logging.py
# پروژه کامل با مدل LSTM و بهینه‌سازی هایپرپارامترها با الگوریتم ژنتیک
# شامل:
# 1. DataCleaner (پاکسازی داده)
# 2. PrepareDataForTrain (تولید ویژگی و آماده‌سازی)
# 3. ThresholdFinder (جستجوی آستانه منفی و مثبت)
# 4. مدل LSTM (با Keras)
# 5. الگوریتم ژنتیک (DEAP) برای بهینه‌سازی هایپرپارامترهای LSTM
# 6. تقسیم داده به 4 بخش (train, threshold, test, final)
# 7. ارزیابی نهایی مدل
# 8. ثبت لاگ‌ها با کتابخانه logging
############################################################

import pandas as pd
import numpy as np
import random
import logging
import copy
import multiprocessing
import warnings
from collections import Counter
import os

warnings.filterwarnings("ignore")

# =====================
# کتابخانه‌های اسکیکیت
# =====================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# =====================
# کتابخانه‌های Keras
# =====================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K

# =====================
# کتابخانه الگوریتم ژنتیک DEAP
# =====================
from deap import base, creator, tools

##################################################################
# 1) کلاس DataCleaner برای پاک‌سازی داده (مانند پروژه قبلی)
##################################################################
class DataCleaner:
    """
    این کلاس وظیفه دارد داده را از فایل CSV بخواند، نویزها و مقادیر پرت را حذف کند،
    و در صورت وجود دادهٔ گمشده (NaN)، آن‌ها را برطرف نماید.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Reads data from a CSV file and parses the 'time' column as DateTime."""
        self.data = pd.read_csv(self.file_path, parse_dates=['time'])

    def remove_noise(self):
        """Replaces inf and -inf with NaN, and drops duplicates."""
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.drop_duplicates(inplace=True)

    def fix_missing_values(self):
        """
        Fills missing values in numerical columns with mean, 
        and in categorical/object columns with mode.
        """
        for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def clean_data(self):
        """Executes the entire cleaning process and returns the cleaned DataFrame."""
        self.load_data()
        self.remove_noise()
        self.fix_missing_values()
        return self.data


##################################################################
# 2) کلاس ThresholdFinder برای جستجوی آستانه‌های منفی و مثبت
##################################################################
class ThresholdFinder:
    """
    این کلاس وظیفه دارد با آزمون آستانه‌های مختلف (neg_threshold, pos_threshold)،
    بهترین دقت (Accuracy) را بیابد. در اینجا از روش ساده brute-force استفاده شده است.
    """
    def __init__(self, steps=3000, min_predictions_ratio=2/3):
        """
        steps: تعداد گام برای جستجوی آستانه (مثلاً 3000 یعنی قطع‌بندی 0 تا 1 به 3000 قسمت).
        min_predictions_ratio: حداقل درصد از داده که باید تصمیم‌گیری صریح (0 یا 1) شود.
        """
        self.steps = steps
        self.min_predictions_ratio = min_predictions_ratio

    def _calculate_acc(self, thresholds, proba, y_true):
        th_neg, th_pos = thresholds
        if th_neg > th_pos:
            return 0, th_neg, th_pos, 0, 0

        pos_indices = proba >= th_pos
        neg_indices = proba <= th_neg

        wins = np.sum((pos_indices & (y_true == 1)) | (neg_indices & (y_true == 0)))
        loses = np.sum((pos_indices & (y_true == 0)) | (neg_indices & (y_true == 1)))

        total_predicted = wins + loses
        if total_predicted < round(len(y_true) * self.min_predictions_ratio):
            return 0, th_neg, th_pos, wins, loses

        acc = wins / (wins + loses) if (wins + loses) else 0
        return acc, th_neg, th_pos, wins, loses

    def find_best_thresholds(self, proba, y_true):
        """
        به صورت brute-force در فضای [0,1]x[0,1] جستجو می‌کند و بهترین آستانه‌ها را برمی‌گرداند.
        """
        proba = np.array(proba)
        y_true = np.array(y_true)
        threshold_pairs = [
            (k/self.steps, l/self.steps)
            for k in range(self.steps+1)
            for l in range(self.steps+1)
        ]

        best_acc = -1
        best_neg, best_pos = 0, 1
        best_wins, best_loses = 0, 0

        for (th_neg, th_pos) in threshold_pairs:
            acc, _, _, w, l = self._calculate_acc((th_neg, th_pos), proba, y_true)
            if acc > best_acc:
                best_acc = acc
                best_neg, best_pos = th_neg, th_pos
                best_wins, best_loses = w, l

        return best_neg, best_pos, best_acc, best_wins, best_loses


##################################################################
# 3) کلاس PrepareDataForTrain برای تولید ویژگی و آماده‌سازی داده
##################################################################
class PrepareDataForTrain:
    """
    این کلاس مسئول آماده‌سازی داده به شکل دلخواه ماست.
    شامل:
      - ایجاد ستون‌های ساعتی
      - چند محاسبه ساده (MA, ROC, ...)
      - تعریف متغیر هدف
      - حذف مقادیر غیرضروری
      - انجام differencing
      - اعمال روش‌های انتخاب ویژگی (VarianceThreshold, Correlation, MI)
      - windowing داده
    در عمل، می‌توانید اندیکاتورهای بیشتری را اضافه کنید.
    """
    def __init__(self):
        pass

    def ready(self, data: pd.DataFrame, window: int = 1, top_k_features=300):
        """
        data: دادهٔ اولیه
        window: اندازه پنجره (مثلاً 10 یعنی هر نمونه شامل 10 کندل قبلی)
        top_k_features: تعداد فیچرهای برتر که با MI انتخاب می‌شوند
        """
        df = data.copy()

        # اضافه کردن ستون‌های زمانی
        if 'time' in df.columns:
            df['Hour'] = df['time'].dt.hour
            df['DayOfWeek'] = df['time'].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
            df.drop(columns=['time'], inplace=True)

        # چند محاسبه ساده
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ReturnDifference'] = df['close'].diff()
        df['ROC'] = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-9) * 100

        df.dropna(axis=0, how='any', inplace=True)

        # هدف (Target) باینری بر اساس صعود یا نزول در کندل بعدی
        target = ((df['close'].shift(-1) - df['close']) > 0).astype(int)
        target = target[:-1]
        df = df[:-1]

        # differencing از همه ستون‌ها به جز Hour
        hour = df['Hour']
        df.drop(columns=['Hour'], inplace=True)

        df_diff = df.diff().dropna().copy()
        target = target.loc[df_diff.index].copy()
        df_diff['Hour'] = hour.loc[df_diff.index]

        df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_diff.dropna(axis=0, how='any', inplace=True)
        target = target.loc[df_diff.index].copy()
        df_diff.reset_index(drop=True, inplace=True)
        target.reset_index(drop=True, inplace=True)

        # حذف فیچرهای کم‌تنوع با VarianceThreshold
        selector_var = VarianceThreshold(threshold=0.01)
        selector_var.fit(df_diff)
        df_diff = df_diff[df_diff.columns[selector_var.get_support()]]

        # حذف فیچرهای با همبستگی بالا (رابطه بالای 0.9)
        corr_matrix = df_diff.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        df_diff.drop(columns=to_drop, inplace=True, errors='ignore')

        # انتخاب ویژگی بر اساس Mutual Information
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df_diff)
        mi_values = mutual_info_classif(X_scaled, target, discrete_features='auto')
        mi_series = pd.Series(mi_values, index=df_diff.columns).sort_values(ascending=False)
        top_k_features = min(top_k_features, len(mi_series))
        top_features = mi_series.head(top_k_features).index
        df_diff = df_diff[top_features].copy()

        # اگر window=1 باشد، بدون پنجره‌بندی برمی‌گردیم
        if window < 1:
            window = 1
        if window == 1:
            return df_diff, target

        # در غیر اینصورت، اعمال windowing
        arr_list = []
        new_index = []
        for i in range(window - 1, len(df_diff)):
            row_feat = []
            for offset in range(window):
                idx = i - offset
                row_feat.extend(df_diff.iloc[idx].values)
            arr_list.append(row_feat)
            new_index.append(i)

        X_windowed = pd.DataFrame(arr_list, index=new_index)
        y_windowed = target.loc[X_windowed.index].copy()

        X_windowed.reset_index(drop=True, inplace=True)
        y_windowed.reset_index(drop=True, inplace=True)

        X_windowed.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_windowed.dropna(axis=0, how='any', inplace=True)

        y_windowed = y_windowed.loc[X_windowed.index]
        y_windowed.reset_index(drop=True, inplace=True)

        return X_windowed, y_windowed


##################################################################
# تابع کمکی برای SMOTE
##################################################################
def get_smote(n_samples, random_state=42, k_neighbors=3):
    """
    ساخت شیء SMOTE با تعداد همسایه مشخص (k_neighbors).
    اگر کلاس مثبت کمتر از k_neighbors باشد، ممکن است SMOTE خطا دهد.
    """
    if n_samples < k_neighbors:
        return None
    return SMOTE(random_state=random_state, k_neighbors=k_neighbors)


##################################################################
# 4) کلاس HoursGene (اختیاری): تنظیم ماسک ساعت
##################################################################
class HoursGene:
    """
    برای فیلترکردن ساعت‌ها (در صورت تمایل) استفاده می‌شود.
    در اینجا بیشتر جنبه نمایشی دارد. 
    اگر بخواهید واقعاً ماسک ساعت اعمال کنید، باید در evaluate_individual 
    داده را بر اساس این ساعات فیلتر نمایید.
    """
    def __init__(self, valid_hours, excluded_hours=[0,1,2,3]):
        self.valid_hours = [h for h in valid_hours if h not in excluded_hours]

    def init_hours_subset(self, num_selected):
        n = len(self.valid_hours)
        if num_selected > n:
            num_selected = n
        arr = [1]*num_selected + [0]*(n - num_selected)
        random.shuffle(arr)
        return arr


##################################################################
# تابع ساخت مدل LSTM با Keras
##################################################################
def build_lstm_model(
    input_shape,
    lstm_units=64,
    dropout_rate=0.2,
    num_layers=1,
    optimizer='adam'
):
    """
    ساخت یک مدل LSTM برای طبقه‌بندی دودویی.
    پارامترها:
      input_shape   : (window_size, num_features) 
      lstm_units    : تعداد نورون‌های هر لایه LSTM
      dropout_rate  : نرخ دراپ‌آوت
      num_layers    : تعداد لایه‌های LSTM
      optimizer     : نوع اپتیمایزر (مثلاً 'adam', 'sgd', 'rmsprop', ...)
    
    خروجی:
      model         : مدل Sequential کامپایل‌شده
    """
    model = Sequential()
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        if i == 0:
            # لایهٔ اول، نیاز به input_shape دارد
            model.add(
                LSTM(
                    lstm_units,
                    return_sequences=return_sequences,
                    input_shape=input_shape
                )
            )
        else:
            # لایه‌های میانی
            model.add(LSTM(lstm_units, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

    # در انتها یک Dense با خروجی یک نرون (برای طبقه‌بندی دودویی)
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


##################################################################
# 5) کلاس اصلی GeneticAlgorithmRunner
##################################################################
class GeneticAlgorithmRunner:
    """
    در این کلاس:
      - داده را لود و پاکسازی می‌کنیم
      - الگوریتم ژنتیک را برای تنظیم هایپرپارامترهای LSTM اجرا می‌کنیم
      - بهترین فرد را انتخاب می‌کنیم
      - آن را روی 4 بخش داده (train, threshold, test, final) ارزیابی می‌کنیم
    """

    def __init__(self):
        """
        مقادیر پیش‌فرض برای الگوریتم ژنتیک:
          POPULATION_SIZE, N_GENERATIONS, CX_PB, MUT_PB, EARLY_STOPPING_THRESHOLD
        همچنین پارامترهای مختلف را در همین init می‌توانیم تنظیم کنیم.
        """
        self.POPULATION_SIZE = 20       # تعداد افراد در جمعیت
        self.N_GENERATIONS   = 10       # تعداد نسل‌ها
        self.CX_PB           = 0.6      # احتمال کراس‌اور
        self.MUT_PB          = 0.4      # احتمال جهش
        self.EARLY_STOPPING_THRESHOLD = 0.95

        random.seed(42)
        np.random.seed(42)

        self.global_data = None
        self.hours_gene = None

    def load_and_prepare_data(self, csv_file='XAUUSD60.csv'):
        """
        داده را می‌خواند و پاکسازی می‌کند و در self.global_data قرار می‌دهد.
        """
        cleaner = DataCleaner(csv_file)
        data = cleaner.clean_data()
        self.global_data = data
        if 'time' in self.global_data.columns:
            self.global_data['Hour'] = self.global_data['time'].dt.hour

        valid_hours = sorted(self.global_data['Hour'].unique())
        self.hours_gene = HoursGene(valid_hours=valid_hours)

    ###########################################################
    # 5.1) توابع موردنیاز DEAP (init_individual, evaluate_individual, ...)
    ###########################################################
    def init_individual(self):
        """
        ساخت یک فرد (کروموزوم) با هایپرپارامترهای LSTM و SMOTE و ...
        ساختار: [
          window_size, 
          k_neighbors_smote,
          top_k_features,
          num_hours_for_mask,
          hours_mask,
          lstm_units,
          dropout_rate,
          num_layers,
          batch_size,
          epochs
        ]
        شما می‌توانید هایپرپارامترهای بیشتری (یا کمتری) اضافه کنید.
        """
        window_size       = random.randint(3, 20)
        k_neighbors_smote = random.randint(2, 5)
        top_k_feat        = random.randint(50, 150)
        num_hours         = random.randint(5, 20)
        hours_mask        = self.hours_gene.init_hours_subset(num_hours)

        # هایپرهای مخصوص LSTM
        lstm_units  = random.randint(32, 128)   # تعداد واحدهای هر لایه LSTM
        dropout_rate= round(random.uniform(0.1, 0.5), 2)
        num_layers  = random.randint(1, 3)      # تعداد لایه‌های LSTM
        batch_size  = random.choice([16, 32, 64])
        epochs      = random.choice([10, 20, 30])

        individual = [
            window_size,
            k_neighbors_smote,
            top_k_feat,
            num_hours,
            hours_mask,
            lstm_units,
            dropout_rate,
            num_layers,
            batch_size,
            epochs
        ]
        return creator.Individual(individual)

    def evaluate_individual(self, individual):
        """
        [0=window_size, 
         1=k_neighbors, 
         2=top_k_features,
         3=num_hours, 
         4=hours_mask (لیستی از 0/1),
         5=lstm_units,
         6=dropout_rate,
         7=num_layers,
         8=batch_size,
         9=epochs]

        هدف: آموزش مدل LSTM با این هایپرپارامترها روی دادهٔ train با کراس‌ولیدیشن سری زمانی
             سپس محاسبه میانگین F1.
        در صورت کمبود داده، می‌توانید تعداد splits در TimeSeriesSplit را کم/زیاد کنید.
        """
        window_size  = individual[0]
        k_neighbors  = individual[1]
        top_k_feat   = individual[2]
        # num_hours    = individual[3]   # فعلاً استفاده نمی‌کنیم مگر بخواهیم اعمال ماسک ساعت کنیم
        # hours_mask   = individual[4]   # همینطور

        lstm_units   = individual[5]
        dropout_rate = individual[6]
        num_layers   = individual[7]
        batch_size   = individual[8]
        epochs       = individual[9]

        # تقسیم داده به train (70%) در همین تابع
        n = len(self.global_data)
        train_end = int(0.70 * n)
        data_train = self.global_data.iloc[:train_end].copy()

        # آماده‌سازی داده
        prep = PrepareDataForTrain()
        X_train, y_train = prep.ready(data_train, window=window_size, top_k_features=top_k_feat)

        if len(X_train) == 0:
            logging.warning(f"Individual {individual} has empty X_train after preparation.")
            return (0.0,)

        # بالانس کلاس مثبت با SMOTE (اختیاری)
        sm = get_smote(Counter(y_train)[1], k_neighbors=k_neighbors)
        if sm:
            try:
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception as e:
                logging.error(f"SMOTE failed for individual {individual}: {e}")
                return (0.0,)

        # شکل‌دهی برای LSTM => (samples, timesteps, features)
        # در حال حاضر X_train به شکل (n_samples, window_size * n_features) است.
        # باید آن را تبدیل به (n_samples, window_size, n_features)
        # ابتدا num_features را پیدا می‌کنیم:
        num_features = X_train.shape[1] // window_size
        if (window_size * num_features) != X_train.shape[1]:
            # ناسازگاری شکل؛ برمی‌گردانیم 0
            logging.warning(f"Shape mismatch for individual {individual}: window_size={window_size}, num_features={num_features}")
            return (0.0,)
        X_train_np = X_train.values.reshape((X_train.shape[0], window_size, num_features))
        y_train_np = y_train.values

        # کراس‌ولیدیشن با TimeSeriesSplit
        # اگر دیتاست کوچک است، splits را خیلی بزرگ نکنید
        tscv = TimeSeriesSplit(n_splits=3)
        f1_scores = []

        for train_idx, val_idx in tscv.split(X_train_np, y_train_np):
            X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
            y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]

            # ساخت مدل LSTM
            input_shape = (window_size, num_features)
            model = build_lstm_model(
                input_shape=input_shape,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                num_layers=num_layers,
                optimizer='adam'
            )

            # آموزش
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

            # پیش‌بینی روی val
            y_pred_val = (model.predict(X_val) >= 0.5).astype(int)
            score = f1_score(y_val, y_pred_val)
            f1_scores.append(score)

            # پاکسازی session
            K.clear_session()

        if not f1_scores:
            return (0.0,)
        mean_f1 = float(np.mean(f1_scores))
        logging.info(f"Individual {individual} - Mean F1: {mean_f1:.4f}")
        return (mean_f1,)

    def evaluate_final(self, best_ind):
        """
        بر اساس فرد نهایی (best_ind)، 
        داده را به 4 بخش تقسیم می‌کنیم (train=70%, threshold=10%, test=17%, final=3%)
        و آستانه را روی threshold & test پیدا کرده و در بخش final ارزیابی می‌کنیم.
        """
        n = len(self.global_data)
        train_end = int(0.70*n)
        threshold_start = train_end
        threshold_end = train_end + int(0.10*n)
        test_start = threshold_end
        final_test_start = n - int(0.03*n)
        if final_test_start < test_start:
            final_test_start = test_start + 1

        data_train   = self.global_data.iloc[:train_end].copy()
        data_thresh  = self.global_data.iloc[threshold_start:threshold_end].copy()
        data_test    = self.global_data.iloc[threshold_end:final_test_start].copy()
        data_final   = self.global_data.iloc[final_test_start:].copy()

        # استخراج هایپرپارامترها
        window_size  = best_ind[0]
        k_neighbors  = best_ind[1]
        top_k_feat   = best_ind[2]
        lstm_units   = best_ind[5]
        dropout_rate = best_ind[6]
        num_layers   = best_ind[7]
        batch_size   = best_ind[8]
        epochs       = best_ind[9]

        logging.info(f"Evaluating final model with best individual: {best_ind}")

        # آماده‌سازی Train
        prep = PrepareDataForTrain()
        X_train, y_train = prep.ready(data_train, window=window_size, top_k_features=top_k_feat)
        sm = get_smote(Counter(y_train)[1], k_neighbors=k_neighbors)
        if sm:
            try:
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception as e:
                logging.error(f"SMOTE failed during final training: {e}")

        # شکل‌دهی برای LSTM
        num_features = X_train.shape[1] // window_size
        if (window_size * num_features) != X_train.shape[1]:
            logging.error(f"Shape mismatch in final training: window_size={window_size}, num_features={num_features}")
            return 0.0

        X_train_np = X_train.values.reshape((X_train.shape[0], window_size, num_features))
        y_train_np = y_train.values

        # ساخت مدل نهایی و fit روی تمام train
        input_shape = (window_size, num_features)
        final_model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            optimizer='adam'
        )
        final_model.fit(
            X_train_np, y_train_np,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        logging.info("Final model trained on the entire training set.")

        # تابع کمکی برای ساخت X,y هر بخشی و ری‌شیپ آن
        def make_xy_for_section(df):
            X_sec, y_sec = prep.ready(df, window=window_size, top_k_features=top_k_feat)
            if len(X_sec) == 0:
                return None, None
            X_sec = X_sec.reindex(columns=X_train.columns, fill_value=0.0)  # اطمینان از برابر بودن ستون‌ها
            X_sec_np = X_sec.values.reshape((X_sec.shape[0], window_size, num_features))
            y_sec_np = y_sec.values
            return X_sec_np, y_sec_np

        # ===========
        # داده THRESH
        # ===========
        X_thresh_np, y_thresh_np = make_xy_for_section(data_thresh)
        if X_thresh_np is None or y_thresh_np is None or len(X_thresh_np) == 0:
            logging.error("Threshold data is empty after preparation.")
            return 0.0
        y_proba_thresh = final_model.predict(X_thresh_np).ravel()  # خروجی np.array
        tfinder = ThresholdFinder(steps=3000, min_predictions_ratio=2/3)
        neg_th1, pos_th1, acc_th1, w1, l1 = tfinder.find_best_thresholds(y_proba_thresh, y_thresh_np)
        logging.info(f"Threshold Data - Neg: {neg_th1:.4f}, Pos: {pos_th1:.4f}, Acc: {acc_th1:.4f}")

        # ===========
        # داده TEST
        # ===========
        X_test_np, y_test_np = make_xy_for_section(data_test)
        if X_test_np is None or y_test_np is None or len(X_test_np) == 0:
            logging.error("Test data is empty after preparation.")
            return 0.0
        y_proba_test = final_model.predict(X_test_np).ravel()
        neg_th2, pos_th2, acc_test, w2, l2 = tfinder.find_best_thresholds(y_proba_test, y_test_np)
        logging.info(f"Test Data - Neg: {neg_th2:.4f}, Pos: {pos_th2:.4f}, Acc: {acc_test:.4f}")

        # آستانه نهایی میانگین دو بخش
        avg_neg = (neg_th1 + neg_th2) / 2
        avg_pos = (pos_th1 + pos_th2) / 2
        logging.info(f"Average Thresholds - Neg: {avg_neg:.4f}, Pos: {avg_pos:.4f}")

        # ===========
        # داده FINAL
        # ===========
        X_final_np, y_final_np = make_xy_for_section(data_final)
        if X_final_np is None or y_final_np is None or len(X_final_np) == 0:
            logging.error("Final test data is empty after preparation.")
            return 0.0
        y_proba_final = final_model.predict(X_final_np).ravel()

        # محاسبه دقت با آستانه‌های avg_neg و avg_pos
        final_acc = self.calculate_accuracy_with_thresholds(
            y_proba_final, y_final_np,
            neg_threshold=avg_neg, pos_threshold=avg_pos,
            min_predictions_ratio=2/3
        )
        logging.info(f"Final Test Accuracy: {final_acc:.4f}")
        logging.info(f"Final Thresholds - Neg: {avg_neg:.4f}, Pos: {avg_pos:.4f}")

        return final_acc

    def calculate_accuracy_with_thresholds(self, proba, ytrue,
                                           neg_threshold=0.3, pos_threshold=0.7,
                                           min_predictions_ratio=2/3):
        """
        مشابه پروژه اصلی: ابتدا با دو آستانه pos و neg طبقه‌بندی می‌کنیم.
        اگر بخشی از نمونه‌ها نامطمئن ماندند (-1)، تا جایی که
        min_predictions_ratio تامین شود، آن‌ها را حول 0.5 تعیین تکلیف می‌کنیم.
        """
        y_pred = np.full_like(ytrue, -1)
        y_pred[proba <= neg_threshold] = 0
        y_pred[proba >= pos_threshold] = 1

        uncertain_mask = (y_pred == -1)
        num_uncertain = np.sum(uncertain_mask)
        max_allowed_uncertain = len(ytrue) - int(len(ytrue)*min_predictions_ratio)
        if num_uncertain > max_allowed_uncertain and max_allowed_uncertain > 0:
            dist05 = np.abs(proba[uncertain_mask] - 0.5)
            uncertain_indices = np.where(uncertain_mask)[0]
            sorted_inds = uncertain_indices[np.argsort(dist05)]
            can_fix_count = num_uncertain - max_allowed_uncertain
            fix_idx = sorted_inds[:can_fix_count]
            y_pred[fix_idx] = (proba[fix_idx] >= 0.5).astype(int)

        valid_mask = (y_pred != -1)
        if np.sum(valid_mask) == 0:
            return 0.0
        correct = np.sum(y_pred[valid_mask] == ytrue[valid_mask])
        acc = correct / np.sum(valid_mask)
        return acc

    ###########################################################
    # 5.2) متد اصلی main برای اجرای کل پروژه
    ###########################################################
    def main(self):
        """
        اجرای کل فرایند:
          1) ساخت جمعیت اولیه
          2) ارزیابی افراد
          3) تکرار تا N_GENERATIONS
          4) انتخاب بهترین فرد
          5) ارزیابی نهایی با تقسیم‌بندی 70%,10%,17%,3%
        """
        if self.global_data is None:
            logging.error("No data loaded. Please run load_and_prepare_data first.")
            print("No data loaded. Please run load_and_prepare_data first.")
            return None, 0.0, 0.0

        # تعیین تعداد پردازش‌ها بر اساس تعداد هسته‌های CPU
        num_cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_cpus - 1 if num_cpus > 1 else 1)
        toolbox.register("map", pool.map)

        # ساخت جمعیت اولیه
        pop = []
        for _ in range(self.POPULATION_SIZE):
            ind = self.init_individual()
            pop.append(ind)

        logging.info(f"Initial population size: {len(pop)}")

        # ارزیابی اولیه
        try:
            fitnesses = toolbox.map(self.evaluate_individual, pop)
            for ind, fitv in zip(pop, fitnesses):
                ind.fitness.values = fitv
        except Exception as e:
            logging.error(f"Error evaluating initial population: {e}")
            pool.close()
            pool.join()
            print(f"Error evaluating initial population: {e}")
            return None, 0.0, 0.0

        best_overall = 0.0
        for gen in range(1, self.N_GENERATIONS+1):
            logging.info(f"Generation {gen}/{self.N_GENERATIONS}")
            print(f"Generation {gen}/{self.N_GENERATIONS}")

            offspring = toolbox.select(pop, len(pop))
            offspring = [copy.deepcopy(o) for o in offspring]

            # انجام crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CX_PB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # انجام mutation
            for mut in offspring:
                if random.random() < self.MUT_PB:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            # ارزیابی مجدد فرزندان جدید
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(self.evaluate_individual, invalids)
            for ind_, fv in zip(invalids, fits):
                ind_.fitness.values = fv

            pop[:] = offspring
            best_ind = tools.selBest(pop, 1)[0]
            best_f1 = best_ind.fitness.values[0]

            if best_f1 > best_overall:
                best_overall = best_f1

            logging.info(f"Best so far => F1={best_f1:.4f}")
            print(f"Best so far => F1={best_f1:.4f}")

            if best_f1 >= self.EARLY_STOPPING_THRESHOLD:
                logging.info("Early stopping triggered.")
                print("Early stopping triggered.")
                break

        # انتخاب بهترین
        best_ind = tools.selBest(pop, 1)[0]
        best_f1  = best_ind.fitness.values[0]
        logging.info(f"Optimization finished. Best individual: {best_ind}, Best F1: {best_f1:.4f}")
        print("Optimization finished.")
        print("Best individual:", best_ind)
        print(f"Best F1={best_f1:.4f}")

        # ارزیابی نهایی
        final_acc = self.evaluate_final(best_ind)
        logging.info(f"Final Test Accuracy = {final_acc:.4f}")
        print(f"Final Test Accuracy = {final_acc:.4f}")

        pool.close()
        pool.join()

        return best_ind, best_f1, final_acc


############################################################
# DEAP Setup
############################################################
# تعریف نوع Fitness و Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# عملگرهای ژنتیک
toolbox.register("mate", tools.cxTwoPoint)

def mutate_individual(individual, indpb=0.2):
    """
    تابع جهش (Mutation) برای یک کروموزوم:
    ساختار فرد:
      [0] window_size
      [1] k_neighbors
      [2] top_k_features
      [3] num_hours
      [4] hours_mask
      [5] lstm_units
      [6] dropout_rate
      [7] num_layers
      [8] batch_size
      [9] epochs
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:
                # window_size
                individual[i] = random.randint(3, 20)
            elif i == 1:
                # k_neighbors
                individual[i] = random.randint(2, 5)
            elif i == 2:
                # top_k_features
                individual[i] = random.randint(50, 150)
            elif i == 3:
                # num_hours
                individual[i] = random.randint(5, 20)
            elif i == 4:
                # hours_mask => بازتولید ماسک
                pass  # در اینجا ماسک را به صورت ثابت نگه می‌داریم
            elif i == 5:
                # lstm_units
                individual[i] = random.randint(32, 128)
            elif i == 6:
                # dropout_rate
                individual[i] = round(random.uniform(0.1, 0.5), 2)
            elif i == 7:
                # num_layers
                individual[i] = random.randint(1, 3)
            elif i == 8:
                # batch_size
                individual[i] = random.choice([16, 32, 64])
            elif i == 9:
                # epochs
                individual[i] = random.choice([10, 20, 30])
    return (individual,)

toolbox.register("mutate", mutate_individual, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


############################################################
# بخش اصلی اجرا
############################################################
if __name__=="__main__":
    # تنظیم لاگ‌گیری
    logging.basicConfig(
        filename='lstm_genetic_project.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Project started.")

    # ساخت شیء اجرا
    runner = GeneticAlgorithmRunner()

    # بارگذاری داده
    runner.load_and_prepare_data('XAUUSD_M30.csv')  # در صورت نیاز نام فایل را عوض کنید
    logging.info("Data loaded and cleaned.")

    # تنظیم تعداد نخ‌های TensorFlow
    num_cpus = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(num_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
    logging.info(f"TensorFlow configured to use {num_cpus} threads.")

    # اجرای فرایند اصلی
    best_individual, best_f1, final_accuracy = runner.main()
    logging.info(f"Done. Best F1={best_f1:.4f}, Final Accuracy={final_accuracy:.4f}")
    print(f"\n\nDone. Best F1={best_f1:.4f}, Final Accuracy={final_accuracy:.4f}")
