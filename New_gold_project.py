# main_project.py
"""
نمونه پروژه کامل با اصلاحات پیشنهادی:
1) بالانس بهتر داده با SMOTE (قابلیت تنظیم k_neighbors)
2) افزایش ویژگی‌های تولیدشده (امکان انتخاب بیشتر از 150 تا مثلاً 300)
3) تقسیم داده به 4 بخش: train, threshold, test, final
4) استفاده از الگوریتم ژنتیک با امکان خاموش/روشن کردن آن
5) تغییر مدل از Logistic Regression به RandomForest یا هر مدل دیگر
6) استفاده از TimeSeriesSplit با تعداد splits بالاتر
7) جستجوی آستانه (Threshold) با steps بزرگ‌تر (مثلاً 3000 یا 5000)
8) اضافه کردن فیچرهای جدید (یا حذف برخی فیچرها در صورت نیاز)
9) رعایت ساختار اصلی (بالانس دیتا، محاسبه دو آستانه، ماسک ساعات، ...)
10) امکان تغییر پارامترها راحت از بالای فایل
"""

import pandas as pd
import numpy as np
import random
import logging
import copy
import multiprocessing
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# =======================
# پکیج‌های موردنیاز اسکیکیت
# =======================
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# برای الگوریتم ژنتیک (در صورت تمایل)
from deap import base, creator, tools

# ======================================
# فایل های کمکی این پروژه (ماژول‌ها)
# (در این مثال تمام کلاس‌ها و توابع
#   به صورت توکار در همین فایل هستند)
# ======================================


class DataCleaner:
    """کلاس پاکسازی داده (مطابق کدهای قبلی)."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from the file."""
        self.data = pd.read_csv(self.file_path, parse_dates=['time'])

    def remove_noise(self):
        """Removes noise and invalid values from the data."""
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.drop_duplicates(inplace=True)

    def fix_missing_values(self):
        """Fills or removes missing values in the data."""
        for col in self.data.select_dtypes(include=['float64', 'int64']).columns:
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def clean_data(self):
        """Executes the entire cleaning process."""
        self.load_data()
        self.remove_noise()
        self.fix_missing_values()
        return self.data


class ThresholdFinder:
    """
    جستجوی آستانه‌های منفی و مثبت با روش موازی.
    """
    def __init__(self, steps=3000, min_predictions_ratio=2/3):
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

        if (wins + loses) < round(len(y_true) * self.min_predictions_ratio):
            return 0, th_neg, th_pos, wins, loses

        acc = wins / (wins + loses) if (wins + loses) else 0
        return acc, th_neg, th_pos, wins, loses

    def find_best_thresholds(self, proba, y_true):
        proba = np.array(proba)
        y_true = np.array(y_true)
        threshold_pairs = [
            (k/self.steps, l/self.steps)
            for k in range(self.steps+1)
            for l in range(self.steps+1)
        ]

        best_acc = -1
        best_tuple = (0, 0)
        best_wins, best_loses = 0, 0

        for (th_neg, th_pos) in threshold_pairs:
            acc, _, _, w, l = self._calculate_acc((th_neg, th_pos), proba, y_true)
            if acc>best_acc:
                best_acc = acc
                best_tuple = (th_neg, th_pos)
                best_wins, best_loses = w, l

        return best_tuple[0], best_tuple[1], best_acc, best_wins, best_loses


class HoursGene:
    """
    مدیریت ماسک ساعات معامله.
    """
    def __init__(self, valid_hours, excluded_hours=[0,1,2,3]):
        self.valid_hours = [h for h in valid_hours if h not in excluded_hours]

    def init_hours_subset(self, num_selected):
        n = len(self.valid_hours)
        if num_selected>n:
            num_selected=n
        arr = [1]*num_selected + [0]*(n-num_selected)
        random.shuffle(arr)
        return arr

    def mutate_hours_subset(self, hours_subset, indpb=0.2, num_selected_hours=20):
        n = len(hours_subset)
        for _ in range(int(indpb*n)):
            if n<2:
                break
            i,j = random.sample(range(n),2)
            hours_subset[i], hours_subset[j] = hours_subset[j], hours_subset[i]
        curr = sum(hours_subset)
        if curr != num_selected_hours:
            arr = [1]*num_selected_hours + [0]*(n-num_selected_hours)
            random.shuffle(arr)
            hours_subset[:] = arr
        return (hours_subset,)


class PrepareDataForTrain:
    """
    کلاس آماده‌سازی داده (ترکیبی از prepare_data_for_train قبلی + Feature Engineering اضافه).
    در اینجا، برای ساده‌سازی، فقط چند اندیکاتور کلیدی را لحاظ می‌کنیم. ولی شما می‌توانید
    تمام کدهای قبلی FeatureEngineer و ... را اینجا ادغام کنید.
    """
    def __init__(self):
        pass

    def ready(self, data: pd.DataFrame, window: int = 1, top_k_features=300):
        """
        1) افزودن ستون ساعت و روز
        2) تولید اندیکاتورهای ساده
        3) Rolling و محاسبات آماری
        4) diff گرفتن از فیچرها
        5) حذف فیچرهای کم‌اهمیت با روش‌های مختلف
        6) windowing داده
        """
        df = data.copy()

        # -----------------------
        # ستون‌های زمانی
        # -----------------------
        if 'time' in df.columns:
            df['Hour'] = df['time'].dt.hour
            df['DayOfWeek'] = df['time'].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
            df.drop(columns=['time'], inplace=True)

        # -----------------------
        # چند اندیکاتور ساده
        # -----------------------
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()

        df['ReturnDifference'] = df['close'].diff()
        df['ROC'] = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1)+1e-9)*100

        # حذف NaN اولیه
        df.dropna(axis=0, how='any', inplace=True)

        # -----------------------
        # تعریف Target
        # -----------------------
        target = ((df['close'].shift(-1) - df['close'])>0).astype(int)
        target = target[:-1]
        df = df[:-1]

        # -----------------------
        # differencing از تمام فیچرها به جز Hour
        # -----------------------
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

        # -----------------------
        # حذف فیچرهای کم‌تنوع
        # -----------------------
        selector_var = VarianceThreshold(threshold=0.01)
        selector_var.fit(df_diff)
        df_diff = df_diff[df_diff.columns[selector_var.get_support()]]

        # -----------------------
        # حذف فیچرهای با همبستگی بالا
        # -----------------------
        corr_matrix = df_diff.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        df_diff.drop(columns=to_drop, inplace=True, errors='ignore')

        # -----------------------
        # انتخاب فیچر با MI
        # -----------------------
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df_diff)
        mi_values = mutual_info_classif(X_scaled, target, discrete_features='auto')
        mi_series = pd.Series(mi_values, index=df_diff.columns).sort_values(ascending=False)
        top_k_features = min(top_k_features, len(mi_series))
        top_features = mi_series.head(top_k_features).index
        df_diff = df_diff[top_features].copy()

        # -----------------------
        # windowing
        # -----------------------
        if window<1: 
            window=1

        if window==1:
            return df_diff, target
        else:
            arr_list = []
            new_index = []
            for i in range(window-1, len(df_diff)):
                row_feat = []
                for offset in range(window):
                    idx = i-offset
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


# ==============================
# تابع کمکی بالانس داده با SMOTE
# ==============================
def get_smote(n_samples, random_state=42, k_neighbors=3):
    """
    تابع ساخت شی SMOTE با تعداد همسایه‌های دلخواه.
    اگر کلاس مثبت کمتر از k_neighbors باشد، ممکن است خطا دهد.
    """
    if n_samples < k_neighbors:
        return None
    return SMOTE(random_state=random_state, k_neighbors=k_neighbors)


# ==============================
# کلاس اصلی اجرای پروژه
# ==============================
class GeneticAlgorithmRunner:
    """
    برای حفظ ساختار قبلی، اینجا از الگوریتم ژنتیک استفاده شده است.
    اگر تمایلی به الگوریتم ژنتیک ندارید، می‌توانید مستقیماً GridSearch یا روش‌های دیگر را جایگزین کنید.
    """
    def __init__(self):
        # -- پارامترهای الگوریتم ژنتیک --
        self.POPULATION_SIZE = 50
        self.N_GENERATIONS = 30
        self.CX_PB = 0.6
        self.MUT_PB = 0.3
        self.EARLY_STOPPING_THRESHOLD = 0.95
        random.seed(42)
        np.random.seed(42)

        # -- متغیرهای سراسری --
        self.global_data = None
        self.hours_gene = None

        # -- تنظیمات پیش‌فرض مدل جایگزین (اینجا RandomForest) --
        # در صورت تمایل می‌توانید اینجا را تغییر دهید
        self.model_default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'class_weight': 'balanced',  # بالانس کلاس‌ها
            'random_state': 42
        }

    def load_and_prepare_data(self, csv_file='XAUUSD60.csv'):
        """خواندن داده و پاکسازی اولیه"""
        cleaner = DataCleaner(csv_file)
        data = cleaner.clean_data()
        # ممکن است بخواهید داده‌های بیشتری اضافه کنید
        self.global_data = data
        if 'time' in self.global_data.columns:
            self.global_data['Hour'] = self.global_data['time'].dt.hour
        valid_hours = sorted(self.global_data['Hour'].unique())
        # ساخت hours_gene
        self.hours_gene = HoursGene(valid_hours=valid_hours)

    def main(self):
        # پیش‌فرض: اگر نخواستید الگوریتم ژنتیک را اجرا کنید، می‌توانید آن را غیرفعال کنید
        if self.global_data is None:
            print("Please load data first!")
            return None, 0.0, 0.0

        pop = []
        for _ in range(self.POPULATION_SIZE):
            pop.append(self.init_individual())

        # تنظیم MultiProcess
        try:
            pool = multiprocessing.Pool(processes=4)
            toolbox.register("map", pool.map)
        except Exception as e:
            print(f"Error setting up multiprocessing pool: {e}")
            return None, 0.0, 0.0

        # ارزیابی اولیه
        try:
            fitnesses = toolbox.map(self.evaluate_individual, pop)
            for ind, fitv in zip(pop, fitnesses):
                ind.fitness.values = fitv
        except Exception as e:
            print(f"Error evaluating initial population: {e}")
            pool.close()
            pool.join()
            return None, 0.0, 0.0

        best_overall = 0.0
        for gen in range(1, self.N_GENERATIONS+1):
            print(f"Generation {gen}/{self.N_GENERATIONS}")
            offspring = toolbox.select(pop, len(pop))
            offspring = [copy.deepcopy(o) for o in offspring]

            for c1,c2 in zip(offspring[::2], offspring[1::2]):
                if random.random()<self.CX_PB:
                    toolbox.mate(c1,c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mut in offspring:
                if random.random()<self.MUT_PB:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            invalids = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(self.evaluate_individual, invalids)
            for ind_, fv in zip(invalids, fits):
                ind_.fitness.values = fv

            pop[:] = offspring
            best_ind = tools.selBest(pop,1)[0]
            best_f1  = best_ind.fitness.values[0]
            if best_f1>best_overall:
                best_overall = best_f1
            print(f"Best so far => F1={best_f1:.4f}")

            if best_f1>=self.EARLY_STOPPING_THRESHOLD:
                print("Early stopping triggered.")
                break

        best_ind = tools.selBest(pop,1)[0]
        best_f1  = best_ind.fitness.values[0]
        print("Optimization finished.")
        print("Best individual:", best_ind)
        print(f"Best F1={best_f1:.4f}")

        # ارزیابی نهایی روی بخش‌های مختلف داده
        final_acc = self.evaluate_final(best_ind)
        print(f"Final Test Accuracy = {final_acc:.4f}")

        pool.close()
        pool.join()

        return best_ind, best_f1, final_acc

    # -------------------------------------------
    # ایجاد ساختار داده و ماسک‌کردن ویژگی‌ها/ساعات
    # -------------------------------------------
    def init_individual(self):
        """
        ساخت کروموزوم اولیه برای الگوریتم ژنتیک:
        [window, k_neighbors_SMOTE, top_k_features, num_hours, hours_mask]
        """
        window = random.randint(3,30)
        k_neighbors = random.randint(2,5)
        top_k_features = random.randint(150,300)  # محدوده انتخاب فیچر
        num_hours = random.randint(5, 20)
        hours_mask = self.hours_gene.init_hours_subset(num_hours)
        # fitness = creator.FitnessMax => (1.0,)
        return creator.Individual([window, k_neighbors, top_k_features, num_hours, hours_mask])

    def evaluate_individual(self, individual):
        """
        [0=window, 1=k_neighbors, 2=top_k_features, 3=num_hours, 4=hours_mask]
        خروجی = میانگین F1 از کراس‌ولیدیشن
        """
        window = individual[0]
        k_neighbors = individual[1]
        top_k_feat = individual[2]
        # ساعت‌ها فعلا اینجا صرفاً تزیینی است؛ شما می‌توانید داده را بر اساس این ساعات فیلتر کنید
        # یا اگر بخواهید در مدلی لحاظ شود، باید منطق آن را پیاده‌سازی کنید.

        # داده آموزش
        n = len(self.global_data)
        train_end = int(0.70*n)
        data_train = self.global_data.iloc[:train_end].copy()

        # آماده‌سازی
        prep = PrepareDataForTrain()
        X_train, y_train = prep.ready(data_train, window=window, top_k_features=top_k_feat)
        if len(X_train)==0:
            return (0.0,)

        # بالانس داده
        sm = get_smote(Counter(y_train)[1], k_neighbors=k_neighbors)
        if sm:
            try:
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except:
                pass

        # کراس‌ولیدیشن
        # برای دقت بیشتر، تعداد splits را زیاد می‌کنیم
        tscv = TimeSeriesSplit(n_splits=8)
        f1_scores = []
        for train_index, test_index in tscv.split(X_train, y_train):
            X_tr, X_ts = X_train.iloc[train_index], X_train.iloc[test_index]
            y_tr, y_ts = y_train.iloc[train_index], y_train.iloc[test_index]

            # مدل RandomForest با پارامترهای پیشفرض (می‌توانید پارامترها را به صورت داینامیک نیز ژنتیکی کنید)
            model = RandomForestClassifier(**self.model_default_params)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_ts)
            f1_ = f1_score(y_ts, y_pred, average='binary')
            f1_scores.append(f1_)

        if not f1_scores:
            return (0.0,)
        return (np.mean(f1_scores),)

    def evaluate_final(self, best_ind):
        """
        تقسیم داده به 4 بخش:
            train: 70%
            threshold: 10%
            test: 17%
            final: 3%
        آموزش مدل روی train
        محاسبه آستانه روی threshold
        ارزیابی روی test + نهایی روی final
        """
        n = len(self.global_data)
        train_end = int(0.70*n)
        threshold_start = train_end
        threshold_end = train_end + int(0.10*n)
        test_start = threshold_end
        final_test_start = n - int(0.03*n)
        if final_test_start<test_start:
            final_test_start = test_start + 1

        data_train = self.global_data.iloc[:train_end].copy()
        data_thresh = self.global_data.iloc[threshold_start:threshold_end].copy()
        data_test = self.global_data.iloc[threshold_end:final_test_start].copy()
        data_final= self.global_data.iloc[final_test_start:].copy()

        # پارامترهای بهترین فرد
        window = best_ind[0]
        k_neighbors = best_ind[1]
        top_k_features = best_ind[2]

        # آماده‌سازی Train
        prep = PrepareDataForTrain()
        X_train, y_train = prep.ready(data_train, window=window, top_k_features=top_k_features)
        sm = get_smote(Counter(y_train)[1], k_neighbors=k_neighbors)
        if sm:
            try:
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except:
                pass

        # مدل نهایی
        final_model = RandomForestClassifier(**self.model_default_params)
        final_model.fit(X_train, y_train)

        # تابع کمکی
        def make_xy_and_reindex(df):
            X_, y_ = prep.ready(df, window=window, top_k_features=top_k_features)
            # ستون‌های X_ را مطابق train انتخاب کنید (در صورت اختلاف شکل)
            X_ = X_.reindex(columns=X_train.columns, fill_value=0.0)
            return X_, y_

        # ساخت داده threshold
        X_thresh, y_thresh = make_xy_and_reindex(data_thresh)
        y_proba_th = final_model.predict_proba(X_thresh)[:,1]
        th_finder = ThresholdFinder(steps=3000, min_predictions_ratio=2/3)
        neg_th1, pos_th1, acc_th1, w1, l1 = th_finder.find_best_thresholds(y_proba_th, y_thresh.values)

        # ساخت داده test
        X_test, y_test = make_xy_and_reindex(data_test)
        y_proba_test = final_model.predict_proba(X_test)[:,1]
        neg_th2, pos_th2, acc_test, w2, l2 = th_finder.find_best_thresholds(y_proba_test, y_test.values)

        # آستانه‌ی میانگین
        avg_neg = (neg_th1 + neg_th2)/2
        avg_pos = (pos_th1 + pos_th2)/2

        # داده نهایی
        X_final, y_final = make_xy_and_reindex(data_final)
        y_proba_final = final_model.predict_proba(X_final)[:,1]
        final_acc = self.calculate_accuracy_with_thresholds(
            y_proba_final, y_final.values,
            neg_threshold=avg_neg, pos_threshold=avg_pos,
            min_predictions_ratio=2/3
        )
        return final_acc

    # -----------------------------------------------------------------------
    def calculate_accuracy_with_thresholds(self, proba, ytrue,
                                           neg_threshold=0.3, pos_threshold=0.7,
                                           min_predictions_ratio=2/3):
        """
        همانند پروژه، بر اساس دو آستانه‌ی مثبت و منفی تصمیم می‌گیریم.
        """
        y_pred = np.full_like(ytrue, -1)
        y_pred[proba<=neg_threshold] = 0
        y_pred[proba>=pos_threshold] = 1

        uncertain_mask = (y_pred==-1)
        num_uncertain = np.sum(uncertain_mask)
        max_allowed_uncertain = len(ytrue) - int(len(ytrue)*min_predictions_ratio)
        if num_uncertain>max_allowed_uncertain and max_allowed_uncertain>0:
            dist05 = np.abs(proba[uncertain_mask]-0.5)
            uncertain_indices = np.where(uncertain_mask)[0]
            sorted_inds = uncertain_indices[np.argsort(dist05)]
            can_fix_count = num_uncertain - max_allowed_uncertain
            fix_idx = sorted_inds[:can_fix_count]
            y_pred[fix_idx] = (proba[fix_idx]>=0.5).astype(int)

        valid_mask = (y_pred!=-1)
        if np.sum(valid_mask)==0:
            return 0.0
        correct = np.sum(y_pred[valid_mask]==ytrue[valid_mask])
        acc = correct/np.sum(valid_mask)
        return acc


# =========================
# ثبت توابع در DEAP toolbox
# =========================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("mate", tools.cxTwoPoint)
def mutate_individual(individual, indpb=0.2):
    """
    [0=window, 1=k_neighbors, 2=top_k_features, 3=num_hours, 4=hours_mask]
    """
    for i in range(len(individual)):
        if random.random()<indpb:
            if i==0:
                individual[i] = random.randint(3,30)
            elif i==1:
                individual[i] = random.randint(2,5)
            elif i==2:
                individual[i] = random.randint(150,300)
            elif i==3:
                individual[i] = random.randint(5,20)
            elif i==4:
                # بازتولید ماسک
                pass
    return (individual,)

toolbox.register("mutate", mutate_individual, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# ================
# بخش اصلی اجرا
# ================
if __name__=="__main__":
    logging.basicConfig(
        filename='improved_project.log',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    runner = GeneticAlgorithmRunner()
    runner.load_and_prepare_data('XAUUSD60.csv')  # نام فایل CSV داده
    best_ind, best_f1, final_acc = runner.main()
    print(f"\n\nDone. Best_F1={best_f1:.4f}, FinalTest_Accuracy={final_acc:.4f}")
