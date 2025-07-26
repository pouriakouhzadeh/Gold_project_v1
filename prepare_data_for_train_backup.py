import numpy as np
import pandas as pd
import ta
import warnings
from scipy.stats import skew, kurtosis
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import logging

# اندیکاتورهای سفارشی
from custom_indicators import (
    KSTCustomIndicator,
    VortexCustomIndicator,
    CustomIchimokuIndicator,
    CustomWilliamsRIndicator,
    CustomVolumeRateOfChangeIndicator,
    CustomPivotPointIndicator,
    CustomCandlestickPattern
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.DEBUG,
    filename='prepare_data.log',
    format='%(asctime)s %(levelname)s:%(message)s'
)


class PREPARE_DATA_FOR_TRAIN:
    def __init__(self, filepaths: dict = None, main_timeframe: str = '30T'):
        """
        اگر فایل‌پث داده نشود، پیشفرض:
        {
          '30T': 'XAUUSD_M30.csv',
          '1H':  'XAUUSD_H1.csv',
          '15T': 'XAUUSD_M15.csv',
          '5T':  'XAUUSD_M5.csv'
        }
        """
        if filepaths is None:
            filepaths = {
                '30T': 'XAUUSD_M30.csv',
                '1H':  'XAUUSD_H1.csv',
                '15T': 'XAUUSD_M15.csv',
                '5T':  'XAUUSD_M5.csv'
            }
        self.filepaths = filepaths
        self.main_timeframe = main_timeframe
        logging.info(f"[INIT] filepaths={self.filepaths}, main_timeframe={self.main_timeframe}")

    def _convert_timedelta_to_seconds(self, df: pd.DataFrame):
        """
        هر ستون از نوع TimeDelta64 را به تعداد ثانیه (float) تبدیل می‌کند.
        """
        for col in df.columns:
            if pd.api.types.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds()

    def load_and_process_timeframe(self, timeframe_label: str, filepath: str) -> pd.DataFrame:
        """
        بارگذاری یک تایم‌فریم، resample به main_timeframe درصورت نیاز،
        محاسبه اندیکاتورهای مختلف (بدون حذف هیچ فیچری)،
        فقط پرکردن جاهای خالی و تبدیل TimeDelta به float.
        """
        try:
            data = pd.read_csv(filepath)
            print(f"Data for timeframe {timeframe_label} loaded successfully.")
            logging.info(f"[load_and_process_timeframe] {timeframe_label}: file loaded => shape={data.shape}")
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            logging.error(f"[load_and_process_timeframe] Error loading {timeframe_label}: {e}")
            raise

        if 'time' not in data.columns:
            raise ValueError(f"'time' column not found in {filepath}.")

        # مرتب‌سازی بر اساس زمان
        data['time'] = pd.to_datetime(data['time'])
        data.sort_values('time', inplace=True)
        data.set_index('time', drop=True, inplace=True)

        # اگر تایم‌فریم فعلی با main_timeframe فرق داشت، resample
        if timeframe_label != self.main_timeframe:
            logging.debug(f"[{timeframe_label}] Resampling to main timeframe={self.main_timeframe}")
            data = data.resample(self.main_timeframe).agg({
                'open':  'first',
                'high':  'max',
                'low':   'min',
                'close': 'last',
                'volume':'sum'
            })

        # پرکردن جاهای خالی
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(axis=0, how='all', inplace=True)

        # reset index
        data.reset_index(inplace=True)
        data.rename(columns={'time': f'{timeframe_label}_time'}, inplace=True)

        prefix = f"{timeframe_label}_"

        # محاسبه ستون‌های زمانی
        if f'{timeframe_label}_time' in data.columns:
            data[f'{prefix}Hour']      = data[f'{timeframe_label}_time'].dt.hour
            data[f'{prefix}DayOfWeek'] = data[f'{timeframe_label}_time'].dt.dayofweek
            data[f'{prefix}IsWeekend'] = data[f'{prefix}DayOfWeek'].isin([5,6]).astype(int)

        # Rolling ساده
        data[f'{prefix}ma20'] = data['close'].rolling(20).mean()
        data[f'{prefix}ma50'] = data['close'].rolling(50).mean()
        data[f'{prefix}ma_volume20'] = data['volume'].rolling(20).mean()
        data[f'{prefix}ReturnDifference'] = data['close'].diff()
        data[f'{prefix}ROC'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1)+1e-9)*100

        data[f'{prefix}Rolling_Mean_20']   = data['close'].rolling(20).mean()
        data[f'{prefix}Rolling_Std_20']    = data['close'].rolling(20).std()
        data[f'{prefix}Rolling_Skew_20']   = data['close'].rolling(20).apply(lambda x: skew(x), raw=False)
        data[f'{prefix}Rolling_Kurt_20']   = data['close'].rolling(20).apply(lambda x: kurtosis(x, fisher=True), raw=False)
        data[f'{prefix}Rolling_Median_20'] = data['close'].rolling(20).median()
        data[f'{prefix}Rolling_UpCount_20']= data['close'].rolling(20).apply(lambda x: np.sum(x.diff()>0), raw=False)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(axis=0, how='all', inplace=True)

        # اندیکاتورهای کتابخانه ta
        indicators = ta.add_all_ta_features(
            data.copy(),
            open="open", high="high", low="low", close="close", volume="volume",
            fillna=False
        )
        indicators = indicators.add_prefix(prefix)
        data = pd.concat([data, indicators], axis=1)
        data = data.loc[:, ~data.columns.duplicated()]

        # پارابولیک سار
        data[f'{prefix}Parabolic_SAR'] = ta.trend.PSARIndicator(
            high=data['high'], low=data['low'], close=data['close'],
            step=0.018, max_step=0.2
        ).psar()

        data[f'{prefix}Momentum_14'] = data['close'] - data['close'].shift(14)
        data[f'{prefix}Trix_15']     = ta.trend.TRIXIndicator(data['close'], window=15).trix()
        data[f'{prefix}Ultimate_Osc']= ta.momentum.UltimateOscillator(data['high'], data['low'], data['close'],7,14,28).ultimate_oscillator()
        data[f'{prefix}Daily_Range'] = data['high'] - data['low']

        # HV, GarmanKlass, Parkinson
        logret= np.log(data['close']/(data['close'].shift(1)+1e-9))
        data[f'{prefix}HV_20'] = logret.rolling(20).std()*np.sqrt(24*365)

        try:
            hl_log= np.log(data['high']/(data['low']+1e-9)+1e-9)
            co_log= np.log(data['close']/(data['open']+1e-9)+1e-9)
            data[f'{prefix}GarmanKlass'] = (
                0.5*(hl_log**2) - (2*(co_log**2))**0.5
            ).rolling(20).mean()
        except:
            data[f'{prefix}GarmanKlass'] = np.nan

        try:
            ln_hl= np.log(data['high']/(data['low']+1e-9)+1e-9)
            data[f'{prefix}Parkinson_20']= (ln_hl**2).rolling(20).sum()
            data[f'{prefix}Parkinson_20']= np.sqrt(
                data[f'{prefix}Parkinson_20']*(1/(4*np.log(2)*20))
            )
        except:
            data[f'{prefix}Parkinson_20']= np.nan

        data[f'{prefix}UlcerIndex_14'] = ta.volatility.UlcerIndex(data['close'],14).ulcer_index()
        data[f'{prefix}MFI'] = ta.volume.MFIIndicator(data['high'], data['low'], data['close'], data['volume'],14, fillna=False).money_flow_index()
        data[f'{prefix}EoM_14'] = ta.volume.EaseOfMovementIndicator(data['high'], data['low'], data['volume'],14).sma_ease_of_movement()

        period_dpo=20
        data[f'{prefix}DPO_20'] = data['close'] - data['close'].rolling(period_dpo//2+1).mean()

        data[f'{prefix}MACD']  = ta.trend.MACD(data['close']).macd()
        data[f'{prefix}RSI_14']= ta.momentum.RSIIndicator(data['close'],14).rsi()

        bb= ta.volatility.BollingerBands(data['close'],20,2)
        data[f'{prefix}Bollinger_High'] = bb.bollinger_hband()
        data[f'{prefix}Bollinger_Low']  = bb.bollinger_lband()
        data[f'{prefix}Bollinger_Width']= bb.bollinger_wband()

        data[f'{prefix}SMA10']= ta.trend.SMAIndicator(data['close'],10).sma_indicator()
        data[f'{prefix}SMA50']= ta.trend.SMAIndicator(data['close'],50).sma_indicator()
        data[f'{prefix}SMA10_SMA50_diff']= data[f'{prefix}SMA10']- data[f'{prefix}SMA50']

        data[f'{prefix}EMA10']= ta.trend.EMAIndicator(data['close'],10).ema_indicator()
        data[f'{prefix}EMA50']= ta.trend.EMAIndicator(data['close'],50).ema_indicator()
        data[f'{prefix}EMA10_50_diff']= data[f'{prefix}EMA10']- data[f'{prefix}EMA50']

        data[f'{prefix}ATR_14']= ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'],14).average_true_range()
        data[f'{prefix}Bollinger_Width_Ratio']= data[f'{prefix}Bollinger_Width']/(data[f'{prefix}Rolling_Std_20']+1e-9)

        # اندیکاتورهای کاستوم
        kst= KSTCustomIndicator(data['close'],10,15,20,30,10,10,10,15,9, fillna=True)
        data[f'{prefix}KST_MAIN']   = kst.kst()
        data[f'{prefix}KST_SIGNAL'] = kst.kst_signal()
        data[f'{prefix}KST_DIFF']   = kst.kst_diff()

        vtx= VortexCustomIndicator(data['high'], data['low'], data['close'],14, fillna=True)
        data[f'{prefix}Vortex_Pos']= vtx.vortex_pos()
        data[f'{prefix}Vortex_Neg']= vtx.vortex_neg()

        ichi= CustomIchimokuIndicator(data['high'], data['low'], data['close'],9,26,52)
        data[f'{prefix}Ichimoku_Conversion_Line']= ichi.ichimoku_conversion_line()
        data[f'{prefix}Ichimoku_Base_Line']      = ichi.ichimoku_base_line()
        data[f'{prefix}Ichimoku_A']              = ichi.ichimoku_a()
        data[f'{prefix}Ichimoku_B']              = ichi.ichimoku_b()

        will= CustomWilliamsRIndicator(data['high'], data['low'], data['close'],14)
        data[f'{prefix}Williams_%R']= will.williams_r()

        vroc= CustomVolumeRateOfChangeIndicator(data['volume'],20)
        data[f'{prefix}VROC']= vroc.volume_rate_of_change()

        piv= CustomPivotPointIndicator(data['high'], data['low'], data['close'],5)
        data[f'{prefix}Pivot']= piv.pivot()
        data[f'{prefix}Support_1']= piv.support_1()
        data[f'{prefix}Support_2']= piv.support_2()
        data[f'{prefix}Resistance_1']= piv.resistance_1()
        data[f'{prefix}Resistance_2']= piv.resistance_2()

        candle= CustomCandlestickPattern(data['open'], data['high'], data['low'], data['close'])
        data[f'{prefix}Engulfing']= candle.engulfing()
        data[f'{prefix}Doji']= candle.doji()

        # Creative
        ha_close= (data['open']+data['high']+data['low']+data['close'])/4
        ha_open= ha_close.shift(1)
        ha_open.fillna(method='bfill', inplace=True)
        data[f'{prefix}HeikinAshi_Open']= ha_open
        data[f'{prefix}HeikinAshi_Close']= ha_close

        data[f'{prefix}Range_Close_Ratio']= (data['high']-data['low'])/(data['close']+1e-9)
        data[f'{prefix}Bull_Power']= (data['close']-data['low'])/((data['high']-data['low'])+1e-9)

        def last_local_max_idx(series):
            return len(series)- np.argmax(series.values)-1
        def last_local_min_idx(series):
            return len(series)- np.argmin(series.values)-1

        data[f'{prefix}BarsFromLocalMax_20']= data['close'].rolling(20).apply(last_local_max_idx, raw=False)
        data[f'{prefix}BarsFromLocalMin_20']= data['close'].rolling(20).apply(last_local_min_idx, raw=False)

        data[f'{prefix}RSI_MACD']= data[f'{prefix}RSI_14']* data[f'{prefix}MACD']
        data[f'{prefix}MA20_MA50_Ratio']= data[f'{prefix}ma20']/(data[f'{prefix}ma50']+1e-9)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(axis=0, how='all', inplace=True)

        self._convert_timedelta_to_seconds(data)

        # logging.info(f"[load_and_process_timeframe] {timeframe_label} => final shape={data.shape}")
        return data

    def load_data(self) -> pd.DataFrame:
        logging.info("[load_data] Start loading all timeframes.")
        main_df = self.load_and_process_timeframe(
            self.main_timeframe, self.filepaths[self.main_timeframe]
        )
        main_df.set_index(f"{self.main_timeframe}_time", inplace=True, drop=False)
        # logging.debug(f"[load_data] main_df shape after main timeframe load => {main_df.shape}")

        for tf_label, fp in self.filepaths.items():
            if tf_label == self.main_timeframe:
                continue
            df = self.load_and_process_timeframe(tf_label, fp)
            df.set_index(f"{tf_label}_time", inplace=True, drop=False)
            # logging.debug(f"[load_data] shape before join => main_df:{main_df.shape}, df:{df.shape}")
            main_df = main_df.join(df, how='outer', rsuffix=f"_{tf_label}")
            # logging.debug(f"[load_data] shape after join {tf_label} => {main_df.shape}")

        main_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        main_df.fillna(method='ffill', inplace=True)
        main_df.fillna(method='bfill', inplace=True)
        main_df.dropna(axis=0, how='all', inplace=True)

        self._convert_timedelta_to_seconds(main_df)

        print("All timeframes loaded and merged successfully.")
        logging.info(f"[load_data] All timeframes loaded and merged => shape={main_df.shape}")
        return main_df

    def select_features(self, X: pd.DataFrame, y: pd.Series, top_k=300) -> list:
        """
        انتخاب فیچر با VarThresh, CorrThresh, Mutual Info
        (بدون حذف کلی، صرفاً محدود کردن تعداد فیچرها به top_k=300).
        """
        logging.info(f"[select_features] Start VarThreshold => X.shape={X.shape}, y.shape={y.shape}")
        vt= VarianceThreshold(0.01)
        vt.fit(X)
        X_var= X[X.columns[vt.get_support()]]
        logging.debug(f"[select_features] after VarThresh => shape={X_var.shape}")
        print(f"Features after Variance Threshold: {len(X_var.columns)}")

        corrm= X_var.corr().abs()
        upper= corrm.where(np.triu(np.ones(corrm.shape), k=1).astype(bool))
        to_drop= [col for col in upper.columns if any(upper[col]>0.9)]
        X_corr= X_var.drop(columns=to_drop, errors='ignore')
        logging.debug(f"[select_features] after CorrThresh => shape={X_corr.shape}")
        print(f"Features after Correlation Threshold: {len(X_corr.columns)}")

        scaler= MinMaxScaler()
        Xs= scaler.fit_transform(X_corr)
        mi= mutual_info_classif(Xs, y, discrete_features='auto')
        mi_series= pd.Series(mi, index=X_corr.columns).sort_values(ascending=False)
        top_feats= mi_series.head(top_k).index.tolist()
        logging.debug(f"[select_features] after MutualInfo => top_k={len(top_feats)} selected")
        print(f"Features after Mutual Info: {len(top_feats)}")

        return top_feats

    def ready(self, data: pd.DataFrame, window: int=1, selected_features: list=None):
        """
        ساخت تارگت (باینری) براساس اختلاف قیمت close،
        diff() روی فیچرها،
        reset_index جهت جلوگیری از خطا،
        select_features (درصورت نبود selected_features)،
        windowing (درصورت نیاز).
        هیچ فیچری حذف نمی‌شود مگر در مرحله select_features به‌خاطر Variance/Corr/MI.
        """
        logging.info("Data ready is begining ...")
        main_close= f"{self.main_timeframe}_close"
        if main_close not in data.columns:
            raise ValueError(f"{main_close} not found in data columns.")

        # ساخت تارگت
        target_close= data[main_close].copy()
        # logging.debug(f"[ready] target_close.shape={target_close.shape}, first idxs={target_close.index[:5]}")
        target= ((target_close.shift(-1) - target_close)>0).astype(int)
        data= data.iloc[:-1].copy()
        target= target.iloc[:-1].copy()
        # logging.debug(f"[ready] after shift => data.shape={data.shape}, target.shape={target.shape}")
        # logging.debug(f"[ready] data.index[:5]={data.index[:5]}, target.index[:5]={target.index[:5]}")

        # جداکردن ستون‌های زمانی
        time_cols= [c for c in data.columns if 'Hour' in c or 'DayOfWeek' in c or 'IsWeekend' in c]
        feat_cols= [c for c in data.columns if c not in time_cols+[main_close]]
        # logging.debug(f"[ready] time_cols => {time_cols[:5]}..., feat_cols => {len(feat_cols)} cols")

        # diff
        df_diff= data[feat_cols].diff()
        # logging.debug(f"[ready] df_diff after diff => shape={df_diff.shape}, index[:5]={df_diff.index[:5]}")
        df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_diff.fillna(method='ffill', inplace=True)
        df_diff.fillna(method='bfill', inplace=True)
        df_diff.dropna(axis=0, how='all', inplace=True)
        # logging.debug(f"[ready] df_diff after dropna => shape={df_diff.shape}, index[:5]={df_diff.index[:5]}")

        # ریست ایندکس => ۰..N-1
        # logging.debug(f"[ready] df_diff.index before reset => {df_diff.index[:10]}")
        df_diff.reset_index(drop=True, inplace=True)
        # logging.debug(f"[ready] df_diff.index after reset => {df_diff.index[:10]}, shape={df_diff.shape}")

        # هماهنگ‌سازی target با df_diff
        # logging.debug(f"[ready] target.shape={target.shape}, target.index[:5]={target.index[:5]}")
        target.reset_index(drop=True, inplace=True)
        # logging.debug(f"[ready] target.index after reset => {target.index[:10]}")
        # اگر size یکسان نباشد، می‌توان درصورت نیاز slice کرد
        if len(target) > len(df_diff):
            # logging.warning(f"[ready] target bigger => slicing target to len(df_diff)={len(df_diff)}")
            target = target.iloc[:len(df_diff)].copy()
        elif len(df_diff) > len(target):
            logging.warning(f"[ready] df_diff bigger => slicing df_diff to len(target)={len(target)}")
            df_diff = df_diff.iloc[:len(target)].copy()

        # logging.debug(f"[ready] after align => df_diff.shape={df_diff.shape}, target.shape={target.shape}")
        # logging.debug(f"[ready] df_diff.index[:5]={df_diff.index[:5]}, target.index[:5]={target.index[:5]}")

        self._convert_timedelta_to_seconds(df_diff)

        # انتخاب فیچر
        if selected_features is None:
            # logging.debug(f"[ready] calling select_features => df_diff.shape={df_diff.shape}, target.shape={target.shape}")
            feats= self.select_features(df_diff, target, top_k=300)
        else:
            feats= [f for f in selected_features if f in df_diff.columns]
            # logging.debug(f"[ready] custom feats => {len(feats)} features found in df_diff")

        X_filtered= df_diff[feats].copy()
        if X_filtered.empty:
            print("No features => empty.")
            logging.warning("[ready] X_filtered empty => returning empty")
            return pd.DataFrame(), pd.Series(dtype=int), feats

        # logging.debug(f"[ready] X_filtered => shape={X_filtered.shape}, feats[:10]={feats[:10]}")

        if window<1:
            window=1

        if window==1:
            X_final= X_filtered
            y_final= target
            # logging.info(f"[ready] window=1 => no windowing, X_final.shape={X_final.shape}, y_final.shape={y_final.shape}")
        else:
            length= len(X_filtered)
            if length< window:
                print("Not enough rows for window.")
                logging.warning(f"[ready] length({length})<window({window}) => returning empty")
                return pd.DataFrame(), pd.Series(dtype=int), feats

            arr_list=[]
            new_idx=[]
            for i in range(window-1, length):
                row_val=[]
                for off in range(window):
                    idx= i-off
                    row_val.extend(X_filtered.iloc[idx].values)
                arr_list.append(row_val)
                new_idx.append(i)

            X_win= pd.DataFrame(arr_list, index=new_idx)
            # logging.debug(f"[ready] after window => X_win.shape={X_win.shape}, index[:5]={X_win.index[:5]}")
            if X_win.shape[1]>0:
                colnames=[]
                for off in range(window):
                    stp= f"_tminus{off}"
                    for c in X_filtered.columns:
                        colnames.append(f"{c}{stp}")
                if len(colnames)== X_win.shape[1]:
                    X_win.columns= colnames

            y_win= target.loc[X_win.index].copy()

            X_win.reset_index(drop=True, inplace=True)
            y_win.reset_index(drop=True, inplace=True)

            X_win.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_win.fillna(method='ffill', inplace=True)
            X_win.fillna(method='bfill', inplace=True)
            X_win.dropna(axis=0, how='all', inplace=True)

            y_win= y_win.loc[X_win.index].copy()
            y_win.reset_index(drop=True, inplace=True)

            self._convert_timedelta_to_seconds(X_win)

            X_final= X_win
            y_final= y_win
            # logging.info(f"[ready] window={window} => X_final.shape={X_final.shape}, y_final.shape={y_final.shape}")

        # در پایان، ریست ایندکس نهایی
        X_final.reset_index(drop=True, inplace=True)
        y_final.reset_index(drop=True, inplace=True)
        # logging.info(f"[ready] final dataset => X_final.shape={X_final.shape}, y_final.shape={y_final.shape}")
        logging.info("Data ready is finish and return X_final , y_final , feats")
        return X_final, y_final, feats

    def get_prepared_data(self, window: int=1):
        logging.info(f"[get_prepared_data] Called with window={window}")
        merged_df= self.load_data()
        X_fin, y_fin, feats= self.ready(merged_df, window=window)
        logging.info(f"[get_prepared_data] Done => X_fin.shape={X_fin.shape}, y_fin.shape={y_fin.shape}, feats={len(feats)}")
        return X_fin, y_fin, feats
