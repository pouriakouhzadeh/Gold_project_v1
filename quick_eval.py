# quick_eval.py
import joblib, pandas as pd, numpy as np
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN
from sklearn.metrics import accuracy_score, f1_score

model = joblib.load("best_model.pkl")
window_size = model['window_size']
cols        = model['train_window_cols']

prep = PREPARE_DATA_FOR_TRAIN(main_timeframe="30T")
data = prep.load_data()

# تست دقیقاً همان پریود شبیه‌سازی
X_all, y_all, _ = prep.ready(data, window=window_size,
                             selected_features=model['feats'], mode='train')
X_all = X_all[cols].astype(float)

proba = model['pipeline'].predict_proba(X_all)[:,1]
y_pred = np.full_like(y_all, -1)
y_pred[proba <= model['neg_thr']] = 0
y_pred[proba >= model['pos_thr']] = 1
mask = y_pred != -1

print("Conf-ratio:", mask.mean())
print("F1:", f1_score(y_all[mask], y_pred[mask]))
print("Acc:", accuracy_score(y_all[mask], y_pred[mask]))
