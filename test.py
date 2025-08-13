import joblib
from prepare_data_for_train import PREPARE_DATA_FOR_TRAIN

payload = joblib.load("best_model.pkl")
feat_cols = payload["train_window_cols"]
win = payload["window_size"]

prep = PREPARE_DATA_FOR_TRAIN(
    main_timeframe="30T",
    filepaths={
        "30T":"XAUUSD_M30.csv",
        "15T":"XAUUSD_M15.csv",
        "5T":"XAUUSD_M5.csv",
        "1H":"XAUUSD_H1.csv",
    },
    verbose=False,
    fast_mode=False   # همان چیزی که زمان TRAIN استفاده شده
)

merged = prep.load_data()
Xtr, ytr, _, _ = prep.ready(merged, window=win, selected_features=feat_cols, mode="train")
missing = [c for c in feat_cols if c not in Xtr.columns]
print("missing:", len(missing), missing[:5])
