import pandas as pd
from extra_features_prototype import add_extra_features

dfH1 = pd.read_csv("XAUUSD_H1.csv", parse_dates=["time"]).sort_values("time")
dfH1 = dfH1.reset_index(drop=True)

dfH1_extra = add_extra_features(dfH1, prefix="H1T_")
dfH1_extra.to_csv("XAUUSD_MH1_with_extra_features.csv", index=False)
