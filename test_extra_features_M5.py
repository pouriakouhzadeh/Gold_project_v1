import pandas as pd
from extra_features_prototype import add_extra_features

df5 = pd.read_csv("XAUUSD_M5.csv", parse_dates=["time"]).sort_values("time")
df5 = df5.reset_index(drop=True)

df5_extra = add_extra_features(df5, prefix="5T_")
df5_extra.to_csv("XAUUSD_M5_with_extra_features.csv", index=False)
