import pandas as pd
from extra_features_prototype import add_extra_features

df15 = pd.read_csv("XAUUSD_M15.csv", parse_dates=["time"]).sort_values("time")
df15 = df15.reset_index(drop=True)

df15_extra = add_extra_features(df15, prefix="15T_")
df15_extra.to_csv("XAUUSD_M15_with_extra_features.csv", index=False)
