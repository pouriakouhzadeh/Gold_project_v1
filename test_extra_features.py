import pandas as pd
from extra_features_prototype import add_extra_features

df30 = pd.read_csv("XAUUSD_M30.csv", parse_dates=["time"]).sort_values("time")
df30 = df30.reset_index(drop=True)

df30_extra = add_extra_features(df30, prefix="30T_")
df30_extra.to_csv("XAUUSD_M30_with_extra_features.csv", index=False)
