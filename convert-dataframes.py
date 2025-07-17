import pandas as pd

df_30 = pd.read_csv('XAUUSD_M30.csv')
df_60 = pd.read_csv('XAUUSD_H1.csv')
df_15 = pd.read_csv('XAUUSD_M15.csv')
df_5 = pd.read_csv('XAUUSD_M5.csv')

df_30 = df_30[-10000:]
df_60 = df_60[-10000:]
df_15 = df_15[-10000:]
df_5 = df_5[-10000:]

df_30.to_csv('XAUUSD_M30.csv',index=False)
df_60.to_csv('XAUUSD_H1.csv',index=False)
df_15.to_csv('XAUUSD_M15.csv',index=False)
df_5.to_csv('XAUUSD_M5.csv',index=False)




