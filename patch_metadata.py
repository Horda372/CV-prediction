import pandas as pd
import numpy as np
import os
from config import CSVConfig

config = CSVConfig()
df_meta = pd.read_csv(config.metadata_file)
if "trade_exit_date" in df_meta.columns:
    print("Already patched!")
    exit(0)

print("Loading original CSVs...")
symbol_data = {}
for symbol in config.symbols:
    csv_filepath = os.path.join(config.csv_dir, f"{symbol}_{config.timeframe}.csv")
    if os.path.exists(csv_filepath):
        df_sym = pd.read_csv(csv_filepath)
        date_col = None
        for col in df_sym.columns:
            if col.lower() in ["date", "time", "timestamp"]: date_col = col; break
        if date_col is None: df_sym.rename(columns={df_sym.columns[0]: "Date"}, inplace=True); date_col = "Date"
        df_sym[date_col] = pd.to_datetime(df_sym[date_col])
        df_sym.set_index(date_col, inplace=True)
        df_sym.sort_index(ascending=True, inplace=True)
        symbol_data[symbol] = df_sym

print("Calculating exit dates...")
trade_exit_dates = []

for idx, row in df_meta.iterrows():
    symbol = row['symbol']
    end_date = pd.to_datetime(row['end_date'])
    exit_type = row['exit_type']
    
    if symbol not in symbol_data:
        trade_exit_dates.append(row['end_date'])
        continue
        
    df_sym = symbol_data[symbol]
    future_data = df_sym.loc[df_sym.index > end_date].head(config.max_holding_period)
    
    if exit_type == 'timeout':
        if len(future_data) > 0:
            trade_exit_dates.append(future_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))
        else:
            trade_exit_dates.append(row['end_date'])
        continue
        
    found = False
    for timestamp, sym_row in future_data.iterrows():
        hit_sl = sym_row['Low'] <= row['exit_price'] if exit_type == 'SL' else False
        hit_tp = sym_row['High'] >= row['exit_price'] if exit_type == 'TP' else False
        
        if hit_sl or hit_tp:
            trade_exit_dates.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            found = True
            break
            
    if not found:
        if len(future_data) > 0:
            trade_exit_dates.append(future_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))
        else:
            trade_exit_dates.append(row['end_date'])

df_meta.insert(7, 'trade_exit_date', trade_exit_dates)
df_meta.to_csv(config.metadata_file, index=False)
print("Metadata patched successfully with trade_exit_date!")
