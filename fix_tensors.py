import os
import pandas as pd
import numpy as np
import csv

def fix_tensors():
    metadata_file = "dataset_market_cv/metadata.csv"
    tensors_dir = "dataset_market_cv/tensors"
    market_data_dir = "market_data"
    
    print("Krytyczna poprawka: Regeneracja tensorów numerycznych...")
    
    df_meta = pd.read_csv(metadata_file)
    
    # Cache market data frames to avoid reloading
    market_dfs = {}
    
    for idx, row in df_meta.iterrows():
        symbol = row['symbol']
        end_date = pd.to_datetime(row['end_date'])
        tensor_filename = row['tensor_filename']
        tensor_filepath = os.path.join(tensors_dir, tensor_filename)
        
        if symbol not in market_dfs:
            csv_filepath = os.path.join(market_data_dir, f"{symbol}_1d.csv")
            m_df = pd.read_csv(csv_filepath)
            
            # Standardize date column
            date_col = None
            for col in m_df.columns:
                if col.lower() in ["date", "time", "timestamp"]:
                    date_col = col
                    break
            if date_col is None:
                m_df.rename(columns={m_df.columns[0]: "Date"}, inplace=True)
                date_col = "Date"
                
            m_df[date_col] = pd.to_datetime(m_df[date_col])
            m_df.set_index(date_col, inplace=True)
            m_df.sort_index(ascending=True, inplace=True)
            
            # Map columns
            col_map = {c: c.capitalize() for c in m_df.columns if c.lower() in ["open", "high", "low", "close", "volume"]}
            m_df.rename(columns=col_map, inplace=True)
            
            # Calculate Indicators
            m_df['SMA_10'] = m_df['Close'].rolling(window=10).mean()
            m_df['SMA_20'] = m_df['Close'].rolling(window=20).mean()
            
            delta = m_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            m_df['RSI'] = 100 - (100 / (1 + rs))
            
            ema_12 = m_df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = m_df['Close'].ewm(span=26, adjust=False).mean()
            m_df['MACD'] = ema_12 - ema_26
            m_df['MACD_Signal'] = m_df['MACD'].ewm(span=9, adjust=False).mean()
            
            high_low = m_df['High'] - m_df['Low']
            high_close = np.abs(m_df['High'] - m_df['Close'].shift())
            low_close = np.abs(m_df['Low'] - m_df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            m_df['ATR'] = true_range.rolling(14).mean()
            
            market_dfs[symbol] = m_df
            
        # Get data window
        m_df = market_dfs[symbol]
        # Find index of end_date
        try:
            end_loc = m_df.index.get_loc(end_date)
            start_loc = end_loc - 20 + 1 # 20 days window
            
            if start_loc < 0:
                continue
                
            window_data = m_df.iloc[start_loc:end_loc+1].copy()
            
            # NEW TENSOR FEATURES: 
            # Replaced raw prices with relative scaling (Z-score like or percentage returns)
            # Adding ATR, adding returns, replacing strict MinMax to preserve volatility magnitude
            
            # Z-Score normalization for prices based on 20-day mean/std to preserve relative volatility
            close_mean = window_data['Close'].mean()
            close_std = window_data['Close'].std() + 1e-8
            
            tensor_data = pd.DataFrame()
            for col in ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_20']:
                tensor_data[col] = (window_data[col] - close_mean) / close_std
                
            tensor_data['Volume'] = (window_data['Volume'] - window_data['Volume'].mean()) / (window_data['Volume'].std() + 1e-8)
            tensor_data['RSI'] = window_data['RSI'] / 100.0
            
            # MACD without MinMax so magnitude is preserved
            tensor_data['MACD'] = window_data['MACD'] / close_std
            tensor_data['MACD_Signal'] = window_data['MACD_Signal'] / close_std
            
            # CRITICAL: Include ATR normalized by price
            tensor_data['ATR_pct'] = window_data['ATR'] / close_mean
            
            # Fill NaNs
            tensor_data = tensor_data.fillna(0)
            
            np.save(tensor_filepath, tensor_data.values.astype(np.float32))
            
        except Exception as e:
            print(f"Error on {symbol} at {end_date}: {e}")
            continue
            
        if idx % 1000 == 0:
            print(f"Re-generated {idx} tensors...")

    print("Zakończono poprawę tensorów!")

if __name__ == "__main__":
    fix_tensors()
