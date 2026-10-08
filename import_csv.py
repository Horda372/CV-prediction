import os
import csv
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from config import CSVConfig

def calculate_indicators(df, config):
    df = df.copy()
    
    for ma in config.moving_averages:
        df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()
        
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = df[['High', 'Low', 'Close']].copy()
    tr['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr['TR'].rolling(window=config.atr_period).mean()
    
    df.dropna(inplace=True)
    return df

def generate_dataset_from_csv(config: CSVConfig):
    images_dir = config.img_dir
    tensors_dir = config.tensor_dir
    
    for d in [images_dir, tensors_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    metadata_file = config.metadata_file
    print(f"Dataset directory at: {os.path.abspath(config.output_dir)}")
    
    processed_files = set()
    file_exists = os.path.exists(metadata_file)
    
    if file_exists:
        try:
            with open(metadata_file, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row: processed_files.add(row[0])
            print(f"Found existing metadata with {len(processed_files)} records. Resuming...")
        except Exception:
            pass
            
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True, volume='in')
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='', figcolor='black', facecolor='black')

    class_0_count = 0
    class_1_count = 0

    with open(metadata_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(metadata_file) == 0:
            writer.writerow([
                "filename", "tensor_filename", "symbol", "timeframe", "window_size", 
                "start_date", "end_date", "trade_exit_date", "start_close_price", "end_close_price",
                "target_label", "exit_type", "exit_price", "pct_change"
            ])

        for symbol in config.symbols:
            csv_filename = f"{symbol}_{config.timeframe}.csv"
            csv_filepath = os.path.join(config.csv_dir, csv_filename)
            
            if not os.path.exists(csv_filepath):
                print(f"Warning: CSV file not found at {csv_filepath}. Skipping symbol {symbol}.")
                continue
                
            print(f"\nProcessing: {symbol}...")
            try:
                df = pd.read_csv(csv_filepath)
                
                date_col = None
                for col in df.columns:
                    if col.lower() in ["date", "time", "timestamp"]:
                        date_col = col
                        break
                        
                if date_col is None:
                    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
                    date_col = "Date"
                    
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
                df.sort_index(ascending=True, inplace=True)
                
                column_mapping = {col: col.capitalize() for col in df.columns if col.lower() in ["open", "high", "low", "close", "volume"]}
                df.rename(columns=column_mapping, inplace=True)
                
                if "Volume" not in df.columns:
                    df["Volume"] = 1000

                df = calculate_indicators(df, config)
                
                total_rows = len(df)
                available_samples = total_rows - config.window_size - config.max_holding_period
                
                if available_samples <= 0:
                    continue

                print(f"Generating {available_samples} sliding window samples...")
                generated_count = 0
                
                for i in range(available_samples):
                    start_idx = i
                    end_idx = i + config.window_size
                    
                    window_data = df.iloc[start_idx:end_idx].copy()
                    
                    current_close = window_data['Close'].iloc[-1]
                    current_atr = window_data['ATR'].iloc[-1]
                    
                    spread = current_close * getattr(config, "spread_pct", 0.00015)
                    entry_price = current_close + spread
                    
                    tp_price = entry_price + (current_atr * config.tp_atr_multiplier)
                    sl_price = entry_price - (current_atr * config.sl_atr_multiplier)
                    
                    future_data = df.iloc[end_idx : end_idx + config.max_holding_period]
                    
                    target_label = 0
                    exit_type = "timeout"
                    exit_price = future_data['Close'].iloc[-1]
                    trade_exit_date = future_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Lookahead evaluation
                    skip_sample = False
                    days_held = 0
                    for timestamp, row in future_data.iterrows():
                        days_held += 1
                        hit_sl = row['Low'] <= sl_price
                        hit_tp = row['High'] >= tp_price
                        
                        if hit_sl and hit_tp:
                            # Intrabar ambiguity: skip sample to prevent lookahead bias
                            skip_sample = True
                            break
                        elif hit_sl:
                            target_label = 0
                            exit_type = "SL"
                            exit_price = sl_price
                            trade_exit_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            break
                        elif hit_tp:
                            target_label = 1
                            exit_type = "TP"
                            exit_price = tp_price
                            trade_exit_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            break
                            
                    if skip_sample:
                        continue

                    swap_cost_pct = days_held * getattr(config, "daily_swap_pct", 0.00005)
                    pct_change = ((exit_price - entry_price) / entry_price) * 100.0 - (swap_cost_pct * 100.0)

                    if target_label == 1:
                        class_1_count += 1
                    else:
                        class_0_count += 1

                    start_date = window_data.index[0].strftime("%Y-%m-%d %H:%M:%S")
                    end_date = window_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")

                    close_mean = window_data['Close'].mean()
                    close_std = window_data['Close'].std() + 1e-8
                    
                    tensor_data = pd.DataFrame(index=window_data.index)
                    cols_price = ['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_20']
                    tensor_data[cols_price] = (window_data[cols_price] - close_mean) / close_std
                        
                    tensor_data['Volume'] = (window_data['Volume'] - window_data['Volume'].mean()) / (window_data['Volume'].std() + 1e-8)
                    tensor_data['RSI'] = window_data['RSI'] / 100.0
                    tensor_data['MACD'] = window_data['MACD'] / close_std
                    tensor_data['MACD_Signal'] = window_data['MACD_Signal'] / close_std
                    tensor_data['ATR_pct'] = window_data['ATR'] / close_mean
                    tensor_data = tensor_data.fillna(0)

                    base_filename = f"{symbol}_{config.timeframe}_w{config.window_size}_{i:05d}"
                    img_filename = f"{base_filename}.png"
                    tensor_filename = f"{base_filename}.npy"
                    
                    tensor_filepath = os.path.join(tensors_dir, tensor_filename)
                    np.save(tensor_filepath, tensor_data.values.astype(np.float32))
                    
                    if img_filename in processed_files:
                        continue

                    p_min = window_data['Low'].min()
                    p_max = window_data['High'].max()
                    if p_max > p_min:
                        window_data[cols_price] = (window_data[cols_price] - p_min) / (p_max - p_min)
                    else:
                        window_data[cols_price] = 0.5
                            
                    v_min, v_max = window_data['Volume'].min(), window_data['Volume'].max()
                    if v_max > v_min:
                        window_data['Volume'] = (window_data['Volume'] - v_min) / (v_max - v_min)
                    else:
                        window_data['Volume'] = 0.5
                        
                    window_data['RSI'] = window_data['RSI'] / 100.0
                    
                    macd_cols = ['MACD', 'MACD_Signal', 'MACD_Hist']
                    m_min = window_data[macd_cols].min().min()
                    m_max = window_data[macd_cols].max().max()
                    if m_max > m_min:
                        window_data[macd_cols] = (window_data[macd_cols] - m_min) / (m_max - m_min)
                    else:
                        window_data[macd_cols] = 0.5

                    apds = []
                    if config.moving_averages:
                        for ma in config.moving_averages:
                            apds.append(mpf.make_addplot(window_data[f'SMA_{ma}'], type='line', width=1.5, panel=0))
                            
                    apds.append(mpf.make_addplot(window_data['RSI'], type='line', width=1.0, color='yellow', panel=2, ylabel='RSI'))
                    apds.append(mpf.make_addplot(window_data['MACD'], type='line', width=1.0, color='cyan', panel=3, ylabel='MACD'))
                    apds.append(mpf.make_addplot(window_data['MACD_Signal'], type='line', width=1.0, color='magenta', panel=3))
                    
                    img_filepath = os.path.join(images_dir, img_filename)
                    
                    plot_kwargs = dict(
                        type='candle', 
                        style=style, 
                        axisoff=True, 
                        volume=True,
                        volume_panel=1,
                        addplot=apds,
                        panel_ratios=(4,1,1,1),
                        savefig=dict(fname=img_filepath, dpi=100, bbox_inches='tight', pad_inches=0),
                        returnfig=True
                    )
                    
                    fig, axlist = mpf.plot(window_data, **plot_kwargs)
                    plt.close(fig)
                    
                    writer.writerow([
                        img_filename, tensor_filename, symbol, config.timeframe, config.window_size, 
                        start_date, end_date, trade_exit_date, round(current_close, 5), round(exit_price, 5),
                        target_label, exit_type, round(exit_price, 5), round(pct_change, 3)
                    ])
                    generated_count += 1
                    
                    if generated_count % 100 == 0:
                        print(f"  Processed {generated_count}/{available_samples} samples...")
                        
                print(f"Successfully generated {generated_count} samples for {symbol}.")
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()

    total_samples = class_0_count + class_1_count
    print(f"\nDataset generation completed successfully.")
    print(f"Total valid samples: {total_samples}")
    if total_samples > 0:
        print(f"  - Class 0 (No-Buy): {class_0_count} ({class_0_count/total_samples*100:.2f}%)")
        print(f"  - Class 1 (Buy):    {class_1_count} ({class_1_count/total_samples*100:.2f}%)")

if __name__ == "__main__":
    app_config = CSVConfig()
    generate_dataset_from_csv(app_config)

