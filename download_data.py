import os
import yfinance as yf
import pandas as pd

def download_forex_data():
    # Folder to save CSVs
    output_dir = "market_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Mapping from Yahoo Finance symbols to clean filenames
    symbol_mapping = {
        "EURUSD=X": "EURUSD_1d.csv",
        "GBPUSD=X": "GBPUSD_1d.csv",
        "USDJPY=X": "USDJPY_1d.csv",
        "AUDUSD=X": "AUDUSD_1d.csv",
        "USDCAD=X": "USDCAD_1d.csv",
        "USDCHF=X": "USDCHF_1d.csv",
        "NZDUSD=X": "NZDUSD_1d.csv",
        "EURGBP=X": "EURGBP_1d.csv",
        "EURJPY=X": "EURJPY_1d.csv",
        "GBPJPY=X": "GBPJPY_1d.csv"
    }
    
    print(f"Downloading 10 years of historical daily price data into: {os.path.abspath(output_dir)}")
    
    for yf_symbol, filename in symbol_mapping.items():
        print(f"Fetching: {yf_symbol}...")
        try:
            # Download daily data for 10 years
            df = yf.download(yf_symbol, period="10y", interval="1d")
            
            if df.empty:
                print(f"Warning: No data returned for {yf_symbol}. Skipping.")
                continue
                
            # Clean and reset index
            df = df.dropna()
            
            # Ensure columns are simple single-indexed standard names
            # yfinance sometimes returns multi-level headers if multiple symbols are passed or depending on versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            # Retain only necessary columns
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[col for col in required_cols if col in df.columns]]
            
            # Save to CSV
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath)
            
            print(f"Successfully saved {len(df)} daily bars of {yf_symbol} to {filepath}")
            
        except Exception as e:
            print(f"Error downloading {yf_symbol}: {e}")
            
    print("\nData download process completed.")

if __name__ == "__main__":
    download_forex_data()
