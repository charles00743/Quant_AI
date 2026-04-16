import yfinance as yf
import pandas as pd
import os
import datetime

# --- Configuration ---
DJIA_TICKER = "^DJI"
OUTPUT_DIR = "dow_data" # Folder to save the CSV files
YEARS_OF_DATA = 5
INTERVAL = "1d" # Data interval (1 day)

# --- DJIA Component Tickers ---
# IMPORTANT: This list is current as of mid-2024 and may change.
# You might need to update this list manually if the index composition changes.
DJIA_COMPONENTS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

# Combine index and components
ALL_TICKERS = [DJIA_TICKER] + DJIA_COMPONENTS

# --- Date Range Calculation ---
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=YEARS_OF_DATA * 365)

# Convert dates to string format suitable for yfinance
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Data will be saved in: {os.path.abspath(OUTPUT_DIR)}")

# --- Data Downloading ---
print("\n--- Starting Data Download --- ")
print(f"Requesting data from {start_date_str} to {end_date_str} with interval {INTERVAL}.")
print("WARNING: Yahoo Finance typically provides limited historical data for intraday intervals (like 15m).")
print(f"You will likely receive data covering a shorter period than the requested {YEARS_OF_DATA} years.")

failed_tickers = []

for i, ticker in enumerate(ALL_TICKERS):
    print(f"\n[{i+1}/{len(ALL_TICKERS)}] Fetching data for: {ticker}...")
    try:
        # Download data for the current ticker
        data = yf.download(ticker, start=start_date_str, end=end_date_str, interval=INTERVAL, progress=False)
        
        if data.empty:
            print(f"Warning: No data returned for {ticker} for the requested period/interval.")
            failed_tickers.append(ticker + " (No data)")
            continue # Skip to the next ticker
            
        # Define the output file path
        output_path = os.path.join(OUTPUT_DIR, f"{ticker.replace('^', 'INDEX_')}.csv") # Replace ^ for filename compatibility
        
        # Save the data to CSV
        data.to_csv(output_path)
        print(f"Successfully saved data for {ticker} to {output_path}")
        
    except Exception as e:
        print(f"Error downloading or saving data for {ticker}: {e}")
        failed_tickers.append(ticker + f" (Error: {e})")

# --- Summary ---
print("\n--- Download Complete ---")
if failed_tickers:
    print("\nFailed to download or process data for the following tickers:")
    for failed in failed_tickers:
        print(f"- {failed}")
else:
    print("\nAll data downloaded and saved successfully (within provider limitations).")

print(f"\nReminder: Check the actual date range covered in the downloaded CSV files for interval '{INTERVAL}'.") 