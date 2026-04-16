import pandas as pd
import numpy as np
import os

# --- Configuration ---
# IMPORTANT: This DATA_DIR should point to where your *preprocessed* individual stock CSVs are stored.
# These CSVs are assumed to have 'Date' as index, and 'Close', 'Log Return' columns.
# This is the output that `analyze_dow_data.py` might generate if it saves processed data per ticker.
PREPROCESSED_DATA_DIR = "dow_data_preprocessed" # Example: "dow_data_preprocessed_close_only" from your analyze_dow_data.py
OUTPUT_FACTOR_CSV = "djia_weekly_factors_v2.csv"

# DJIA Tickers (ensure this list is consistent with your other scripts)
DJIA_TICKER_INDEX = "^DJI" # Used for getting overall trading calendar
DJIA_COMPONENTS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]

# Define factor calculation windows (in trading days)
FACTOR_WINDOWS = {
    "MOM_1M": 21, "RS_1M": 21, 
    "MOM_3M": 63, "RS_3M": 63, "Sharpe_3M": 63, "VOL_3M_STD": 63,
    "MOM_6M": 126, "RS_6M": 126, "Sharpe_6M": 126,
    "MOM_12M": 252, "RS_12M": 252,
    "VOL_1M_STD": 21,
    "SMA50": 50,
    "SMA200": 200
}

# --- Helper Functions ---

def load_single_stock_data(ticker, data_dir):
    """Loads preprocessed data for a single stock."""
    file_path_option1 = os.path.join(data_dir, f"{ticker.replace('^', 'INDEX_')}_preprocessed.csv")
    file_path_option2 = os.path.join(data_dir, f"{ticker.replace('^', 'INDEX_')}.csv")
    
    file_path_to_try = None
    if os.path.exists(file_path_option1):
        file_path_to_try = file_path_option1
    elif os.path.exists(file_path_option2) and data_dir == "dow_data":
        print(f"Warning: Using raw downloaded data for {ticker} as preprocessed not found. Ensure it is suitable.")
        try:
            df = pd.read_csv(file_path_option2, index_col='Date', parse_dates=True)
            if 'Adj Close' in df.columns and 'Close' not in df.columns: df['Close'] = df['Adj Close']
            if 'Close' in df.columns:
                df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
                df.dropna(subset=['Log Return'], inplace=True)
            return df
        except Exception as e:
            print(f"Error loading raw data for {ticker} from {file_path_option2}: {e}")
            return None
    elif not os.path.exists(file_path_option1) and not os.path.exists(file_path_option2):
        print(f"Warning: Data file not found for {ticker} at {file_path_option1} or {file_path_option2}. Skipping.")
        return None
    
    if file_path_to_try is None:
        print(f"Final Warning: No valid data path found for {ticker} in specified directories. Skipping.")
        return None
        
    try:
        df = pd.read_csv(file_path_to_try, index_col='Date', parse_dates=True)
        if 'Close' not in df.columns or 'Log Return' not in df.columns:
            print(f"Warning: 'Close' or 'Log Return' not in {file_path_to_try} for {ticker}. Attempting to derive.")
            if 'Adj Close' in df.columns and 'Close' not in df.columns: df['Close'] = df['Adj Close']
            if 'Close' in df.columns and 'Log Return' not in df.columns:
                df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
                df.dropna(subset=['Log Return'], inplace=True)
            if 'Close' not in df.columns or 'Log Return' not in df.columns:
                print(f"Error: Still missing critical columns for {ticker} after attempting to derive. Skipping.")
                return None
        return df
    except Exception as e:
        print(f"Error loading data for {ticker} from {file_path_to_try}: {e}")
        return None

def get_weekly_rebalance_dates(trading_days_index):
    """Identifies the last trading day of each week from a DatetimeIndex."""
    if not isinstance(trading_days_index, pd.DatetimeIndex):
        raise ValueError("trading_days_index must be a pandas DatetimeIndex.")
    return trading_days_index.to_series().groupby([trading_days_index.year, trading_days_index.isocalendar().week]).tail(1).index

def calculate_momentum(price_series, window):
    """Calculates log momentum over a given window of trading days."""
    if len(price_series) < window + 1:
        return np.nan
    current_price = price_series.iloc[-1]
    prior_price = price_series.iloc[-(window + 1)]
    if prior_price <= 0:
        return np.nan
    return np.log(current_price / prior_price)

def calculate_volatility(log_return_series, window):
    """Calculates standard deviation of log returns over a window."""
    if len(log_return_series) < window:
        return np.nan
    return log_return_series.iloc[-window:].std()

def calculate_sma(price_series, window):
    """Calculates Simple Moving Average."""
    if len(price_series) < window:
        return np.nan
    return price_series.iloc[-window:].mean()

def calculate_sharpe(log_return_series, window, periods_per_year=252):
    """Calculates annualized Sharpe ratio (simplified, Rf=0) over a window."""
    if len(log_return_series) < window:
        return np.nan
    # Select returns for the window
    window_returns = log_return_series.iloc[-window:]
    if window_returns.empty:
        return np.nan
        
    mean_ret = window_returns.mean()
    std_ret = window_returns.std()
    
    # Handle zero or negligible standard deviation
    if pd.isna(std_ret) or std_ret < 1e-8: 
        # If std is zero and mean is non-zero, Sharpe is +/- inf. If mean is also zero, Sharpe is undefined (NaN or 0).
        # Let's return 0 for stability, implying no risk-adjusted return.
        return 0.0 
    
    # Annualize mean and std dev
    # annualized_mean = mean_ret * periods_per_year 
    # annualized_std = std_ret * np.sqrt(periods_per_year)
    # sharpe = annualized_mean / annualized_std
    # Simplified: (mean/std) * sqrt(periods_per_year)
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    return sharpe

# --- Main Factor Calculation Script ---
def main():
    print("--- Starting Weekly Factor Calculation (V2: Added RS, Sharpe) ---")

    # 1. Load data for all constituents AND the index
    all_stock_data = {}
    print(f"Loading preprocessed stock data from: {PREPROCESSED_DATA_DIR}")
    if not os.path.isdir(PREPROCESSED_DATA_DIR):
        print(f"Error: Preprocessed data directory not found: {PREPROCESSED_DATA_DIR}")
        if PREPROCESSED_DATA_DIR != "dow_data" and os.path.isdir("dow_data"):
            print(f"Attempting fallback to load from: dow_data...")
            current_data_dir = "dow_data"
        else:
            return
    else:
        current_data_dir = PREPROCESSED_DATA_DIR

    for ticker in DJIA_COMPONENTS + [DJIA_TICKER_INDEX]: 
        df = load_single_stock_data(ticker, current_data_dir)
        if df is not None and not df.empty:
            all_stock_data[ticker] = df
        else:
            print(f"Warning: Could not load or data empty for {ticker}.")

    if DJIA_TICKER_INDEX not in all_stock_data:
        print(f"Error: DJIA index ({DJIA_TICKER_INDEX}) data not found. Needed for RS calculation. Exiting.")
        return
    if not any(ticker in all_stock_data for ticker in DJIA_COMPONENTS):
        print("Error: No data loaded for any DJIA constituent stock. Exiting.")
        return

    print(f"Loaded data for {len(all_stock_data)-1} constituents and the DJIA index.")

    # 2. Get rebalance dates
    trading_calendar = all_stock_data[DJIA_TICKER_INDEX].index
    rebalance_dates = get_weekly_rebalance_dates(trading_calendar)
    if rebalance_dates.empty:
        print("Error: No rebalance dates could be determined.")
        return
    print(f"Identified {len(rebalance_dates)} weekly rebalance dates, from {rebalance_dates.min().strftime('%Y-%m-%d')} to {rebalance_dates.max().strftime('%Y-%m-%d')}.")

    # 3. Calculate factors for each stock at each rebalance date
    all_factors_list = []
    dji_data_full = all_stock_data[DJIA_TICKER_INDEX] # Get DJIA data once

    for rb_date in rebalance_dates:
        print(f"Processing rebalance date: {rb_date.strftime('%Y-%m-%d')}...")
        # Get DJIA data up to rebalance date for RS calculation
        dji_data_at_rb = dji_data_full[dji_data_full.index <= rb_date]
        
        for ticker in DJIA_COMPONENTS:
            if ticker not in all_stock_data:
                continue
            
            stock_df = all_stock_data[ticker]
            data_for_stock_at_rb = stock_df[stock_df.index <= rb_date]
            
            if data_for_stock_at_rb.empty:
                continue

            current_factors = {"Ticker": ticker, "Date": rb_date}

            # --- Original Momentum & Volatility --- 
            for mom_label, window in [("MOM_1M", FACTOR_WINDOWS["MOM_1M"]), 
                                      ("MOM_3M", FACTOR_WINDOWS["MOM_3M"]),
                                      ("MOM_6M", FACTOR_WINDOWS["MOM_6M"]), 
                                      ("MOM_12M", FACTOR_WINDOWS["MOM_12M"])]:
                current_factors[mom_label] = calculate_momentum(data_for_stock_at_rb['Close'], window)

            for vol_label, window in [("VOL_1M_STD", FACTOR_WINDOWS["VOL_1M_STD"]),
                                      ("VOL_3M_STD", FACTOR_WINDOWS["VOL_3M_STD"])]:
                current_factors[vol_label] = calculate_volatility(data_for_stock_at_rb['Log Return'], window)

            # --- ADDED: Relative Strength (RS) --- 
            for rs_label, window in [("RS_1M", FACTOR_WINDOWS["RS_1M"]), 
                                     ("RS_3M", FACTOR_WINDOWS["RS_3M"]),
                                     ("RS_6M", FACTOR_WINDOWS["RS_6M"]), 
                                     ("RS_12M", FACTOR_WINDOWS["RS_12M"])]:
                stock_mom = calculate_momentum(data_for_stock_at_rb['Close'], window)
                dji_mom = calculate_momentum(dji_data_at_rb['Close'], window)
                # RS = Stock Momentum - Index Momentum
                if pd.isna(stock_mom) or pd.isna(dji_mom):
                    current_factors[rs_label] = np.nan
                else:
                    current_factors[rs_label] = stock_mom - dji_mom
            
            # --- ADDED: Sharpe Ratio (Simplified) --- 
            for sharpe_label, window in [("Sharpe_3M", FACTOR_WINDOWS["Sharpe_3M"]),
                                         ("Sharpe_6M", FACTOR_WINDOWS["Sharpe_6M"])]:
                current_factors[sharpe_label] = calculate_sharpe(data_for_stock_at_rb['Log Return'], window)

            # --- Original SMA and Price-to-SMA --- 
            current_price = data_for_stock_at_rb['Close'].iloc[-1] if not data_for_stock_at_rb['Close'].empty else np.nan
            sma50 = calculate_sma(data_for_stock_at_rb['Close'], FACTOR_WINDOWS["SMA50"])
            sma200 = calculate_sma(data_for_stock_at_rb['Close'], FACTOR_WINDOWS["SMA200"])
            # current_factors["SMA50"] = sma50 # Keep SMA values if needed
            # current_factors["SMA200"] = sma200
            current_factors["PriceToSMA50"] = current_price / sma50 if sma50 and not pd.isna(sma50) and sma50 != 0 else np.nan
            current_factors["PriceToSMA200"] = current_price / sma200 if sma200 and not pd.isna(sma200) and sma200 != 0 else np.nan
            
            all_factors_list.append(current_factors)
        
    if not all_factors_list:
        print("No factors were calculated. Check data availability and lookback periods.")
        return

    # 4. Combine into a master DataFrame
    master_factor_df = pd.DataFrame(all_factors_list)
    master_factor_df.set_index(['Date', 'Ticker'], inplace=True)
    
    print("\n--- Factor Calculation Complete (V2) ---")
    if not master_factor_df.empty:
        print("Master Factor DataFrame (V2 - first 5 rows):")
        print(master_factor_df.head())
        print("\nMaster Factor DataFrame (V2 - last 5 rows):")
        print(master_factor_df.tail())
        print(f"\nShape of master factor DataFrame: {master_factor_df.shape}")
        print("\nColumns in factor DataFrame:", master_factor_df.columns.tolist())
    else:
        print("Master Factor DataFrame is empty.")

    # 5. Save to CSV
    try:
        master_factor_df.to_csv(OUTPUT_FACTOR_CSV)
        print(f"\nSuccessfully saved weekly factors (V2) to: {OUTPUT_FACTOR_CSV}")
    except Exception as e:
        print(f"\nError saving factors (V2) to CSV: {e}")

if __name__ == "__main__":
    main() 