import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration / Constants ---
DATA_DIR = "dow_data" # Folder where the downloaded CSVs are stored
DJIA_TICKER = "^DJI"

# IMPORTANT: This list needs to match the one used in download_dow_data.py
# Ensure it's up-to-date if the index composition changed.
DJIA_COMPONENTS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]
ALL_TICKERS = [DJIA_TICKER] + DJIA_COMPONENTS

# --- Helper Function to Load Data ---
def load_stock_data(ticker_symbol, file_name):
    """Loads data for a single ticker from its CSV file, skipping first 3 rows."""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: File not found for {ticker_symbol} at {file_path}. Skipping.")
        return None
    try:
        # Skip first 3 rows, no header in data block, assign column names
        # A=Date, B=Close, C=High, D=Low, E=Open, F=Volume
        df = pd.read_csv(
            file_path,
            skiprows=3,
            header=None,
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
            na_values=['null', 'NaN', ''] # Handle potential null strings
        )
        
        if df.empty:
            print(f"Warning: Dataframe for {ticker_symbol} is empty after loading.")
            return None

        # Set 'Date' column as index and parse it
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except Exception as date_err:
            print(f"Warning: Could not set or parse 'Date' column for {ticker_symbol}. Error: {date_err}")
            # If Date parsing is critical and fails, we might want to return None
            # For now, we'll allow it to proceed and see if other columns are useful
            pass

        # Convert relevant columns to numeric, coercing errors
        cols_to_numeric = ['Close', 'High', 'Low', 'Open', 'Volume']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Expected column '{col}' not found in {ticker_symbol}.")
        
        # Drop rows where 'Close' is NaN as it's crucial for returns
        df.dropna(subset=['Close'], inplace=True)

        if not isinstance(df.index, pd.DatetimeIndex):
             print(f"Warning: Index for {ticker_symbol} is not DatetimeIndex. Type: {type(df.index)}")

        return df
    except Exception as e:
        print(f"Error loading data for {ticker_symbol} from {file_path}: {e}")
        return None

# --- Part 1: Load All Data ---
all_data = {} # Dictionary to hold DataFrames: {'TICKER': DataFrame}
print(f"\nLoading data from '{DATA_DIR}'...")
if not os.path.isdir(DATA_DIR):
     print(f"Error: Data directory '{DATA_DIR}' not found. Please run download_dow_data.py first.")
     exit()

for ticker in ALL_TICKERS:
    file_name = f"{ticker.replace('^', 'INDEX_')}.csv"
    df = load_stock_data(ticker, file_name)
    if df is not None and not df.empty: # Ensure df is not None and not empty
        all_data[ticker] = df
    else:
        print(f"Note: No data loaded for {ticker}.")


print(f"\nLoaded data for {len(all_data)} out of {len(ALL_TICKERS)} requested tickers.")
if len(all_data) == 0:
    print("No data loaded. Exiting.")
    exit()

# --- Part 2: Preprocessing --- 
print("\n--- Starting Preprocessing --- ")

for ticker, df_original in all_data.items():
    # Work on a copy to avoid SettingWithCopyWarning if df is a slice
    df = df_original.copy()
    
    print(f"Processing {ticker}...", end="")
    original_rows = len(df)
    
    # Relevant columns based on your CSV structure (Date is index)
    # We will use 'Close' for returns instead of 'Adj Close'
    cols_to_check = ['Close', 'Volume', 'Open', 'High', 'Low']
    
    # Handle Missing Values (NaN) that might have occurred from to_numeric coercion or were in CSV
    # Fill with ffill then bfill. This is a common approach for price data.
    df.ffill(inplace=True) # Use obj.ffill() directly
    df.bfill(inplace=True) # Use obj.bfill() directly

    # After filling, check if any critical columns are entirely NaN (e.g., if a file was truly empty or malformed)
    if df['Close'].isnull().all():
        print(f"  Warning: 'Close' column for {ticker} is entirely NaN after fill. Skipping further processing for this ticker.")
        all_data[ticker] = None # Or remove from dict: del all_data[ticker]
        continue # Skip to the next ticker

    # Calculate Log Returns based on 'Close' price
    # Ensure 'Close' exists and is numeric
    if 'Close' in df.columns and pd.api.types.is_numeric_dtype(df['Close']):
        df_close_positive = df['Close'][df['Close'] > 0]
        if not df_close_positive.empty:
            # Ensure the index is sorted for correct shift calculation
            if not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)
            df['Log Return'] = np.log(df_close_positive / df_close_positive.shift(1))
        else:
            df['Log Return'] = np.nan
            print(f"  Warning: 'Close' column contains non-positive values in {ticker}, cannot calculate log returns accurately.")
    else:
        print(f"  Warning: 'Close' column not found or not numeric in {ticker} for return calculation.")
        df['Log Return'] = np.nan
        
    # Handle initial NaN in Log Return (first row after shift)
    # Also drop any rows where 'Log Return' became NaN due to issues above
    df.dropna(subset=['Log Return'], inplace=True) 
    
    rows_after = len(df)
    
    if rows_after > 0 :
        print(f" Done. Rows before: {original_rows}, Rows after: {rows_after}.")
    else:
        print(f" Done with issues. Rows before: {original_rows}, Rows after: {rows_after}. Check warnings for {ticker}.")
    
    # Update the dictionary with the processed dataframe
    all_data[ticker] = df 

# Remove tickers that might have become None during preprocessing
all_data = {k: v for k, v in all_data.items() if v is not None and not v.empty}

print("--- Preprocessing Complete ---")
if not all_data:
    print("No data available after preprocessing. Exiting.")
    exit()

# --- Part 3: Descriptive Statistics --- 
print("\n--- Calculating Descriptive Statistics --- ")

all_stats_summary = {}

for ticker, df in all_data.items():
    if df.empty:
        print(f"\n--- Statistics for {ticker}: DataFrame is empty, skipping ---")
        continue

    print(f"\n--- Statistics for {ticker} ---")
    stats = {}
    if 'Close' in df.columns:
        print("\nClose Price Statistics:") # Changed from Adj Close
        close_stats = df['Close'].describe()
        print(close_stats)
        stats['Close'] = close_stats.to_dict()
    else:
        print("\nClose Price Statistics: Not available")
        
    if 'Log Return' in df.columns and not df['Log Return'].empty:
        print("\nLog Return Statistics (based on Close price):")
        log_return_stats = df['Log Return'].describe()
        try:
            log_return_stats['skewness'] = df['Log Return'].skew()
            log_return_stats['kurtosis'] = df['Log Return'].kurtosis()
        except Exception as e:
            print(f"Could not calculate skew/kurtosis for {ticker}: {e}")
            log_return_stats['skewness'] = np.nan
            log_return_stats['kurtosis'] = np.nan
        print(log_return_stats)
        stats['Log Return'] = log_return_stats.to_dict()
    else:
        print("\nLog Return Statistics: Not available or empty")
        
    all_stats_summary[ticker] = stats

# Correlation of Log Returns (based on 'Close' price)
print("\n--- Calculating Log Return Correlation Matrix (based on Close price) --- ")
log_returns_df = pd.concat(
    {ticker: df['Log Return'] for ticker, df in all_data.items() if 'Log Return' in df.columns and not df['Log Return'].empty},
    axis=1
)

if not log_returns_df.empty:
    correlation_matrix = log_returns_df.corr()
    print("\nCorrelation Matrix (partial view - correlation with DJIA):")
    if DJIA_TICKER in correlation_matrix.columns:
        djia_correlations = correlation_matrix[DJIA_TICKER].sort_values(ascending=False)
        print(djia_correlations)
    else:
        print(f"DJIA Index ('{DJIA_TICKER}') not found in the correlation matrix (possibly no log returns for it).")
    
    # print("\nFull Correlation Matrix:")
    # print(correlation_matrix)
else:
    print("Could not generate correlation matrix as no log return data is available.")

print("\n--- Descriptive Statistics Complete ---")

# --- Part 4: Visualization (Placeholder) ---
print("\n--- Generating Visualizations --- ")
if all_data:
    # Example: Plot DJIA 'Close' price if available
    if DJIA_TICKER in all_data and 'Close' in all_data[DJIA_TICKER].columns:
        try:
            plt.figure(figsize=(12,6))
            all_data[DJIA_TICKER]['Close'].plot(title=f'{DJIA_TICKER} Close Price')
            plt.ylabel("Price")
            plt.xlabel("Date")
            plt.grid(True)
            plt.show() # Show plot
            print(f"Plotted {DJIA_TICKER} Close Price.")
        except Exception as e:
            print(f"Could not plot DJIA Close Price: {e}")
    else:
        print(f"DJIA ticker {DJIA_TICKER} or its 'Close' column not found in loaded data for plotting.")

    # Example: Heatmap of the correlation matrix if available
    if 'correlation_matrix' in locals() and not correlation_matrix.empty:
        try:
            plt.figure(figsize=(15,12)) # Adjusted for potentially many tickers
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1) # annot=False for large matrices
            plt.title('Log Return Correlation Matrix (based on Close Price)')
            plt.show() # Show plot
            print("Plotted Correlation Matrix.")
        except Exception as e:
            print(f"Could not plot correlation heatmap: {e}")
    else:
        print("Correlation matrix not available for plotting.")

else:
    print("No data available for visualization.")

# --- Part X+1: Detailed Analysis per Market Regime (ADDED CODE) ---
print("\n--- Detailed Analysis per Market Regime ---")

# Ensure DJIA_TICKER is defined, e.g., DJIA_TICKER = "^DJI"
# Ensure all_data dictionary is populated and contains DJIA_TICKER data.

if DJIA_TICKER not in all_data or all_data[DJIA_TICKER].empty:
    print(f"Error: {DJIA_TICKER} data not found or empty in all_data. Cannot proceed with regime analysis.")
else:
    try:
        features_df_with_regimes = pd.read_csv("regime_features_with_labels.csv", index_col='Date', parse_dates=True)
        if 'Market_Regime' not in features_df_with_regimes.columns:
            print("Error: 'Market_Regime' column not found in regime_features_with_labels.csv. Make sure GMM process ran and saved it.")
            features_df_with_regimes = None # Indicate failure
        elif not isinstance(features_df_with_regimes.index, pd.DatetimeIndex):
            print("Error: Index of regime_features_with_labels.csv is not a DatetimeIndex after loading.")
            features_df_with_regimes = None # Indicate failure

    except FileNotFoundError:
        print("Error: regime_features_with_labels.csv not found. Please ensure GMM part of analyze_dow_data.py has run and saved the file.")
        features_df_with_regimes = None # Indicate failure
    except Exception as e:
        print(f"Error loading regime_features_with_labels.csv: {e}")
        features_df_with_regimes = None # Indicate failure

    if features_df_with_regimes is not None:
        djia_df_original = all_data[DJIA_TICKER] # Get the original DJIA DataFrame
        
        # Ensure djia_df_original has 'Log Return' and its index is DatetimeIndex
        if 'Log Return' not in djia_df_original.columns:
            print(f"Error: 'Log Return' not found for {DJIA_TICKER} in all_data. Preprocessing might have failed for it.")
            features_df_with_regimes = None # Cannot proceed
        elif not isinstance(djia_df_original.index, pd.DatetimeIndex):
            print(f"Error: Index for {DJIA_TICKER} in all_data is not DatetimeIndex.")
            features_df_with_regimes = None # Cannot proceed


    if features_df_with_regimes is not None:
        # Attempt to join regime information with DJIA's log returns and GMM features
        # We need to ensure feature_columns_for_gmm are present in features_df_with_regimes
        
        # IMPORTANT: Define the features that were used to train the GMM model.
        # This is a best guess. PLEASE VERIFY AND UPDATE THIS LIST.
        # It should be the columns from features_df_with_regimes *excluding* 'Market_Regime' and any direct price/return columns if they were not GMM inputs.
        potential_gmm_feature_cols = [col for col in features_df_with_regimes.columns if col not in ['Market_Regime']]
        print(f"Potential GMM input features identified from CSV: {potential_gmm_feature_cols}")
        # ****** USER ACTION REQUIRED: Verify and if necessary, explicitly define gmm_input_features ******
        # Example: gmm_input_features = ['RSI_14', 'MACD_Signal_Diff', 'Volatility_20', 'Some_Other_Feature'] 
        gmm_input_features = potential_gmm_feature_cols # Replace if necessary

        # Select only the GMM features and Market_Regime for joining, to avoid duplicate columns if ^DJI_Close was in features_df_with_regimes
        columns_to_join_from_regime_df = gmm_input_features + ['Market_Regime']
        valid_columns_to_join = [col for col in columns_to_join_from_regime_df if col in features_df_with_regimes.columns]
        
        if 'Market_Regime' not in valid_columns_to_join:
            print("Critical Error: 'Market_Regime' column is missing from selection for join.")
            regime_data_for_analysis = pd.DataFrame() # Empty
        else:
            regime_data_for_analysis = djia_df_original.join(features_df_with_regimes[valid_columns_to_join], how='inner')

        if regime_data_for_analysis.empty or 'Market_Regime' not in regime_data_for_analysis.columns:
            print("Error: Failed to merge DJIA data with regime information, or result is empty. Check indices and column names.")
        else:
            print(f"Successfully merged DJIA data with {len(valid_columns_to_join)-1} features and Market_Regime information.")
            print(f"Columns available for per-regime analysis: {regime_data_for_analysis.columns.tolist()}")
            
            unique_regimes = sorted(regime_data_for_analysis['Market_Regime'].unique())
            regime_definitions = {}

            for regime_id in unique_regimes:
                print(f"\n--- Analyzing Regime {int(regime_id)} ---")
                # Ensure regime_id is a valid type for filtering (usually int or float if GMM output was float)
                try:
                    current_regime_id_typed = type(regime_data_for_analysis['Market_Regime'].iloc[0])(regime_id)
                except ValueError:
                    print(f"Warning: Could not convert regime_id {regime_id} to type of 'Market_Regime' column. Skipping.")
                    continue

                current_regime_data = regime_data_for_analysis[regime_data_for_analysis['Market_Regime'] == current_regime_id_typed].copy() # Use .copy()

                if current_regime_data.empty:
                    print(f"No data found for Regime {int(regime_id)}. Skipping.")
                    continue

                # 1. 日收益率分析
                djia_returns_in_regime = current_regime_data['Log Return'].dropna()
                if not djia_returns_in_regime.empty:
                    print("\nDJIA Log Returns Statistics for this Regime:")
                    print(djia_returns_in_regime.describe())
                    
                    plt.figure(figsize=(10, 5))
                    sns.histplot(djia_returns_in_regime, kde=True, bins=50)
                    plt.title(f'DJIA Log Return Distribution - Regime {int(regime_id)}')
                    plt.xlabel('Log Return')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    plt.show()
                    print(f"Plotted DJIA Log Return Distribution for Regime {int(regime_id)}.")

                    avg_return = djia_returns_in_regime.mean()
                    std_return_as_volatility = djia_returns_in_regime.std()
                    print(f"Average DJIA Log Return in Regime {int(regime_id)}: {avg_return:.6f}")
                    print(f"StdDev of DJIA Log Returns (Proxy for Volatility) in Regime {int(regime_id)}: {std_return_as_volatility:.6f}")
                else:
                    print("No DJIA Log Return data for this regime.")
                    avg_return = np.nan
                    std_return_as_volatility = np.nan
                
                # 2. GMM 输入特征的描述性统计
                # Ensure gmm_input_features are present in current_regime_data
                actual_gmm_features_in_data = [f for f in gmm_input_features if f in current_regime_data.columns]
                if actual_gmm_features_in_data:
                    regime_gmm_features_data = current_regime_data[actual_gmm_features_in_data].copy() # use .copy()
                    if not regime_gmm_features_data.empty:
                        print(f"\nStatistics of GMM Input Features ({len(actual_gmm_features_in_data)} features) for Regime {int(regime_id)}:")
                        desc_stats = regime_gmm_features_data.describe().T
                        try:
                            desc_stats['skew'] = regime_gmm_features_data.skew()
                            desc_stats['kurt'] = regime_gmm_features_data.kurtosis()
                        except Exception as e_skew_kurt:
                            print(f"Could not calculate skew/kurtosis for GMM features in regime {regime_id}: {e_skew_kurt}")
                            desc_stats['skew'] = np.nan
                            desc_stats['kurt'] = np.nan
                        print(desc_stats)
                        avg_gmm_features = regime_gmm_features_data.mean()
                    else:
                        print("No GMM input feature data (after selection) for this regime.")
                        avg_gmm_features = pd.Series(dtype=float)
                else:
                    print("Defined GMM input features not found in the current regime data columns.")
                    avg_gmm_features = pd.Series(dtype=float)

                # 4. 尝试定义市场状态
                definition_parts = []
                if not np.isnan(avg_return):
                    # Example thresholds: Adjust based on your data's scale and typical daily returns
                    if avg_return > 0.0003: 
                        definition_parts.append("Positive Returns (Suggests Bullish Tendency)")
                    elif avg_return < -0.0003:
                        definition_parts.append("Negative Returns (Suggests Bearish Tendency)")
                    else:
                        definition_parts.append("Near-Zero/Mixed Returns")

                if not np.isnan(std_return_as_volatility):
                    # Qualitative assessment of volatility (e.g., low, medium, high) would require
                    # comparison across all regimes' std_return_as_volatility values.
                    # For now, just stating the value.
                    definition_parts.append(f"Log Return StdDev (Volatility Proxy): {std_return_as_volatility:.6f}")
                
                # Add insights from specific GMM features if they are well-understood
                # For example, if 'Volatility_20' (a direct volatility measure) was a GMM feature:
                # if 'Volatility_20' in avg_gmm_features and not pd.isna(avg_gmm_features['Volatility_20']):
                #     vol_feature_val = avg_gmm_features['Volatility_20']
                #     # Add qualitative description based on vol_feature_val
                #     definition_parts.append(f"Avg GMM Volatility Feature ('Volatility_20'): {vol_feature_val:.4f}")

                regime_description = f"Regime {int(regime_id)}: " + "; ".join(definition_parts)
                if not definition_parts:
                     regime_description = f"Regime {int(regime_id)}: Insufficient numeric data for detailed characterization or features not clearly defined."
                
                regime_definitions[int(regime_id)] = regime_description
                print(f"\nProposed Definition for Regime {int(regime_id)}: {regime_description}")

            print("\n--- Summary of Regime Definitions (Preliminary) ---")
            if regime_definitions:
                for regime_id_key, desc in sorted(regime_definitions.items()): # Sort by key for consistent order
                    print(desc)
                print("\nNote: Refine these definitions based on cross-regime comparisons and deeper GMM feature understanding.")
            else:
                print("No regime definitions were generated.")
    # This 'else' corresponds to 'if features_df_with_regimes is not None:'
    else:
        print("Skipping detailed analysis per market regime due to earlier errors in loading or processing regime data.")

# --- End of ADDED CODE for Detailed Analysis per Market Regime ---

print("\n--- Analysis Script Complete ---")

# --- ADDED/MODIFIED: Save Preprocessed Data ---
print("\n--- Saving Preprocessed Data ---")
preprocessed_dir = "dow_data_preprocessed" # Matches PREPROCESSED_DATA_DIR in factor_calculator.py
os.makedirs(preprocessed_dir, exist_ok=True)
saved_count = 0
failed_count = 0

if 'all_data' in locals() and isinstance(all_data, dict):
    for ticker, df in all_data.items():
        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex) and 'Close' in df.columns and 'Log Return' in df.columns:
            try:
                # Ensure consistent naming, replacing ^ for index tickers
                safe_ticker_name = ticker.replace('^', 'INDEX_') 
                output_path = os.path.join(preprocessed_dir, f"{safe_ticker_name}_preprocessed.csv")
                df.to_csv(output_path)
                # print(f"Successfully saved preprocessed data for {ticker} to {output_path}")
                saved_count += 1
            except Exception as e:
                print(f"Error saving preprocessed data for {ticker}: {e}")
                failed_count += 1
        else:
            print(f"Skipping save for {ticker}: Data is None, empty, or missing critical columns/index.")
            failed_count += 1
    print(f"--- Preprocessed Data Saving Complete --- ")
    print(f"Successfully saved: {saved_count} tickers.")
    print(f"Failed to save/skipped: {failed_count} tickers.")
else:
    print("Warning: 'all_data' dictionary not found or not a dictionary. Cannot save preprocessed data.")

# --- Part 5: Feature Engineering for Market Regime (Aggregated from Constituents) ---
print("\n--- Part 5: Engineering Features for Market Regime Analysis --- ")

# Ensure all_data contains the preprocessed data for constituents
# We need 'Log Return' for each constituent stock

# Get a common date index from one of the processed DataFrames (e.g., DJIA or the first available stock)
# This ensures all calculations are aligned if some stocks have slightly different date ranges after preprocessing.
common_index = None
if DJIA_TICKER in all_data and not all_data[DJIA_TICKER].empty:
    common_index = all_data[DJIA_TICKER].index
elif all_data: # If DJIA is not there, try to get index from the first available ticker
    first_available_ticker = next(iter(all_data))
    if all_data[first_available_ticker] is not None and not all_data[first_available_ticker].empty:
        common_index = all_data[first_available_ticker].index

if common_index is None or common_index.empty:
    print("Error: Cannot determine a common date index for regime feature engineering. Exiting Part 5.")
    # Optionally exit script or handle this error
else:
    regime_features_df = pd.DataFrame(index=common_index)
    window_size = 20 # Define the rolling window size

    # --- Feature 1 & 2: Average and Std Dev of constituent average log returns ---
    # Store individual ticker's rolling mean log returns temporarily
    all_ticker_rolling_mean_log_returns = {}
    constituent_tickers = [t for t in DJIA_COMPONENTS if t in all_data and not all_data[t].empty and 'Log Return' in all_data[t].columns]

    if not constituent_tickers:
        print("Warning: No constituent stock data available for regime feature calculation.")
    else:
        for ticker in constituent_tickers:
            df_stock = all_data[ticker]
            # Ensure 'Log Return' is present and not all NaN
            if 'Log Return' in df_stock.columns and not df_stock['Log Return'].isnull().all():
                all_ticker_rolling_mean_log_returns[ticker] = df_stock['Log Return'].rolling(window=window_size, min_periods=int(window_size*0.8)).mean()
            else:
                print(f"Note: 'Log Return' not available or all NaN for {ticker} when calculating rolling mean.")

        # Combine into a single DataFrame for easier calculation of mean and std across tickers
        if all_ticker_rolling_mean_log_returns:
            rolling_mean_returns_panel = pd.DataFrame(all_ticker_rolling_mean_log_returns)
            # Reindex to common_index to ensure alignment, fill missing values that might arise from reindexing
            rolling_mean_returns_panel = rolling_mean_returns_panel.reindex(common_index).ffill().bfill() 

            regime_features_df[f'avg_mean_log_return_{window_size}d'] = rolling_mean_returns_panel.mean(axis=1)
            regime_features_df[f'std_mean_log_return_{window_size}d'] = rolling_mean_returns_panel.std(axis=1)
        else:
            print("Warning: No valid rolling mean log returns calculated for any constituent.")
            regime_features_df[f'avg_mean_log_return_{window_size}d'] = np.nan
            regime_features_df[f'std_mean_log_return_{window_size}d'] = np.nan

    # --- Feature 3: Average of constituent volatilities ---
    all_ticker_rolling_volatility = {}
    if constituent_tickers: # Check again if we have constituents
        for ticker in constituent_tickers:
            df_stock = all_data[ticker]
            if 'Log Return' in df_stock.columns and not df_stock['Log Return'].isnull().all():
                all_ticker_rolling_volatility[ticker] = df_stock['Log Return'].rolling(window=window_size, min_periods=int(window_size*0.8)).std()
            else:
                print(f"Note: 'Log Return' not available or all NaN for {ticker} when calculating rolling std (volatility).")
        
        if all_ticker_rolling_volatility:
            rolling_volatility_panel = pd.DataFrame(all_ticker_rolling_volatility)
            rolling_volatility_panel = rolling_volatility_panel.reindex(common_index).ffill().bfill()
            regime_features_df[f'avg_volatility_{window_size}d'] = rolling_volatility_panel.mean(axis=1)
        else:
            print("Warning: No valid rolling volatilities calculated for any constituent.")
            regime_features_df[f'avg_volatility_{window_size}d'] = np.nan
    else:
        regime_features_df[f'avg_volatility_{window_size}d'] = np.nan # if no constituents from the start

    # --- Feature 4: Smoothed ratio of positive returning constituents ---
    if constituent_tickers:
        positive_returns_mask = pd.DataFrame(index=common_index)
        for ticker in constituent_tickers:
            df_stock = all_data[ticker]
            if 'Log Return' in df_stock.columns:
                # Align series to common_index before comparison
                aligned_log_returns = df_stock['Log Return'].reindex(common_index) # ffill().bfill() might be needed if series are sparse
                positive_returns_mask[ticker] = (aligned_log_returns > 0).astype(int)
            else:
                positive_returns_mask[ticker] = 0 # Or np.nan if preferred for missing data
        
        # Sum of stocks with positive returns per day / total number of constituents considered
        daily_positive_ratio = positive_returns_mask.sum(axis=1) / len(constituent_tickers)
        regime_features_df[f'positive_return_ratio_{window_size}d'] = daily_positive_ratio.rolling(window=window_size, min_periods=int(window_size*0.8)).mean()
    else:
        regime_features_df[f'positive_return_ratio_{window_size}d'] = np.nan

    # Clean up: Drop rows where all new features are NaN (typically at the beginning due to rolling windows)
    initial_feature_rows = len(regime_features_df)
    regime_features_df.dropna(how='all', inplace=True)
    print(f"Regime features generated. Shape: {regime_features_df.shape}. Dropped {initial_feature_rows - len(regime_features_df)} rows with all NaNs.")

    # Display some info about the generated features
    if not regime_features_df.empty:
        print("\nFirst 5 rows of regime features:")
        print(regime_features_df.head())
        print("\nLast 5 rows of regime features:")
        print(regime_features_df.tail())
        print("\nDescriptive stats of regime features:")
        print(regime_features_df.describe())
    else:
        print("Regime features DataFrame is empty after processing.")

print("--- Part 5: Regime Feature Engineering Complete ---")

# --- Part 6: Unsupervised Market Regime Classification (Aggregated Features) ---
print("\n--- Part 6: Unsupervised Market Regime Classification (Using Real + Synthetic Data) --- ")

if 'regime_features_df' not in locals() or regime_features_df.empty:
    print("Error: regime_features_df is not available or empty. Skipping Part 6.")
else:
    # 1. Data Preparation and Preprocessing
    features_for_clustering_real = regime_features_df.copy()

    features_for_clustering_real.ffill(inplace=True)
    features_for_clustering_real.bfill(inplace=True)

    if features_for_clustering_real.isnull().sum().any():
        print("Warning: NaNs still present in real features before scaling. Dropping rows with NaNs.")
        features_for_clustering_real.dropna(inplace=True)
    
    if features_for_clustering_real.empty:
        print("Error: No real data left for clustering after NaN handling. Skipping Part 6.")
        # Ensure scaled_features_real is defined to avoid later errors if this path is taken
        scaled_features_real = None 
    else:
        print(f"Real features prepared for clustering. Shape: {features_for_clustering_real.shape}")
        
        # --- Load and Prepare Synthetic Data ---
        synthetic_features_df_loaded = None
        try:
            synthetic_features_df_loaded = pd.read_csv("all_regimes_synthetic_features.csv")
            print(f"Successfully loaded synthetic features. Shape: {synthetic_features_df_loaded.shape}")
            
            # These are the columns used for GMM input features from regime_features_df
            # Must match the columns in synthetic data (excluding 'Original_Market_Regime')
            # It's critical these are the *exact* features used to train VAE and for GMM
            gmm_input_feature_columns = [col for col in features_for_clustering_real.columns if col != 'Market_Regime']
            # This line ensures 'Market_Regime' is not accidentally included if it was in regime_features_df.
            # It's assumed features_for_clustering_real at this point only contains the GMM input features.

            print(f"Expected GMM input feature columns (from real data): {gmm_input_feature_columns}")

            # Ensure synthetic data has these columns and in the correct order for transform
            # Select only the GMM input feature columns from the loaded synthetic data
            if all(col in synthetic_features_df_loaded.columns for col in gmm_input_feature_columns):
                features_for_clustering_synthetic_raw = synthetic_features_df_loaded[gmm_input_feature_columns]
                if features_for_clustering_synthetic_raw.isnull().sum().any():
                    print("Warning: NaNs found in raw synthetic features. Filling with column means.")
                    features_for_clustering_synthetic_raw = features_for_clustering_synthetic_raw.fillna(features_for_clustering_synthetic_raw.mean())
            else:
                missing_cols = [col for col in gmm_input_feature_columns if col not in synthetic_features_df_loaded.columns]
                print(f"Error: Synthetic data is missing expected GMM input feature columns: {missing_cols}. Skipping synthetic data.")
                features_for_clustering_synthetic_raw = None

        except FileNotFoundError:
            print("Synthetic features file 'all_regimes_synthetic_features.csv' not found. Proceeding with real data only.")
            features_for_clustering_synthetic_raw = None
        except Exception as e:
            print(f"Error loading or processing synthetic features: {e}. Proceeding with real data only.")
            features_for_clustering_synthetic_raw = None
        # --- END Load and Prepare Synthetic Data ---

        # 2. Feature Standardization 
        scaled_features_real = None
        scaled_features_synthetic = None
        scaler = None

        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Fit scaler ONLY on real data and transform real data
            # gmm_input_feature_columns should now reliably define what to scale from features_for_clustering_real
            if not features_for_clustering_real[gmm_input_feature_columns].empty:
                scaled_features_real = scaler.fit_transform(features_for_clustering_real[gmm_input_feature_columns])
                print("Real features standardized.")
            else:
                 print("Error: Real feature set for scaling is empty. Cannot proceed.")
                 raise ValueError("Empty real feature set for scaling")


            # Transform synthetic data using the SAME scaler (fitted on real data)
            if features_for_clustering_synthetic_raw is not None and not features_for_clustering_synthetic_raw.empty:
                # Ensure columns are in the same order as for fitting the scaler
                scaled_features_synthetic = scaler.transform(features_for_clustering_synthetic_raw[gmm_input_feature_columns])
                print("Synthetic features standardized using scaler from real data.")
            
        except ImportError:
            print("Error: scikit-learn is not installed. Cannot perform clustering.")
        except Exception as e:
            print(f"Error during feature standardization: {e}.")
            # Ensure scaled_features_real might be None if an error occurs before its assignment
            if 'scaled_features_real' not in locals() or scaled_features_real is None : 
                scaled_features_real = None # Explicitly set to None
            scaled_features_synthetic = None # Synthetic definitely won't be available


        if scaled_features_real is not None:
            # Combine real and synthetic scaled features for K-Means model training
            if scaled_features_synthetic is not None and scaled_features_synthetic.shape[0] > 0:
                print(f"Combining {scaled_features_real.shape[0]} real samples and {scaled_features_synthetic.shape[0]} synthetic samples for clustering.")
                scaled_features_combined_for_fitting_kmeans = np.concatenate((scaled_features_real, scaled_features_synthetic), axis=0)
            else:
                print("Proceeding with real scaled features only for K-Means training (synthetic data not available or empty).")
                scaled_features_combined_for_fitting_kmeans = scaled_features_real
            
            # 3. Determine optimal K using Silhouette Score (on combined data if available, else on real)
            best_k = 3 # Default K
            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                
                silhouette_scores = {}
                k_range = range(2, 11) 
                
                print("\nCalculating Silhouette Scores for different K values (on combined or real data)...")
                for k_val in k_range:
                    kmeans_model_for_k_selection = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                    cluster_labels_for_k_selection = kmeans_model_for_k_selection.fit_predict(scaled_features_combined_for_fitting_kmeans)
                    
                    if len(set(cluster_labels_for_k_selection)) > 1: 
                        score = silhouette_score(scaled_features_combined_for_fitting_kmeans, cluster_labels_for_k_selection)
                        silhouette_scores[k_val] = score
                        print(f"  K={k_val}, Silhouette Score: {score:.4f}")
                    else:
                        print(f"  K={k_val}, Only one cluster found, Silhouette Score not applicable.")
                        silhouette_scores[k_val] = -1 

                if silhouette_scores:
                    best_k = max(silhouette_scores, key=silhouette_scores.get)
                    print(f"\nBest K based on Silhouette Score: {best_k} (Score: {silhouette_scores[best_k]:.4f})")
                else:
                    print("Could not calculate silhouette scores. Using default K=3.")

            except ImportError:
                print("scikit-learn components for K-Means/Silhouette not found. Using default K=3.")
            except Exception as e:
                print(f"Error during K selection: {e}. Using default K=3.")

            # 4. Apply K-Means with the best K
            print(f"\nApplying K-Means with K={best_k}...")
            kmeans_final_model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
            
            # Fit the final K-Means model on the combined data (if available)
            kmeans_final_model.fit(scaled_features_combined_for_fitting_kmeans)

            # Predict regime labels ONLY for the REAL data part
            final_cluster_labels_for_real_data = kmeans_final_model.predict(scaled_features_real)

            # 5. Add new regime labels to the original real features DataFrame
            # features_for_clustering_real still has the correct DatetimeIndex
            if len(final_cluster_labels_for_real_data) == len(features_for_clustering_real):
                # Use a new column name for the synthetically enhanced regime labels
                new_regime_column_name = 'Market_Regime_SynthEnhanced'
                features_for_clustering_real[new_regime_column_name] = final_cluster_labels_for_real_data
                print(f"'{new_regime_column_name}' labels added to the real features DataFrame.")
                print(f"\nValue counts for {new_regime_column_name} (K={best_k}):")
                print(features_for_clustering_real[new_regime_column_name].value_counts().sort_index())
                
                # Prepare the DataFrame for saving: original features + new synth-enhanced regime labels
                # We need to ensure that regime_features_df (which might have original 'Market_Regime')
                # gets updated or we save the correct DataFrame.
                # It's safer to create a new DataFrame or update regime_features_df carefully.
                
                # If regime_features_df was the source, we update it with the new regime column.
                # Drop the old 'Market_Regime' if it exists to avoid confusion, or save both.
                # For this purpose, we'll create a df that is regime_features_df's GMM input columns + the new regime labels
                
                output_df_synth_enhanced = features_for_clustering_real[gmm_input_feature_columns + [new_regime_column_name]].copy()
                
                # This output_df_synth_enhanced now contains the original GMM input features for real data
                # and the new Market_Regime_SynthEnhanced labels. Its index is the DatetimeIndex.

            else:
                print("Error: Length mismatch between new cluster labels and real feature data. Cannot assign new labels.")
                output_df_synth_enhanced = None # Indicate failure to produce the df for saving
        else: # This else is for "if scaled_features_real is not None:"
            print("Skipping K-Means clustering as real features were not scaled (likely due to earlier errors).")
            output_df_synth_enhanced = None


# --- Part 7: Analysis and Visualization of Regimes ---
# This part might need to be adjusted or duplicated if we want to analyze both original and synth-enhanced regimes.
# For now, it will use whatever 'Market_Regime' column is present in 'regime_features_df' (which is the original one).
# If we want to visualize the *new* regimes, the saving/loading logic needs to be clear.

print("\n--- Part 7: Analysis and Visualization of Regimes (using original GMM labels for now) --- ") # MODIFIED: Clarified this uses original

# Original 'Market_Regime' is still in 'regime_features_df' if it was loaded correctly
# and not overwritten before this point by mistake.
# The 'output_df_synth_enhanced' contains the *new* regime labels.

if 'regime_features_df' not in locals() or regime_features_df.empty or 'Market_Regime' not in regime_features_df.columns:
    print("Error: Original 'regime_features_df' with 'Market_Regime' is not available for Part 7.")
else:
    # ... (original Part 7 code for visualizing 'Market_Regime' from regime_features_df) ...
    # (Content of original Part 7, lines 637-740 from your file, would go here)
    # For brevity, I'm not reproducing all of it, but it should remain to analyze the *original* regimes.
    # Example snippet:
    if DJIA_TICKER in all_data and not all_data[DJIA_TICKER].empty:
        dji_prices = all_data[DJIA_TICKER]['Close'] 
        plot_df = pd.DataFrame(index=dji_prices.index)
        plot_df['DJIA_Close'] = dji_prices
        # This uses the ORIGINAL 'Market_Regime' from regime_features_df
        plot_df['Market_Regime_Original'] = regime_features_df['Market_Regime'].reindex(plot_df.index, method='ffill')
        plot_df.dropna(subset=['Market_Regime_Original'], inplace=True)
        # ... rest of the plotting code from original Part 7, adapting to use Market_Regime_Original
        # ... (ensure best_k here refers to the k for original regimes if that was determined in an earlier Part 6 run)
        # ... or if Part 6 is now *only* for synth-enhanced, this visualization part might need its own k.
        # This section needs careful review based on whether an "original" Part 6 run is preserved or not.


# --- Save Regime Features with SYNTHETICALLY ENHANCED Labels to NEW CSV ---
# This uses output_df_synth_enhanced which was prepared in the modified Part 6
if 'output_df_synth_enhanced' in locals() and isinstance(output_df_synth_enhanced, pd.DataFrame) and not output_df_synth_enhanced.empty:
    try:
        output_filename_synth_enhanced = "regime_features_with_labels_synth_enhanced.csv"
        output_df_synth_enhanced.to_csv(output_filename_synth_enhanced)
        print(f"\nSuccessfully saved GMM input features with synthetically enhanced regime labels to {output_filename_synth_enhanced}")
    except Exception as e:
        print(f"\nError saving synthetically enhanced regime features to CSV: {e}")
else:
    print("\nWarning: 'output_df_synth_enhanced' was not available or empty. Skipping save of synth-enhanced regimes to CSV.")

# --- Save Regime Features with ORIGINAL Labels to CSV (ensure this still happens if Part 6 was for synth-enhanced) ---
# This part needs to ensure that the *original* regime_features_df (with the original Market_Regime)
# is saved if it hasn't been overwritten.
# If the original Part 6 (KMeans on real data only) is no longer run, then this part might save
# regime_features_df which now contains Market_Regime_SynthEnhanced if we're not careful.

# To be safe, let's assume the original 'regime_features_with_labels.csv' is a result of a *previous*
# run of this script *before* synthetic data integration.
# The current run *only* produces 'regime_features_with_labels_synth_enhanced.csv'.
# So, we might remove the original saving part or make it conditional.
# For now, I will comment out the original save to avoid accidental overwrite with new labels
# under the old name.

# if 'regime_features_df' in locals() and isinstance(regime_features_df, pd.DataFrame) and not regime_features_df.empty and 'Market_Regime' in regime_features_df.columns:
#     try:
#         output_filename_original = "regime_features_with_labels.csv" # Original filename
#         # Ensure we are saving the one with the *original* Market_Regime if that's intended
#         # This depends on how regime_features_df was handled throughout.
#         # If features_for_clustering_real was used to add Market_Regime_SynthEnhanced,
#         # then regime_features_df itself might still hold the original Market_Regime.
#         # This logic is tricky and depends on variable scope and modification.
#         # To avoid issues, it's best if a script run *either* produces original regimes OR synth-enhanced ones,
#         # not both in a way that might cause confusion with variable reuse.
#         # regime_features_df.to_csv(output_filename_original)
#         # print(f"\nSuccessfully saved regime features with original labels to {output_filename_original}")
#     except Exception as e:
#         # print(f"\nError saving original regime features to CSV: {e}")
# else:
#     # print("\nWarning: Original 'regime_features_df' with 'Market_Regime' was not available. Skipping save of original to CSV.")

print("\n--- analyze_dow_data.py script finished ---") 