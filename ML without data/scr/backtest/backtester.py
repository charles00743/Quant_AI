import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FACTOR_DATA_CSV = "djia_weekly_factors_v2.csv"  # <--- MODIFIED: Use V2 factor data
REGIME_DATA_CSV = "regime_features_with_labels.csv" # Input: Daily regime data (from analyze_dow_data.py)
# IMPORTANT: Ensure this path is correct and points to your preprocessed DJIA index data file
DJIA_INDEX_PREPROCESSED_CSV = "dow_data_preprocessed/INDEX_DJI_preprocessed.csv" 

# Strategy Parameters
TOP_N_STOCKS = 5
INITIAL_CAPITAL = 1000000  # 初始资金
TRANSACTION_COST_BPS = 0.001 # 双边交易成本，例如 0.1% = 10 bps (basis points) -> 0.001 for calculation

# Market Regime Definitions (as previously discussed)
REGIME_NAMES = {
    0: "Steady Bull Market",
    1: "Volatile Bear Market / Panic",
    2: "Consolidation / Mild Recovery"
}

# --- NEW: Regime-Specific Factor Configuration ---
REGIME_FACTOR_CONFIG = {
    0: { # Steady Bull Market - Aim for "contrarian" or "value" like behavior
        'MOM_12M': 'negative',    # Lower 12M momentum
        'RS_12M': 'negative',     # Lower 12M relative strength
        'VOL_3M_STD': 'positive', # Higher 3M volatility (raw volatility, not neg_VOL)
        'PriceToSMA200': 'negative',# Lower Price to SMA200 (value)
        # 'Sharpe_6M': 'negative' # Lower Sharpe (contrarian to recent good performers)
    },
    1: { # Volatile Bear Market / Panic - Aim for "strong survivors" or "quality momentum"
        'MOM_12M': 'positive',    # Higher 12M momentum
        'RS_12M': 'positive',     # Higher 12M relative strength
        'VOL_3M_STD': 'negative', # Lower 3M volatility
        'PriceToSMA50': 'positive', # Higher Price to SMA50 (strength)
        # 'Sharpe_6M': 'positive' # Higher Sharpe
    },
    2: { # Consolidation / Mild Recovery - Focus on Ultra-Low Volatility
        'VOL_3M_STD': 'negative', # Lower 3M volatility (strong signal)
        'VOL_1M_STD': 'negative', # Lower 1M volatility (showed positive Q5 return)
        # 'Sharpe_6M': 'positive',  # Higher 6M Sharpe ratio - Removed for this iteration
    }
}


# DJIA Components (ensure this is consistent with other scripts, used for loading daily data)
DJIA_COMPONENTS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]
PREPROCESSED_STOCK_DIR = "dow_data_preprocessed" # Directory for individual stock daily data

# --- Helper Functions ---

def load_data(factor_csv, regime_csv, dji_csv):
    """Loads factor data, regime data, and DJIA index data."""
    print(f"Loading weekly factor data from: {factor_csv}")
    if not os.path.exists(factor_csv):
        raise FileNotFoundError(f"Factor data file not found: {factor_csv}")
    factor_df = pd.read_csv(factor_csv, parse_dates=['Date'])
    factor_df.set_index(['Date', 'Ticker'], inplace=True)
    
    print(f"Loading daily regime data from: {regime_csv}")
    if not os.path.exists(regime_csv):
        raise FileNotFoundError(f"Regime data file not found: {regime_csv}")
    regime_df_daily = pd.read_csv(regime_csv, index_col='Date', parse_dates=True)
    
    print(f"Loading DJIA preprocessed data from: {dji_csv}")
    if not os.path.exists(dji_csv):
        raise FileNotFoundError(f"DJIA index data file not found: {dji_csv}")
    dji_df = pd.read_csv(dji_csv, index_col='Date', parse_dates=True)
    if 'Log Return' not in dji_df.columns:
        raise ValueError("DJIA data must contain 'Log Return' column.")

    print("Data loading complete.")
    return factor_df, regime_df_daily, dji_df

def align_regime_to_weekly(regime_df_daily, rebalance_dates):
    """Aligns daily regime data to weekly rebalance dates (taking last day's regime)."""
    if not isinstance(rebalance_dates, pd.DatetimeIndex):
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
        
    weekly_regimes = regime_df_daily['Market_Regime'].reindex(rebalance_dates, method='ffill')
    # weekly_regimes.bfill(inplace=True) # <-- MODIFIED: Avoid inplace on slice
    weekly_regimes = weekly_regimes.bfill()
    print("Daily regimes aligned to weekly rebalance dates.")
    return weekly_regimes.to_frame()

def get_factor_scores(current_factors_df_raw, regime, regime_config):
    """
    Calculates composite factor scores for stocks based on the current market regime.
    Dynamically selects factors, adjusts their direction, Z-scores them, and combines.
    """
    if regime not in regime_config:
        print(f"Warning: Regime {regime} not found in REGIME_FACTOR_CONFIG. Returning zero scores.")
        return pd.Series(0, index=current_factors_df_raw.index, dtype=float)

    config_for_regime = regime_config[regime]
    selected_factors_for_scoring = pd.DataFrame(index=current_factors_df_raw.index)
    
    # 1. Select factors and adjust direction
    factors_to_combine_names = []
    for factor_name, direction in config_for_regime.items():
        if factor_name not in current_factors_df_raw.columns:
            print(f"Warning: Factor '{factor_name}' for Regime {regime} not found in data. Skipping this factor.")
            continue
        
        adjusted_factor_series = current_factors_df_raw[factor_name].copy()
        if direction == 'negative':
            adjusted_factor_series = -adjusted_factor_series
        
        selected_factors_for_scoring[factor_name] = adjusted_factor_series
        factors_to_combine_names.append(factor_name)

    if not factors_to_combine_names:
        print(f"Warning: No valid factors selected for Regime {regime} after checking availability. Returning zero scores.")
        return pd.Series(0, index=current_factors_df_raw.index, dtype=float)

    # 2. Z-score the direction-adjusted factors
    standardized_factors = pd.DataFrame(index=selected_factors_for_scoring.index)
    for factor_name in factors_to_combine_names:
        col_data = selected_factors_for_scoring[factor_name]
        if col_data.notna().sum() < 2 or pd.isna(col_data.std()) or col_data.std() == 0:
            standardized_factors[f"Z_{factor_name}"] = 0.0 # Assign 0 if cannot standardize
            # print(f"Info: Cannot standardize {factor_name} for regime {regime} on current date (std is 0 or too few values). Setting Z-score to 0.")
        else:
            standardized_factors[f"Z_{factor_name}"] = (col_data - col_data.mean()) / col_data.std()
        # standardized_factors[f"Z_{factor_name}"].fillna(0, inplace=True) # <-- MODIFIED: Avoid inplace
        standardized_factors[f"Z_{factor_name}"] = standardized_factors[f"Z_{factor_name}"].fillna(0)


    # 3. Combine Z-scores (equal weight for now)
    if standardized_factors.empty:
        print(f"Warning: No factors were standardized for Regime {regime}. Returning zero scores.")
        return pd.Series(0, index=current_factors_df_raw.index, dtype=float)
        
    final_score = standardized_factors.sum(axis=1)
    return final_score


def run_backtest(factor_df, weekly_regimes_df, all_stock_daily_data_mi, dji_daily_log_returns):
    """Main backtesting loop."""
    print("\n--- Starting Backtest ---")
    
    rebalance_dates = factor_df.index.get_level_values('Date').unique().sort_values()
    
    portfolio_log_returns_list = [] 
    portfolio_turnover_list = [] 
    benchmark_log_returns_list = [] 
    actual_rebalance_dates_for_returns = [] # Store dates for which returns are calculated

    current_holdings = {} 
    current_portfolio_value = INITIAL_CAPITAL
    capital_over_time = {rebalance_dates[0] if len(rebalance_dates)>0 else pd.Timestamp.now() : INITIAL_CAPITAL} # Store portfolio value over time

    for i in range(len(rebalance_dates)):
        rb_date = rebalance_dates[i]
        
        if i + 1 >= len(rebalance_dates):
            break
        
        next_rb_date = rebalance_dates[i+1]
        # Holding period is from the day AFTER rb_date up to and including next_rb_date
        
        current_regime_val = weekly_regimes_df.loc[rb_date, 'Market_Regime'] if rb_date in weekly_regimes_df.index else np.nan
        if pd.isna(current_regime_val):
            print(f"Warning: No regime data for {rb_date}. Holding previous portfolio or skipping.")
            # For simplicity, assume we hold previous portfolio and earn its return (or 0 if no prev holdings)
            # This part needs robust handling based on strategy rules for missing data.
            # For now, if no regime, assume 0 return for the portfolio for this week.
            # And benchmark return for the period.
            week_portfolio_log_return = 0.0
            if current_holdings: # If there are previous holdings, calculate their return
                temp_week_return = 0.0
                for stock_ticker, weight in current_holdings.items():
                    holding_period_mask_daily = (all_stock_daily_data_mi.index.get_level_values('Date') > rb_date) & \
                                              (all_stock_daily_data_mi.index.get_level_values('Date') <= next_rb_date)
                    stock_returns_this_week_series = all_stock_daily_data_mi[holding_period_mask_daily].xs(stock_ticker, level='Ticker', drop_level=False)
                    if not stock_returns_this_week_series.empty:
                        temp_week_return += stock_returns_this_week_series.sum() * weight # sum of log returns * weight
                week_portfolio_log_return = temp_week_return
            portfolio_log_returns_list.append(week_portfolio_log_return)
            portfolio_turnover_list.append(0) # No rebalance decision, so 0 turnover if holding.
            
            dji_returns_slice = dji_daily_log_returns[(dji_daily_log_returns.index > rb_date) & (dji_daily_log_returns.index <= next_rb_date)]
            benchmark_log_returns_list.append(dji_returns_slice.sum() if not dji_returns_slice.empty else 0)
            actual_rebalance_dates_for_returns.append(rb_date) # Date when decision (or lack thereof) was made
            current_portfolio_value *= np.exp(week_portfolio_log_return) # Update portfolio value
            capital_over_time[next_rb_date] = current_portfolio_value
            continue
            
        current_regime = int(current_regime_val)
        print(f"\nRebalance Date: {rb_date.strftime('%Y-%m-%d')}, Regime: {current_regime} ({REGIME_NAMES.get(current_regime, 'Unknown')})")

        factors_today_raw = factor_df.loc[rb_date] 
        if factors_today_raw.empty:
            print(f"Warning: No factor data for {rb_date}. Holding previous portfolio or skipping (same logic as no regime).")
            week_portfolio_log_return = 0.0
            if current_holdings:
                temp_week_return = 0.0
                for stock_ticker, weight in current_holdings.items():
                    holding_period_mask_daily = (all_stock_daily_data_mi.index.get_level_values('Date') > rb_date) & \
                                              (all_stock_daily_data_mi.index.get_level_values('Date') <= next_rb_date)
                    stock_returns_this_week_series = all_stock_daily_data_mi[holding_period_mask_daily].xs(stock_ticker, level='Ticker', drop_level=False)
                    if not stock_returns_this_week_series.empty:
                        temp_week_return += stock_returns_this_week_series.sum() * weight
                week_portfolio_log_return = temp_week_return
            portfolio_log_returns_list.append(week_portfolio_log_return)
            portfolio_turnover_list.append(0)
            dji_returns_slice = dji_daily_log_returns[(dji_daily_log_returns.index > rb_date) & (dji_daily_log_returns.index <= next_rb_date)]
            benchmark_log_returns_list.append(dji_returns_slice.sum() if not dji_returns_slice.empty else 0)
            actual_rebalance_dates_for_returns.append(rb_date)
            current_portfolio_value *= np.exp(week_portfolio_log_return)
            capital_over_time[next_rb_date] = current_portfolio_value
            continue
        
        # --- MODIFIED: Pass raw factors to get_factor_scores ---
        composite_scores = get_factor_scores(factors_today_raw, current_regime, REGIME_FACTOR_CONFIG)
        
        ranked_stocks = composite_scores.dropna().sort_values(ascending=False)
        selected_stocks = ranked_stocks.head(TOP_N_STOCKS).index.tolist()
        print(f"Selected stocks ({len(selected_stocks)}): {selected_stocks}")

        new_holdings = {ticker: 1/len(selected_stocks) for ticker in selected_stocks} if selected_stocks else {}
        
        # Turnover Calculation (simplified: fraction of portfolio that changed)
        num_kept = len(set(current_holdings.keys()) & set(new_holdings.keys()))
        turnover_fraction = (len(selected_stocks) - num_kept) / len(selected_stocks) if selected_stocks else (1.0 if current_holdings else 0.0)
        portfolio_turnover_list.append(turnover_fraction)
        
        transaction_costs_this_period = turnover_fraction * current_portfolio_value * TRANSACTION_COST_BPS
        effective_portfolio_value_after_cost_assumption = current_portfolio_value - transaction_costs_this_period

        week_portfolio_log_return = 0.0
        if selected_stocks:
            temp_week_return = 0.0
            for stock_ticker in selected_stocks:
                # Ensure all_stock_daily_data_mi is indexed by (Date, Ticker)
                holding_period_mask_daily = (all_stock_daily_data_mi.index.get_level_values('Date') > rb_date) & \
                                          (all_stock_daily_data_mi.index.get_level_values('Date') <= next_rb_date)
                # Filter for the specific ticker and date range
                stock_daily_returns_for_period = all_stock_daily_data_mi[holding_period_mask_daily]
                if stock_ticker in stock_daily_returns_for_period.index.get_level_values('Ticker'):
                    # Access the 'Log Return' series correctly after filtering by ticker
                    stock_returns_this_week_series = stock_daily_returns_for_period.xs(stock_ticker, level='Ticker') 
                    if not stock_returns_this_week_series.empty:
                        temp_week_return += stock_returns_this_week_series.sum() * new_holdings[stock_ticker] # sum of log returns * weight
                else:
                    print(f"Warning: No daily return data for selected stock {stock_ticker} in period {rb_date} to {next_rb_date}")
            week_portfolio_log_return = temp_week_return 
        
        portfolio_log_returns_list.append(week_portfolio_log_return - (transaction_costs_this_period / current_portfolio_value if current_portfolio_value > 0 else 0) ) # Add check for current_portfolio_value > 0
        actual_rebalance_dates_for_returns.append(rb_date)

        # Update portfolio value for next iteration
        current_portfolio_value = effective_portfolio_value_after_cost_assumption * np.exp(week_portfolio_log_return) 
        capital_over_time[next_rb_date] = current_portfolio_value
        current_holdings = new_holdings
        
        dji_returns_slice = dji_daily_log_returns[(dji_daily_log_returns.index > rb_date) & (dji_daily_log_returns.index <= next_rb_date)]
        benchmark_log_returns_list.append(dji_returns_slice.sum() if not dji_returns_slice.empty else 0)

    print("--- Backtest Loop Complete ---")
    
    if not actual_rebalance_dates_for_returns:
        print("No rebalance periods were processed. Cannot generate results.")
        return pd.DataFrame(), pd.Series(dtype=float)

    results_df = pd.DataFrame({
        'Portfolio_Log_Return': portfolio_log_returns_list,
        'Benchmark_Log_Return': benchmark_log_returns_list[:len(portfolio_log_returns_list)],
        'Turnover': portfolio_turnover_list[:len(portfolio_log_returns_list)]
    }, index=pd.DatetimeIndex(actual_rebalance_dates_for_returns))
    
    # Align capital_over_time Series index to be DatetimeIndex for plotting consistency
    capital_over_time_s = pd.Series(capital_over_time).sort_index()
    
    return results_df, capital_over_time_s


def calculate_performance_metrics(returns_series, risk_free_rate_annual=0.0):
    if returns_series.empty or returns_series.isnull().all():
        print("Warning: Returns series is empty or all NaN. Cannot calculate metrics.")
        return pd.Series(dtype=float, index=['Total Return (Simple)', 'Annualized Log Return', 'Annualized Volatility', 'Sharpe Ratio (Log Approx)', 'Max Drawdown (Simple)', 'Number of Periods'])

    periods_per_year = 52 
    
    cumulative_log_return = returns_series.sum()
    total_return_simple = np.exp(cumulative_log_return) - 1
    mean_log_return = returns_series.mean()
    std_log_return = returns_series.std()
    annualized_log_return = mean_log_return * periods_per_year
    annualized_volatility = std_log_return * np.sqrt(periods_per_year)
    
    sharpe_ratio = (annualized_log_return - risk_free_rate_annual) / annualized_volatility if annualized_volatility != 0 else np.nan
    
    # Corrected drawdown calculation
    # Create a series representing the value of 1 unit invested
    cap_path_for_drawdown = np.exp(returns_series.cumsum())
    # Prepend the initial value of 1 for correct peak calculation from the start
    cap_path_for_drawdown = pd.concat([pd.Series([1.0], index=[returns_series.index[0] - pd.Timedelta(days=1)] if not returns_series.empty else []), cap_path_for_drawdown])
    cap_path_for_drawdown = cap_path_for_drawdown.sort_index()


    peak_simple = cap_path_for_drawdown.cummax()
    drawdown_simple = (cap_path_for_drawdown - peak_simple) / peak_simple
    max_drawdown_simple = drawdown_simple.min()

    metrics = pd.Series({
        'Total Return (Simple)': total_return_simple,
        'Annualized Log Return': annualized_log_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio (Log Approx)': sharpe_ratio,
        'Max Drawdown (Simple)': max_drawdown_simple,
        'Number of Periods': len(returns_series),
    })
    return metrics

def analyze_performance_by_regime(results_df, weekly_regimes_df, risk_free_rate_annual=0.0):
    print("\n--- Performance Analysis by Regime ---")
    regime_performance = {}
    
    # Ensure weekly_regimes_df index is DatetimeIndex for joining
    if not isinstance(weekly_regimes_df.index, pd.DatetimeIndex):
        weekly_regimes_df.index = pd.to_datetime(weekly_regimes_df.index)

    results_with_regimes = results_df.join(weekly_regimes_df['Market_Regime'], how='left')
    
    if 'Market_Regime' not in results_with_regimes.columns:
        print("Error: Could not join Market_Regime to results. Check indices.")
        return {name: pd.Series(dtype=float) for name in REGIME_NAMES.values()}
        
    results_with_regimes.dropna(subset=['Market_Regime'], inplace=True) # MODIFIED: Avoid inplace on potential slice
    # results_with_regimes = results_with_regimes.dropna(subset=['Market_Regime'])


    for regime_val, name in REGIME_NAMES.items():
        print(f"\nMetrics for {name} (Regime {int(regime_val)}):")
        regime_mask = results_with_regimes['Market_Regime'] == regime_val
        regime_subset_returns_strategy = results_with_regimes.loc[regime_mask, 'Portfolio_Log_Return']
        regime_subset_returns_benchmark = results_with_regimes.loc[regime_mask, 'Benchmark_Log_Return']
        
        print("  Strategy Performance in this Regime:")
        if not regime_subset_returns_strategy.empty and not regime_subset_returns_strategy.isnull().all():
            metrics_strategy = calculate_performance_metrics(regime_subset_returns_strategy, risk_free_rate_annual)
            print(metrics_strategy)
            # regime_performance[name] = metrics_strategy # Store strategy metrics if needed elsewhere
        else:
            print(f"  No valid strategy returns data for {name}.")
            # regime_performance[name] = pd.Series(dtype=float, index=['Total Return (Simple)', 'Annualized Log Return', 'Annualized Volatility', 'Sharpe Ratio (Log Approx)', 'Max Drawdown (Simple)', 'Number of Periods'])

        print("\n  Benchmark Performance in this Regime:")
        if not regime_subset_returns_benchmark.empty and not regime_subset_returns_benchmark.isnull().all():
            metrics_benchmark = calculate_performance_metrics(regime_subset_returns_benchmark, risk_free_rate_annual)
            print(metrics_benchmark)
        else:
            print(f"  No valid benchmark returns data for {name}.")
            
    # The function's original purpose was to return regime_performance for the strategy.
    # We can decide if we want to change what it returns or just use it for printing.
    # For now, let's keep the original return structure if it was used elsewhere, though it seems not currently.
    # If it was just for printing, then returning `None` or a more comprehensive dict is fine.
    # Returning an empty dict for now as the primary use here is printing.
    return {}

# --- Main Execution ---
def main():
    print("--- Starting Dow Jones Index Enhancement Strategy Backtest (Regime-Switching V2) ---") # MODIFIED: Updated title
    
    try:
        # Ensure FACTOR_DATA_CSV is updated to v2
        factor_df, regime_df_daily, dji_df = load_data(FACTOR_DATA_CSV, REGIME_DATA_CSV, DJIA_INDEX_PREPROCESSED_CSV)
    except Exception as e:
        print(f"Error in data loading: {e}")
        return

    all_stock_daily_data_dict = {}
    for ticker in DJIA_COMPONENTS: 
        file_path = os.path.join(PREPROCESSED_STOCK_DIR, f"{ticker.replace('^', 'INDEX_')}_preprocessed.csv")
        if os.path.exists(file_path):
            try:
                stock_daily_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                # Ensure 'Log Return' and 'Close' (if needed for other calcs later, though not directly used in returns here)
                if 'Log Return' in stock_daily_df.columns:
                     all_stock_daily_data_dict[ticker] = stock_daily_df[['Log Return']] # Only need Log Return for backtest
                else:
                    print(f"Warning: Missing 'Log Return' for {ticker} in {file_path}")
            except Exception as e:
                print(f"Error loading daily data for {ticker} from {file_path}: {e}")
        else:
            print(f"Warning: Preprocessed daily data file not found for {ticker}: {file_path}")

    if not all_stock_daily_data_dict:
        print("Error: Could not load any preprocessed daily data for constituent stocks. Cannot proceed.")
        return
        
    # Only concat 'Log Return'
    all_stock_daily_data_mi = pd.concat(
        {ticker: df['Log Return'] for ticker, df in all_stock_daily_data_dict.items() if isinstance(df, pd.DataFrame) and 'Log Return' in df.columns},
        names=['Ticker', 'Date'] 
    ).swaplevel().sort_index()

    if not isinstance(all_stock_daily_data_mi.index, pd.MultiIndex):
        print("Error: Failed to create multi-index for daily stock data.")
        return

    weekly_rebalance_dates = factor_df.index.get_level_values('Date').unique().sort_values()
    weekly_regimes_df = align_regime_to_weekly(regime_df_daily, weekly_rebalance_dates)

    results_df, capital_over_time_s = run_backtest(factor_df, weekly_regimes_df, all_stock_daily_data_mi, dji_df['Log Return'])
    
    if results_df.empty:
        print("Backtest did not produce any results. Exiting.")
        return

    # --- ADDED: Save V4 Capital Curve ---
    try:
        capital_over_time_s.to_csv("capital_curve_V4_original_regime.csv", header=True)
        print("\nSuccessfully saved V4 (Original Regime) capital curve to capital_curve_V4_original_regime.csv")
    except Exception as e:
        print(f"\nError saving V4 capital curve: {e}")
    # --- END ADDED ---

    print("\n--- Overall Strategy Performance ---")
    overall_metrics = calculate_performance_metrics(results_df['Portfolio_Log_Return'])
    print(overall_metrics)

    print("\n--- Benchmark (DJIA Index) Performance ---")
    benchmark_metrics = calculate_performance_metrics(results_df['Benchmark_Log_Return'])
    print(benchmark_metrics)
    
    analyze_performance_by_regime(results_df, weekly_regimes_df)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6))
    results_df['Portfolio_Log_Return'].cumsum().plot(label='Strategy')
    results_df['Benchmark_Log_Return'].cumsum().plot(label='DJIA Index')
    plt.title('Cumulative Log Returns: Strategy vs. Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    capital_over_time_s.plot(label='Strategy Capital')
    # Ensure benchmark_capital_values uses the same length as results_df for alignment
    # MODIFIED: Align benchmark plot data carefully with capital_over_time_s dates
    
    # Get the dates from capital_over_time, excluding the very first one (initial capital point)
    dates_for_benchmark_calc = capital_over_time_s.index[1:]
    
    # Align benchmark returns from results_df to these specific dates
    # Use reindex to ensure all dates from dates_for_benchmark_calc are present, ffill for any gaps if necessary
    aligned_benchmark_log_returns = results_df['Benchmark_Log_Return'].reindex(dates_for_benchmark_calc, method='ffill').fillna(0)
    
    benchmark_capital_values = INITIAL_CAPITAL * np.exp(aligned_benchmark_log_returns.cumsum())
    
    start_date_for_benchmark_plot = capital_over_time_s.index[0] if not capital_over_time_s.empty else None

    if start_date_for_benchmark_plot is not None:
        # Create a Series for benchmark capital with initial capital point
        benchmark_capital_plot = pd.Series([INITIAL_CAPITAL], index=[start_date_for_benchmark_plot])
        # benchmark_capital_values already has an index aligned with dates_for_benchmark_calc
        benchmark_capital_plot = pd.concat([benchmark_capital_plot, benchmark_capital_values])
        benchmark_capital_plot.plot(label='Benchmark Capital (DJIA based)', linestyle='--')
    else:
        print("Note: Could not determine start date for benchmark capital plot.")


    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    if 'Turnover' in results_df.columns and not results_df['Turnover'].empty:
        plt.figure(figsize=(12,6))
        results_df['Turnover'].plot(kind='line', title='Weekly Portfolio Turnover', marker='.', linestyle='-')
        plt.ylabel('Turnover Fraction')
        plt.xlabel('Rebalance Date')
        plt.grid(True)
        plt.show()

    print("\n--- Backtesting Script Finished ---")

if __name__ == "__main__":
    main() 