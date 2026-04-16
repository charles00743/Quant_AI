import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FACTOR_DATA_CSV = "djia_weekly_factors_v2.csv"  # <--- Uses V2 factor data
# REGIME_DATA_CSV = "regime_features_with_labels.csv" # Input: Daily regime data (Original)
REGIME_DATA_CSV = "regime_features_with_labels_synth_enhanced.csv" # Input: Daily regime data (Synth-Enhanced)
DJIA_INDEX_PREPROCESSED_CSV = "dow_data_preprocessed/INDEX_DJI_preprocessed.csv"

# Strategy Parameters
TOP_N_STOCKS = 5
INITIAL_CAPITAL = 1000000
TRANSACTION_COST_BPS = 0.001

# Market Regime Definitions (Numbers now correspond to Market_Regime_SynthEnhanced)
REGIME_NAMES = {
    0: "SynthEnhanced Regime 0", # Need to re-interpret based on factor analysis
    1: "SynthEnhanced Regime 1", # Need to re-interpret based on factor analysis
    2: "SynthEnhanced Regime 2"  # Need to re-interpret based on factor analysis
}

# --- MODIFIED: New factor config V6.1 ---
REGIME_FACTOR_CONFIG = {
    0: { # SynthEnhanced Regime 0 - Unchanged for now
        'MOM_12M': 'positive',
        'RS_12M': 'positive',
        'VOL_3M_STD': 'negative',
        'PriceToSMA200': 'positive',
    },
    1: { # SynthEnhanced Regime 1 - Corrected VOL direction
        'MOM_12M': 'negative',
        'RS_12M': 'negative',
        'VOL_3M_STD': 'negative', # Corrected direction
        'PriceToSMA200': 'negative',
    },
    2: { # SynthEnhanced Regime 2 - Focus on Ultra-Low Volatility
        'VOL_1M_STD': 'negative',
        'VOL_3M_STD': 'negative',
        # Removed Sharpe_3M and PriceToSMA50
    }
}
# --- END MODIFIED CONFIG ---

# DJIA Components
DJIA_COMPONENTS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]
PREPROCESSED_STOCK_DIR = "dow_data_preprocessed"

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

    # Check for regime column name in loaded regime_df_daily
    if 'Market_Regime_SynthEnhanced' in regime_df_daily.columns:
        print("Detected 'Market_Regime_SynthEnhanced' column in regime data.")
    elif 'Market_Regime' in regime_df_daily.columns:
        print("Detected 'Market_Regime' column in regime data.")
    else:
        raise ValueError("Loaded regime data CSV does not contain 'Market_Regime' or 'Market_Regime_SynthEnhanced' column.")

    print("Data loading complete.")
    return factor_df, regime_df_daily, dji_df

def align_regime_to_weekly(regime_df_daily, rebalance_dates):
    """Aligns daily regime data to weekly rebalance dates (taking last day's regime)."""
    if not isinstance(rebalance_dates, pd.DatetimeIndex):
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)

    # Determine which regime column to use
    regime_col_name = 'Market_Regime_SynthEnhanced' if 'Market_Regime_SynthEnhanced' in regime_df_daily.columns else 'Market_Regime'
    print(f"Aligning regimes using column: {regime_col_name}")

    weekly_regimes = regime_df_daily[regime_col_name].reindex(rebalance_dates, method='ffill')
    weekly_regimes = weekly_regimes.bfill()
    print("Daily regimes aligned to weekly rebalance dates.")

    # Return as DataFrame with the correct column name
    weekly_regimes_df = weekly_regimes.to_frame()
    weekly_regimes_df.columns = [regime_col_name] # Ensure the column name is preserved
    return weekly_regimes_df

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
            standardized_factors[f"Z_{factor_name}"] = 0.0
        else:
            standardized_factors[f"Z_{factor_name}"] = (col_data - col_data.mean()) / col_data.std()
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

    # Determine which regime column to use from the input weekly_regimes_df
    if 'Market_Regime_SynthEnhanced' in weekly_regimes_df.columns:
        regime_col_name_to_use = 'Market_Regime_SynthEnhanced'
    elif 'Market_Regime' in weekly_regimes_df.columns:
        regime_col_name_to_use = 'Market_Regime'
    else:
        raise ValueError("weekly_regimes_df does not contain a valid regime column.")
    print(f"Running backtest using regime column: {regime_col_name_to_use}")

    rebalance_dates = factor_df.index.get_level_values('Date').unique().sort_values()

    portfolio_log_returns_list = []
    portfolio_turnover_list = []
    benchmark_log_returns_list = []
    actual_rebalance_dates_for_returns = []

    current_holdings = {}
    current_portfolio_value = INITIAL_CAPITAL
    capital_over_time = {rebalance_dates[0] if len(rebalance_dates)>0 else pd.Timestamp.now() : INITIAL_CAPITAL}

    for i in range(len(rebalance_dates)):
        rb_date = rebalance_dates[i]

        if i + 1 >= len(rebalance_dates):
            break # Correctly placed inside the loop

        next_rb_date = rebalance_dates[i+1]

        current_regime_val = weekly_regimes_df.loc[rb_date, regime_col_name_to_use] if rb_date in weekly_regimes_df.index else np.nan
        if pd.isna(current_regime_val):
            print(f"Warning: No regime data for {rb_date}. Holding previous portfolio or skipping.")
            week_portfolio_log_return = 0.0
            if current_holdings:
                temp_week_return = 0.0
                for stock_ticker, weight in current_holdings.items():
                    holding_period_mask_daily = (all_stock_daily_data_mi.index.get_level_values('Date') > rb_date) & \
                                              (all_stock_daily_data_mi.index.get_level_values('Date') <= next_rb_date)
                    # Check if data exists for this period before accessing ticker
                    stock_returns_this_week_period = all_stock_daily_data_mi[holding_period_mask_daily]
                    if not stock_returns_this_week_period.empty and stock_ticker in stock_returns_this_week_period.index.get_level_values('Ticker'):
                         stock_returns_this_week_series = stock_returns_this_week_period.xs(stock_ticker, level='Ticker')
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
            continue # Skip to next rebalance date

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
                    stock_returns_this_week_period = all_stock_daily_data_mi[holding_period_mask_daily]
                    if not stock_returns_this_week_period.empty and stock_ticker in stock_returns_this_week_period.index.get_level_values('Ticker'):
                         stock_returns_this_week_series = stock_returns_this_week_period.xs(stock_ticker, level='Ticker')
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
            continue # Skip to next rebalance date

        composite_scores = get_factor_scores(factors_today_raw, current_regime, REGIME_FACTOR_CONFIG)

        ranked_stocks = composite_scores.dropna().sort_values(ascending=False)
        selected_stocks = ranked_stocks.head(TOP_N_STOCKS).index.tolist()
        print(f"Selected stocks ({len(selected_stocks)}): {selected_stocks}")

        new_holdings = {ticker: 1/len(selected_stocks) for ticker in selected_stocks} if selected_stocks else {}

        # Turnover Calculation
        num_kept = len(set(current_holdings.keys()) & set(new_holdings.keys()))
        turnover_fraction = (len(selected_stocks) - num_kept) / len(selected_stocks) if selected_stocks else (1.0 if current_holdings else 0.0)
        portfolio_turnover_list.append(turnover_fraction)

        transaction_costs_this_period = turnover_fraction * current_portfolio_value * TRANSACTION_COST_BPS
        effective_portfolio_value_after_cost_assumption = current_portfolio_value - transaction_costs_this_period

        week_portfolio_log_return = 0.0
        if selected_stocks:
            temp_week_return = 0.0
            for stock_ticker in selected_stocks:
                holding_period_mask_daily = (all_stock_daily_data_mi.index.get_level_values('Date') > rb_date) & \
                                          (all_stock_daily_data_mi.index.get_level_values('Date') <= next_rb_date)
                stock_daily_returns_for_period = all_stock_daily_data_mi[holding_period_mask_daily]
                if not stock_daily_returns_for_period.empty and stock_ticker in stock_daily_returns_for_period.index.get_level_values('Ticker'):
                    stock_returns_this_week_series = stock_daily_returns_for_period.xs(stock_ticker, level='Ticker')
                    if not stock_returns_this_week_series.empty:
                        temp_week_return += stock_returns_this_week_series.sum() * new_holdings[stock_ticker]
                else:
                    # This case might happen if a stock has no data for the entire week
                    print(f"Warning: No daily return data for selected stock {stock_ticker} in period {rb_date} to {next_rb_date}")
            week_portfolio_log_return = temp_week_return

        portfolio_log_returns_list.append(week_portfolio_log_return - (transaction_costs_this_period / current_portfolio_value if current_portfolio_value > 0 else 0))
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

    capital_over_time_s = pd.Series(capital_over_time).sort_index()
    
    # --- ADDED: Save V6 Capital Curve (or current version) ---
    try:
        # Let's make the filename dynamic or ensure it's for V6.1
        # For now, keeping the V6 name, assuming this is an iteration on V6
        capital_over_time_s.to_csv("capital_curve_V6_synth_enhanced.csv", header=True)
        print("\nSuccessfully saved current (Synth-Enhanced V6.1) capital curve to capital_curve_V6_synth_enhanced.csv")
    except Exception as e:
        print(f"\nError saving current capital curve: {e}")
    # --- END ADDED --- 

    return results_df, capital_over_time_s

def calculate_performance_metrics(returns_series, risk_free_rate_annual=0.0):
    if returns_series.empty or returns_series.isnull().all():
        # print("Warning: Returns series is empty or all NaN. Cannot calculate metrics.") # Reduced verbosity
        return pd.Series(dtype=float, index=['Total Return (Simple)', 'Annualized Log Return', 'Annualized Volatility', 'Sharpe Ratio (Log Approx)', 'Max Drawdown (Simple)', 'Number of Periods'])

    periods_per_year = 52

    cumulative_log_return = returns_series.sum()
    total_return_simple = np.exp(cumulative_log_return) - 1
    mean_log_return = returns_series.mean()
    std_log_return = returns_series.std()
    annualized_log_return = mean_log_return * periods_per_year
    annualized_volatility = std_log_return * np.sqrt(periods_per_year)

    sharpe_ratio = (annualized_log_return - risk_free_rate_annual) / annualized_volatility if annualized_volatility != 0 and not pd.isna(annualized_volatility) else np.nan

    # Corrected drawdown calculation
    if returns_series.empty:
        max_drawdown_simple = np.nan
    else:
        cap_path_for_drawdown = np.exp(returns_series.cumsum())
        start_value = pd.Series([1.0], index=[returns_series.index[0] - pd.Timedelta(days=1)])
        cap_path_for_drawdown = pd.concat([start_value, cap_path_for_drawdown])
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

    # Determine which regime column is present in weekly_regimes_df
    if 'Market_Regime_SynthEnhanced' in weekly_regimes_df.columns:
        regime_col_name_to_use = 'Market_Regime_SynthEnhanced'
    elif 'Market_Regime' in weekly_regimes_df.columns:
        regime_col_name_to_use = 'Market_Regime'
    else:
        print("Error: No valid regime column found in weekly_regimes_df for analysis.")
        return {}
    print(f"Analyzing performance using regime column: {regime_col_name_to_use}")

    # Ensure index is DatetimeIndex for joining
    if not isinstance(weekly_regimes_df.index, pd.DatetimeIndex):
        weekly_regimes_df.index = pd.to_datetime(weekly_regimes_df.index)

    results_with_regimes = results_df.join(weekly_regimes_df[regime_col_name_to_use], how='left')
    results_with_regimes = results_with_regimes.dropna(subset=[regime_col_name_to_use])

    for regime_val, name in REGIME_NAMES.items():
        print(f"\nMetrics for {name} (Regime {int(regime_val)}):")
        regime_mask = results_with_regimes[regime_col_name_to_use] == regime_val
        regime_subset_returns_strategy = results_with_regimes.loc[regime_mask, 'Portfolio_Log_Return']
        regime_subset_returns_benchmark = results_with_regimes.loc[regime_mask, 'Benchmark_Log_Return']

        print("  Strategy Performance in this Regime:")
        if not regime_subset_returns_strategy.empty:
            metrics_strategy = calculate_performance_metrics(regime_subset_returns_strategy, risk_free_rate_annual)
            print(metrics_strategy)
        else:
            print(f"  No valid strategy returns data for {name}.")

        print("\n  Benchmark Performance in this Regime:")
        if not regime_subset_returns_benchmark.empty:
            metrics_benchmark = calculate_performance_metrics(regime_subset_returns_benchmark, risk_free_rate_annual)
            print(metrics_benchmark)
        else:
            print(f"  No valid benchmark returns data for {name}.")

    return {}

# --- Main Execution ---
def main():
    print("--- Starting Dow Jones Index Enhancement Strategy Backtest --- (Synth-Enhanced Regime with Re-Analyzed Factors V6.1) ---") 

    REGIME_DATA_CSV_TO_TEST = "regime_features_with_labels_synth_enhanced.csv"
    print(f"*** Using Regime Definition File: {REGIME_DATA_CSV_TO_TEST} ***")

    try:
        factor_df, regime_df_daily, dji_df = load_data(FACTOR_DATA_CSV, REGIME_DATA_CSV_TO_TEST, DJIA_INDEX_PREPROCESSED_CSV)
    except Exception as e:
        print(f"Error in data loading: {e}")
        return

    all_stock_daily_data_dict = {}
    for ticker in DJIA_COMPONENTS:
        file_path = os.path.join(PREPROCESSED_STOCK_DIR, f"{ticker.replace('^', 'INDEX_')}_preprocessed.csv")
        if os.path.exists(file_path):
            try:
                stock_daily_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if 'Log Return' in stock_daily_df.columns:
                     all_stock_daily_data_dict[ticker] = stock_daily_df[['Log Return']]
                else:
                    print(f"Warning: Missing 'Log Return' for {ticker} in {file_path}")
            except Exception as e:
                print(f"Error loading daily data for {ticker} from {file_path}: {e}")
        else:
            print(f"Warning: Preprocessed daily data file not found for {ticker}: {file_path}")

    if not all_stock_daily_data_dict:
        print("Error: Could not load any preprocessed daily data for constituent stocks. Cannot proceed.")
        return

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
    
    # Save capital curve before printing performance metrics
    # This ensures the CSV is saved even if subsequent plotting fails.
    # The save location in run_backtest is better as it has direct access to capital_over_time_s
    # If not saved in run_backtest, uncomment here:
    # try:
    #     capital_over_time_s.to_csv("capital_curve_V6.1_synth_enhanced.csv", header=True)
    #     print("\nSuccessfully saved V6.1 (Synth-Enhanced) capital curve to capital_curve_V6.1_synth_enhanced.csv")
    # except Exception as e:
    #     print(f"\nError saving V6.1 capital curve: {e}")


    print("\n--- Overall Strategy Performance ---")
    overall_metrics = calculate_performance_metrics(results_df['Portfolio_Log_Return'])
    print(overall_metrics)

    print("\n--- Benchmark (DJIA Index) Performance ---")
    benchmark_metrics = calculate_performance_metrics(results_df['Benchmark_Log_Return'])
    print(benchmark_metrics)

    analyze_performance_by_regime(results_df, weekly_regimes_df)

    # Plotting 
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 6))
        results_df['Portfolio_Log_Return'].cumsum().plot(label='Strategy (V6.1 Synth-Enhanced)')
        results_df['Benchmark_Log_Return'].cumsum().plot(label='DJIA Index')
        plt.title('Cumulative Log Returns: Strategy vs. Benchmark (V6.1)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Log Return')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        capital_over_time_s.plot(label='Strategy Capital (V6.1 Synth-Enhanced)')
        dates_for_benchmark_calc = capital_over_time_s.index[1:]
        if not dates_for_benchmark_calc.empty:
            aligned_benchmark_log_returns = results_df['Benchmark_Log_Return'].reindex(dates_for_benchmark_calc, method='ffill').fillna(0)
            benchmark_capital_values = INITIAL_CAPITAL * np.exp(aligned_benchmark_log_returns.cumsum())
            start_date_for_benchmark_plot = capital_over_time_s.index[0]
            benchmark_capital_plot = pd.Series([INITIAL_CAPITAL], index=[start_date_for_benchmark_plot])
            benchmark_capital_plot = pd.concat([benchmark_capital_plot, benchmark_capital_values])
            benchmark_capital_plot.plot(label='Benchmark Capital (DJIA based)', linestyle='--')
        else:
            print("Note: Could not plot benchmark capital due to missing dates.")

        plt.title('Portfolio Value Over Time (V6.1)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        if 'Turnover' in results_df.columns and not results_df['Turnover'].empty:
            plt.figure(figsize=(12,6))
            results_df['Turnover'].plot(kind='line', title='Weekly Portfolio Turnover (V6.1)', marker='.', linestyle='-')
            plt.ylabel('Turnover Fraction')
            plt.xlabel('Rebalance Date')
            plt.grid(True)
            plt.show()
    except Exception as plot_e:
        print(f"An error occurred during plotting: {plot_e}")

    print("\n--- Backtesting Script Finished (V6.1) ---")

if __name__ == "__main__":
    main() 