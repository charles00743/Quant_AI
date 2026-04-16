import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
FACTOR_DATA_CSV = "djia_weekly_factors_v2.csv"
REGIME_DATA_CSV = "regime_features_with_labels_synth_enhanced.csv"
PREPROCESSED_STOCK_DIR = "dow_data_preprocessed"
DJIA_COMPONENTS = [ # Ensure consistency
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
    'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
]
DJIA_TICKER_INDEX = "^DJI" # Needed? Perhaps just for calendar if not in factor df index

# Quintile Analysis Parameters
N_QUANTILES = 5

# Factors to analyze (ensure these column names exist in factor_df)
FACTORS_TO_ANALYZE = [
    'MOM_1M', 'MOM_3M', 'MOM_6M', 'MOM_12M',
    'RS_1M', 'RS_3M', 'RS_6M', 'RS_12M',
    'Sharpe_3M', 'Sharpe_6M',
    'VOL_1M_STD', 'VOL_3M_STD',
    'PriceToSMA50', 'PriceToSMA200'
    # Add 'SMA50', 'SMA200' if needed, or negative versions of VOL
]

# Define expected factor directions (True if higher factor value should lead to higher returns)
EXPECTED_FACTOR_DIRECTION = {
    'MOM_1M': True, 'MOM_3M': True, 'MOM_6M': True, 'MOM_12M': True,
    'RS_1M': True, 'RS_3M': True, 'RS_6M': True, 'RS_12M': True,
    'Sharpe_3M': True, 'Sharpe_6M': True,
    'VOL_1M_STD': False, 'VOL_3M_STD': False, # Lower volatility expected better (esp. risk-adjusted)
    'PriceToSMA50': True, 'PriceToSMA200': True
}


# --- Helper Functions ---

def load_analyzer_data(factor_csv, regime_csv, stock_dir):
    """Loads factors, daily regimes, and daily stock returns."""
    print(f"Loading weekly factor data from: {factor_csv}")
    if not os.path.exists(factor_csv):
        raise FileNotFoundError(f"Factor data file not found: {factor_csv}")
    factor_df = pd.read_csv(factor_csv, parse_dates=['Date'])
    # Ensure Ticker is string
    factor_df['Ticker'] = factor_df['Ticker'].astype(str)
    factor_df.set_index(['Date', 'Ticker'], inplace=True)

    print(f"Loading daily regime data from: {regime_csv}")
    if not os.path.exists(regime_csv):
        raise FileNotFoundError(f"Regime data file not found: {regime_csv}")
    regime_df_daily = pd.read_csv(regime_csv, index_col='Date', parse_dates=True)
    if 'Market_Regime_SynthEnhanced' in regime_df_daily.columns:
        regime_col_name_to_use = 'Market_Regime_SynthEnhanced'
        print(f"Using regime column: {regime_col_name_to_use}")
    elif 'Market_Regime' in regime_df_daily.columns:
        regime_col_name_to_use = 'Market_Regime'
        print(f"Warning: Synth-enhanced regime column not found, using original: {regime_col_name_to_use}")
    else:
        raise ValueError("Regime data CSV must contain 'Market_Regime' or 'Market_Regime_SynthEnhanced' column.")
    regime_df_daily = regime_df_daily[[regime_col_name_to_use]]

    print(f"Loading daily stock data from: {stock_dir}")
    all_stock_daily_data_dict = {}
    for ticker in DJIA_COMPONENTS: # Assuming components list is correct
        file_path = os.path.join(stock_dir, f"{ticker.replace('^', 'INDEX_')}_preprocessed.csv")
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
        raise ValueError("Could not load any daily stock return data.")

    # Combine daily returns into a multi-index DataFrame
    all_stock_daily_returns_mi = pd.concat(
        {ticker: df['Log Return'] for ticker, df in all_stock_daily_data_dict.items()},
        names=['Ticker', 'Date']
    ).swaplevel().sort_index() # Index (Date, Ticker)

    print("Data loading complete.")
    return factor_df, regime_df_daily, all_stock_daily_returns_mi


def calculate_forward_returns(daily_returns_mi, rebalance_dates):
    """Calculates forward weekly returns for each stock following each rebalance date."""
    forward_returns = {}
    for i in range(len(rebalance_dates) - 1):
        start_date_excl = rebalance_dates[i]
        end_date_incl = rebalance_dates[i+1]
        # Filter daily returns for the period *between* rebalance dates
        period_returns = daily_returns_mi[(daily_returns_mi.index.get_level_values('Date') > start_date_excl) &
                                          (daily_returns_mi.index.get_level_values('Date') <= end_date_incl)]
        # Calculate sum of log returns for each ticker over the period
        weekly_log_returns = period_returns.groupby(level='Ticker').sum()
        # Store these returns associated with the start date (rebalance date)
        forward_returns[start_date_excl] = weekly_log_returns

    # Combine into a DataFrame: Index=Date (rebalance date), Columns=Tickers
    forward_returns_df = pd.DataFrame(forward_returns).T
    # Convert to MultiIndex format (Date, Ticker) -> ForwardReturn
    forward_returns_mi = forward_returns_df.stack()
    forward_returns_mi.index.names = ['Date', 'Ticker']
    forward_returns_mi.name = 'Forward_1W_LogReturn'
    print("Forward weekly returns calculated.")
    return forward_returns_mi


def align_regime_to_weekly(regime_df_daily, rebalance_dates):
    """Aligns daily regime data to weekly rebalance dates (taking last day's regime)."""
    if not isinstance(rebalance_dates, pd.DatetimeIndex):
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
        
    regime_col_name = regime_df_daily.columns[0]
    weekly_regimes = regime_df_daily[regime_col_name].reindex(rebalance_dates, method='ffill')
    weekly_regimes = weekly_regimes.bfill() 
    print(f"Daily regimes (using {regime_col_name}) aligned to weekly rebalance dates.")
    weekly_regimes_df = weekly_regimes.to_frame()
    weekly_regimes_df.columns = [regime_col_name]
    return weekly_regimes_df


def assign_quantile(series, n_quantiles=5):
    """Assigns quantile number based on series values."""
    try:
        quantiles = pd.qcut(series.dropna(), q=n_quantiles, labels=False, duplicates='drop') + 1
        return quantiles.reindex(series.index)
    except ValueError as e:
        # print(f"Warning: qcut failed for a date, possibly too few unique values. Assigning NaN. Error: {e}")
        return pd.Series(np.nan, index=series.index)


def run_quintile_analysis(merged_data, factor, n_quantiles=5):
    """Performs quintile analysis for a given factor."""
    factor_col_name = factor 
    quantile_col_name = f'{factor}_Q'

    if 'Market_Regime_SynthEnhanced' in merged_data.columns:
        regime_col_name = 'Market_Regime_SynthEnhanced'
    elif 'Market_Regime' in merged_data.columns:
        regime_col_name = 'Market_Regime'
    else:
        regime_col_name = None 
        print("Warning: No Market_Regime or Market_Regime_SynthEnhanced column found in merged_data. Cannot perform by-regime analysis.")

    if factor_col_name not in merged_data.columns:
        print(f"Factor {factor_col_name} not found in data. Skipping.")
        return None, None, None # Return tuple of Nones
        
    analysis_data = merged_data.dropna(subset=['Forward_1W_LogReturn', factor_col_name]).copy()

    # Assign quantiles per date using transform
    analysis_data[quantile_col_name] = analysis_data.groupby(level='Date', group_keys=False)[factor_col_name].apply(
        lambda x: assign_quantile(x, n_quantiles)
    )

    analysis_data.dropna(subset=[quantile_col_name], inplace=True)
    if analysis_data.empty:
        print(f"No valid data after quantile assignment for factor {factor}. Skipping.")
        return None, None, None
        
    analysis_data[quantile_col_name] = analysis_data[quantile_col_name].astype(int)

    # --- Calculate overall returns ---    
    quintile_returns_daily = analysis_data.groupby([analysis_data.index.get_level_values('Date'), quantile_col_name])['Forward_1W_LogReturn'].mean()
    quintile_returns_overall = quintile_returns_daily.groupby(level=1).mean()

    # --- Calculate by-regime returns --- 
    quintile_returns_by_regime = None # Initialize
    if regime_col_name:
        # Ensure Market_Regime is not NaN for by-regime grouping
        analysis_data_for_regime_calc = analysis_data.dropna(subset=[regime_col_name])
        if not analysis_data_for_regime_calc.empty:
            quintile_returns_by_regime_date = analysis_data_for_regime_calc.groupby(
                [analysis_data_for_regime_calc.index.get_level_values('Date'), regime_col_name, quantile_col_name]
            )['Forward_1W_LogReturn'].mean()
            
            if not quintile_returns_by_regime_date.empty:
                # Group by Regime (level 1) and Quantile (level 2)
                quintile_returns_by_regime = quintile_returns_by_regime_date.groupby(level=[1, 2]).mean().unstack() 
            else:
                print(f"Warning: Groupby for by-regime returns yielded empty result for factor {factor}.")
        else:
             print(f"Warning: No valid data remaining after dropping NaNs in {regime_col_name} for by-regime analysis (Factor: {factor}).")
    
    print(f"Quintile analysis complete for factor: {factor}")
    return quintile_returns_overall, quintile_returns_by_regime, analysis_data


def plot_quintile_results(factor_name, quintile_returns_overall, quintile_returns_by_regime, analysis_data, expected_direction_positive=True):
    """Generates plots for quintile analysis results."""
    if quintile_returns_overall is None or quintile_returns_overall.empty:
        print(f"No overall quintile results to plot for {factor_name}.")
        return
        
    n_quantiles = len(quintile_returns_overall)
    quantile_labels = [f'Q{i+1}' for i in range(n_quantiles)]

    plt.figure(figsize=(8, 5))
    try:
        plot_data_overall = quintile_returns_overall.copy()
        plot_data_overall.index = quantile_labels 
        plot_data_overall.plot(kind='bar', color='skyblue')
    except Exception as e:
        print(f"Error setting index for overall plot: {e}. Plotting with default index.")
        quintile_returns_overall.plot(kind='bar', color='skyblue')
        
    plt.title(f'Overall Avg Weekly Log Return by {factor_name} Quantile')
    plt.xlabel('Quantile (Q1=Low, Q5=High)')
    plt.ylabel('Avg Weekly Log Return')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.show()

    if analysis_data is not None and f'{factor_name}_Q' in analysis_data.columns and not analysis_data.empty:
        # Ensure the quantile column exists before attempting to access it
        quantile_col = f'{factor_name}_Q'
        if quantile_col in analysis_data.columns:
            q5_returns = analysis_data[analysis_data[quantile_col] == n_quantiles].groupby(level='Date')['Forward_1W_LogReturn'].mean()
            q1_returns = analysis_data[analysis_data[quantile_col] == 1].groupby(level='Date')['Forward_1W_LogReturn'].mean()
            
            spread_returns = q5_returns.subtract(q1_returns, fill_value=0)
            if not expected_direction_positive:
                 spread_returns = -spread_returns # Analyze Q1 - Q5 for factors where lower is better

            cumulative_spread = spread_returns.cumsum()

            plt.figure(figsize=(12, 6))
            cumulative_spread.plot(color='green')
            title_suffix = " (Flipped for Low=Good Factors)" if not expected_direction_positive else ""
            plt.title(f'{factor_name}: Cumulative Log Return of Q5-Q1 Portfolio{title_suffix}')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Log Return')
            plt.grid(True)
            plt.show()
        else:
             print(f"Warning: Quantile column '{quantile_col}' not found in analysis data for Q5-Q1 plot.")

    if quintile_returns_by_regime is not None and not quintile_returns_by_regime.empty:
        unique_regimes = sorted(quintile_returns_by_regime.index.unique())
        n_regimes = len(unique_regimes)
        if n_regimes > 0:
            fig_regime, axes_regime = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 5), sharey=True, squeeze=False) # Ensure axes_regime is always 2D
            axes_regime = axes_regime.flatten()
                 
            for i, regime_id in enumerate(unique_regimes):
                ax = axes_regime[i]
                if regime_id in quintile_returns_by_regime.index:
                    data_to_plot = quintile_returns_by_regime.loc[regime_id]
                    if not data_to_plot.isnull().all(): # Check if data exists for this regime
                        try:
                            plot_data_regime = data_to_plot.copy()
                            plot_data_regime.index = quantile_labels
                            plot_data_regime.plot(kind='bar', ax=ax, color='coral')
                        except Exception as e:
                            print(f"Error setting index for regime {regime_id} plot: {e}. Plotting with default index.")
                            data_to_plot.plot(kind='bar', ax=ax, color='coral')
                            
                        ax.set_title(f'Regime {int(regime_id)}: Avg Wkly Log Rtn')
                        ax.set_xlabel('Quantile')
                        ax.set_ylabel('Avg Weekly Log Return' if i == 0 else '')
                        ax.tick_params(axis='x', rotation=0)
                        ax.grid(axis='y', linestyle='--')
                    else:
                         ax.set_title(f'Regime {int(regime_id)}: No Data')
                         ax.set_xlabel('Quantile')
                else:
                    ax.set_title(f'Regime {int(regime_id)}: Data Missing')
                    ax.set_xlabel('Quantile')

            fig_regime.suptitle(f'{factor_name}: Avg Weekly Log Return by Quantile per Regime', fontsize=16, y=1.02)
            fig_regime.tight_layout()
            plt.show()
        else:
            print(f"No valid regimes found in by-regime results for {factor_name}.")
    else:
        print(f"No by-regime results to plot for {factor_name}.")

# --- Main Execution ---
def main():
    print("--- Starting Factor Performance Analysis (Using Synth-Enhanced Regimes) ---")

    try:
        factor_df, regime_df_daily, all_stock_daily_returns_mi = load_analyzer_data(
            FACTOR_DATA_CSV, REGIME_DATA_CSV, PREPROCESSED_STOCK_DIR
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    rebalance_dates = factor_df.index.get_level_values('Date').unique().sort_values()
    if len(rebalance_dates) < 2:
        print("Not enough rebalance dates found to calculate forward returns.")
        return
        
    weekly_regimes_df = align_regime_to_weekly(regime_df_daily, rebalance_dates)
    forward_returns_mi = calculate_forward_returns(all_stock_daily_returns_mi, rebalance_dates)

    merged_data = factor_df.join(forward_returns_mi, how='left')
    merged_data = merged_data.reset_index().merge(
        weekly_regimes_df.reset_index(),
        on='Date', 
        how='left'
    ).set_index(['Date', 'Ticker'])
    
    print("Data merged for analysis.")
    if merged_data.empty:
        print("Merged data is empty. Cannot proceed.")
        return
        
    print(f"Shape of merged data: {merged_data.shape}")
    print("Merged data columns:", merged_data.columns)

    if 'Forward_1W_LogReturn' not in merged_data.columns:
         print("Error: Forward returns could not be merged correctly.")
         return
    # Check for Market_Regime after merge
    if 'Market_Regime' not in merged_data.columns:
         print("Error: Market regimes could not be merged correctly after joining.")
         # Decide whether to proceed without regime analysis or stop
         # return 

    plt.style.use('seaborn-v0_8-darkgrid')
    
    for factor in FACTORS_TO_ANALYZE:
        print(f"\n--- Analyzing Factor: {factor} ---")
        expected_positive = EXPECTED_FACTOR_DIRECTION.get(factor, True)
        
        factor_to_analyze = factor
        # Create negative version if needed for analysis, store temporarily
        temp_neg_factor_col = None
        if not expected_positive:
            factor_to_analyze = f"neg_{factor}"
            if factor in merged_data.columns:
                merged_data[factor_to_analyze] = -merged_data[factor]
                temp_neg_factor_col = factor_to_analyze # Remember the name to drop later
            else:
                print(f"Warning: Cannot create negative version for non-existent factor {factor}. Skipping.")
                continue

        results = run_quintile_analysis(merged_data, factor_to_analyze, n_quantiles=N_QUANTILES)

        # Clean up the temporary negative factor column if we added it
        if temp_neg_factor_col and temp_neg_factor_col in merged_data.columns:
            merged_data.drop(columns=[temp_neg_factor_col], inplace=True)

        if results and results[0] is not None:
            quintile_returns_overall, quintile_returns_by_regime, analysis_data_for_factor = results
            print("\nOverall Avg Weekly Returns by Quantile:")
            print(quintile_returns_overall)
            if quintile_returns_by_regime is not None:
                 print("\nAvg Weekly Returns by Quantile per Regime:")
                 print(quintile_returns_by_regime)
            else:
                 print("\nBy-Regime analysis could not be performed.")

            # Pass the original factor name to plotting for titles etc.
            plot_quintile_results(factor, quintile_returns_overall, quintile_returns_by_regime, analysis_data_for_factor, expected_positive)
        else:
             print(f"Analysis failed or produced no results for factor: {factor}")

    print("\n--- Factor Performance Analysis Finished ---")


if __name__ == "__main__":
    main() 