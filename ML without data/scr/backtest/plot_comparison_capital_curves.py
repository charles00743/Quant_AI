import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
CAPITAL_V4_CSV = "capital_curve_V4_original_regime.csv"
CAPITAL_V6_CSV = "capital_curve_V6_synth_enhanced.csv"
DJIA_INDEX_PREPROCESSED_CSV = "dow_data_preprocessed/INDEX_DJI_preprocessed.csv"
INITIAL_CAPITAL = 1000000

# --- Main Plotting Script ---
def main():
    print("--- Starting Capital Curve Comparison Plot ---")
    all_series_to_plot = {}

    # Load V4 Strategy Capital Curve
    try:
        v4_capital_df = pd.read_csv(CAPITAL_V4_CSV, index_col=0, parse_dates=True)
        # The CSV from to_csv(header=True) on a Series will have the Series name as the column header.
        # If no name, it will be '0'.
        print(f"Debug V4 CSV columns: {v4_capital_df.columns}")
        if not v4_capital_df.empty:
            v4_capital_series = v4_capital_df[v4_capital_df.columns[0]] # Take the first column by position
            v4_capital_series.name = "Strategy V4 (Original Regime)"
            all_series_to_plot[v4_capital_series.name] = v4_capital_series
            print(f"Successfully loaded V4 capital curve. Length: {len(v4_capital_series)}. Last value: {v4_capital_series.iloc[-1] if not v4_capital_series.empty else 'N/A'}")
            print(v4_capital_series.tail(3))
        else:
            print(f"Error: {CAPITAL_V4_CSV} loaded as an empty DataFrame.")
            v4_capital_series = None

    except FileNotFoundError:
        print(f"Error: {CAPITAL_V4_CSV} not found. Cannot plot V4 strategy.")
        v4_capital_series = None
    except Exception as e:
        print(f"Error loading {CAPITAL_V4_CSV}: {e}")
        v4_capital_series = None

    # Load V6 Strategy Capital Curve
    try:
        v6_capital_df = pd.read_csv(CAPITAL_V6_CSV, index_col=0, parse_dates=True)
        print(f"Debug V6 CSV columns: {v6_capital_df.columns}")
        if not v6_capital_df.empty:
            v6_capital_series = v6_capital_df[v6_capital_df.columns[0]] # Take the first column by position
            v6_capital_series.name = "Strategy V6 (Synth-Enhanced Regime)"
            all_series_to_plot[v6_capital_series.name] = v6_capital_series
            print(f"Successfully loaded V6 capital curve. Length: {len(v6_capital_series)}. Last value: {v6_capital_series.iloc[-1] if not v6_capital_series.empty else 'N/A'}")
            print(v6_capital_series.tail(3))
        else:
            print(f"Error: {CAPITAL_V6_CSV} loaded as an empty DataFrame.")
            v6_capital_series = None
            
    except FileNotFoundError:
        print(f"Error: {CAPITAL_V6_CSV} not found. Cannot plot V6 strategy.")
        v6_capital_series = None
    except Exception as e:
        print(f"Error loading {CAPITAL_V6_CSV}: {e}")
        v6_capital_series = None

    # Load DJIA Benchmark and Calculate its Capital Curve
    djia_capital_series_for_plot = None
    try:
        if not os.path.exists(DJIA_INDEX_PREPROCESSED_CSV):
            raise FileNotFoundError(f"DJIA index data file not found: {DJIA_INDEX_PREPROCESSED_CSV}")
        dji_df = pd.read_csv(DJIA_INDEX_PREPROCESSED_CSV, index_col='Date', parse_dates=True)
        if 'Log Return' not in dji_df.columns:
            raise ValueError("DJIA data must contain 'Log Return' column.")
        
        initial_date_strat = None
        if v4_capital_series is not None and not v4_capital_series.empty:
            initial_date_strat = v4_capital_series.index[0]
        if v6_capital_series is not None and not v6_capital_series.empty:
            if initial_date_strat is None or v6_capital_series.index[0] < initial_date_strat:
                initial_date_strat = v6_capital_series.index[0]
        
        if initial_date_strat:
            djia_log_returns_from_strat_start = dji_df['Log Return'][dji_df.index >= initial_date_strat].copy()
            
            if not djia_log_returns_from_strat_start.empty:
                # Create the capital series, starting with INITIAL_CAPITAL at the first date
                # The first log return applies to the capital *at the start* of that day.
                # So, capital at date D_i = capital at D_{i-1} * exp(log_return_at_D_i)
                # The first point in our SERIES is INITIAL_CAPITAL at initial_date_strat.
                # The value for initial_date_strat after its own return is INITIAL_CAPITAL * exp(return on initial_date_strat)

                # Correct calculation of benchmark capital path:
                # Start with a Series of log returns. The capital path is INITIAL_CAPITAL * exp(cumulative_sum_of_log_returns)
                # However, this needs an initial capital point *before* the first return.

                # Capital at time t = InitialCapital * exp(sum of log returns from start to t)
                cumulative_log_returns = djia_log_returns_from_strat_start.cumsum()
                djia_capital_values = INITIAL_CAPITAL * np.exp(cumulative_log_returns)

                # We need to prepend the INITIAL_CAPITAL at the date *before* the first log return if possible,
                # or at the very first date of the log return series *before* applying that first return.
                # The strategies' capital curves start with INITIAL_CAPITAL as the first point.
                
                # Create a series for DJIA capital, starting with INITIAL_CAPITAL on the first strategy date
                djia_capital_series_for_plot = pd.Series([INITIAL_CAPITAL], index=[initial_date_strat]) 
                # Then append the calculated capital values for subsequent dates
                djia_capital_series_for_plot = pd.concat([djia_capital_series_for_plot, djia_capital_values])
                # Drop the first row of djia_capital_values if its index is initial_date_strat to avoid duplication, 
                # as we manually added INITIAL_CAPITAL at initial_date_strat
                if djia_capital_series_for_plot.index[0] == djia_capital_series_for_plot.index[1]:
                    djia_capital_series_for_plot = djia_capital_series_for_plot.iloc[1:]
                
                # Ensure no duplicate index if initial_date_strat was in djia_capital_values.index
                djia_capital_series_for_plot = djia_capital_series_for_plot[~djia_capital_series_for_plot.index.duplicated(keep='first')]

                djia_capital_series_for_plot.name = "DJIA Benchmark"
                all_series_to_plot[djia_capital_series_for_plot.name] = djia_capital_series_for_plot
                print(f"Calculated DJIA benchmark capital curve. Length: {len(djia_capital_series_for_plot)}. Last value: {djia_capital_series_for_plot.iloc[-1] if not djia_capital_series_for_plot.empty else 'N/A'}")
                print(djia_capital_series_for_plot.tail(3))
            else:
                print("No DJIA log returns available from the determined strategy start date.")
        else:
            print("Could not determine a start date from strategy curves for DJIA alignment.")

    except FileNotFoundError:
        print(f"Error: {DJIA_INDEX_PREPROCESSED_CSV} not found. Cannot plot DJIA benchmark.")
    except ValueError as ve:
        print(f"ValueError loading or processing DJIA data: {ve}")
    except Exception as e:
        print(f"Error loading or processing DJIA data: {e}")

    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))

    for series_name, series_data in all_series_to_plot.items():
        if series_data is not None and not series_data.empty:
            linestyle = '-'
            color = None
            if "V4" in series_name:
                linestyle = '--'
            elif "DJIA" in series_name:
                linestyle = ':'
                color = 'grey'
            series_data.plot(label=series_name, linestyle=linestyle, color=color)
        
    plt.title('Strategy Capital Curves vs. DJIA Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') 
    print("\nSuggestion: Using log scale for y-axis for better comparison of growth rates.")
    
    plt.show()
    print("--- Plotting Complete ---")

if __name__ == "__main__":
    main() 