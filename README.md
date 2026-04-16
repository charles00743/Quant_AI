# Quant_AI
# Regime-Aware Equity Allocation with Synthetic Data Enhancement

A machine-learning-based quantitative trading project that combines market regime detection, synthetic feature generation, regime-adaptive factor selection, and backtesting on DJIA stocks.

---

## Overview

Financial markets are non-stationary and often alternate between bull markets, panic-driven bear markets, and consolidation phases. Static investment strategies usually struggle to maintain stable performance across these changing environments.

This project builds a **regime-aware quantitative trading framework** that adapts factor exposure to market conditions. The pipeline combines:

- historical data collection and preprocessing
- rolling market feature construction
- unsupervised market regime classification using GMM
- synthetic regime-feature generation using VAE
- regime-aware factor selection
- weekly portfolio construction and backtesting
- risk evaluation under normal and stress periods

---

## Project Goal

The goal of this project is to design a dynamic equity allocation strategy that:

- identifies changing market regimes
- adjusts factor preferences under different market conditions
- improves robustness with synthetic data enhancement
- outperforms a passive benchmark on a risk-adjusted basis

---

## Methodology

### 1. Data Preparation

The project uses historical OHLCV data for **DJIA constituent stocks** and the **DJIA index**.

Main preprocessing steps:

- download and merge market data by date
- fill missing values
- remove invalid or unusable series
- compute daily log returns
- normalize price series when needed
- smooth abnormal observations using rolling statistics

---

### 2. Market Regime Classification

To characterize short-term market behavior, the project constructs **20-day rolling market features**, including:

- `avg_mean_log_return_20d`
- `std_mean_log_return_20d`
- `avg_volatility_20d`
- `positive_return_ratio_20d`

These features are standardized and clustered using a **Gaussian Mixture Model (GMM)**.

The model identifies three regimes:

- **Regime 0 — Steady Bull Market**  
  sustained positive returns with relatively low volatility

- **Regime 1 — Volatile Bear Market / Panic**  
  negative returns, high volatility, and rapid drawdowns

- **Regime 2 — Consolidation / Mild Recovery**  
  range-bound movement with mild upward bias

---

### 3. Synthetic Data Enhancement

To improve regime classification stability, the project trains a separate **Variational Autoencoder (VAE)** for each regime.

The VAE is used to:

- learn regime-specific feature structure
- generate synthetic feature samples
- reduce noisy or unstable regime transitions
- improve the robustness of downstream strategy signals

This creates a **synth-enhanced regime labeling** version of the strategy in addition to the baseline regime model.

---

### 4. Factor Calculation

The strategy computes several standard quantitative factors for DJIA stocks, including:

- **Momentum (`MOM_n`)**
- **Volatility (`VOL_n_STD`)**
- **Relative Strength (`RS_n`)**
- **Sharpe Ratio**
- **Simple Moving Average (`SMA_n`)**
- **Price relative to SMA (`PriceToSMA_n`)**

Typical lookback windows include:

- 1 month
- 3 months
- 6 months
- 12 months

---

### 5. Regime-Aware Strategy Design

The trading strategy uses a **weekly rebalance schedule**.

At each rebalance date, it:

1. identifies the current market regime
2. extracts stock-level factor values
3. standardizes selected factors using z-scores
4. adjusts factor signs according to regime-specific preferences
5. combines factor scores into a composite score
6. selects the **top 5 stocks**
7. allocates capital **equally weighted**

This creates a dynamic strategy that changes factor emphasis depending on the market state.

---

## Backtesting Setup

Main backtest settings:

- **Universe:** DJIA constituent stocks
- **Benchmark:** DJIA index
- **Initial capital:** $1,000,000
- **Rebalance frequency:** Weekly
- **Portfolio size:** 5 stocks
- **Weighting:** Equal-weighted
- **Transaction cost:** 10 bps
- **Number of market regimes:** 3

Two strategy versions are compared:

- **Original Regime Strategy**
- **Synth-Enhanced Regime Strategy**

---

## Results Summary

The project shows that the regime-aware strategy outperforms the DJIA benchmark over the test period.

Main findings:

- the original regime-aware strategy outperforms the benchmark
- the synth-enhanced strategy further improves Sharpe ratio
- the synth-enhanced strategy reduces maximum drawdown
- synthetic enhancement improves robustness during volatile market conditions
- the strategy performs especially well during bearish and recovery-related phases

---

## Risk Analysis

The project evaluates risk using:

- CAGR
- annualized volatility
- Sharpe ratio
- maximum drawdown
- skewness
- kurtosis
- VaR (95%)
- CVaR (95%)

It also includes stress-period analysis for market events such as:

- inflation panic
- rate hike volatility
- banking turmoil

This helps evaluate not only return performance, but also resilience under adverse conditions.

---
## Suggested Project Structure

ml-finance-regime-strategy/
├─ README.md
├─ data/
│  └─ processed/
│     ├─ regime_features_with_labels.csv
│     ├─ regime_features_with_labels_synth_enhanced.csv
│     ├─ all_regimes_synthetic_features.csv
│     ├─ djia_weekly_factors.csv
│     └─ djia_weekly_factors_v2.csv
├─ src/
│  ├─ data/
│  │  ├─ download_dow_data.py
│  │  ├─ analyze_dow_data.py
│  │  └─ generate_regime_data.py
│  ├─ regime/
│  │  └─ stock_analyzer.py
│  ├─ factors/
│  │  ├─ factor_calculator.py
│  │  └─ factor_analyzer.py
│  └─ backtest/
│     ├─ backtester.py
│     ├─ backtester_synth_regime.py
│     └─ plot_comparison_capital_curves.py
├─ outputs/
│  └─ backtests/
│     ├─ capital_curve_V4_original_regime.csv
│     └─ capital_curve_V6_synth_enhanced.csv
├─ reports/
│  └─ report.pdf
└─ tests/
   ├─ test_factors.py
   ├─ test_backtester.py
   └─ test_regime_features.py
