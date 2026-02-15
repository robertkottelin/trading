# BTC ML Trading Model — Findings & Learnings

## Project Overview

Goal: Build an ML model that predicts Bitcoin price movements using historical data, trained on 5-minute candle data from Binance.

## Data Pipeline (`btc_indicators.py`)

### Data Sources (kept)
- **Spot klines** (Binance BTCUSDT): 892,415 rows from 2017-08-17, 5m resolution (~113 MB raw)
- **Futures klines** (Binance BTCUSDT perpetual): 677,390 rows from 2019-09-08, 5m resolution (~80 MB raw)

### Data Sources (removed — not enough data or snapshot-only)
- Funding rate (~7K rows, 8h intervals — too sparse)
- Open Interest & Long/Short ratio (limited history)
- Order book snapshots (point-in-time only, no history)
- Whale transactions (limited, noisy)

### Features (116 total)
- **TA indicators (~86)**: via `ta.add_all_ta_features()` — trend (SMA, EMA, MACD, ADX, Aroon, CCI, Ichimoku, PSAR), momentum (RSI, Stochastic, Williams %R, ROC), volatility (Bollinger, ATR, Keltner, Donchian), volume (OBV, VWAP, CMF, ADI, MFI)
- **CVD (3)**: delta per candle, cumulative delta, delta SMA-14
- **Custom price (10)**: returns at 1/6/12/288 periods, log return, SMA50/200 distance, high-low range, hour_of_day, day_of_week
- **Futures (5)**: futures close, basis, basis_pct, futures volume ratio, futures trades

### ML Targets (27 total)
- **Direction (10)**: return and binary direction at 5 horizons (5m, 30m, 1h, 3h, 24h)
- **Trend (9)**: trend strength/up/down at 3 horizons (1h, 3h, 24h)
- **Volatility (8)**: forward volatility, max runup/drawdown, big move detection

### Output
- `btc_data/btc_features_5m.csv` — 892,415 rows x 143 columns (~1.9 GB)
- Supports `--download-only` (raw CSVs only) and `--features-only` (rebuild from cached CSVs)

---

## Model Selection

### Chosen: LightGBM
- Best for tabular data with mixed feature types
- Handles NaN natively (important — futures features are NaN before 2019-09)
- Fast training (~10-15s per model on 624K rows)
- Built-in feature importance
- Strong regularization options for weak financial signals

### Rejected
- **LSTM**: Better for sequence patterns, but LightGBM should be tried first (simpler, faster, interpretable)
- **Transformer**: Overkill for this data size, harder to regularize
- **TCN**: Less proven in financial applications

---

## Training Results

### Iteration 1: Baseline
- Single target (direction_1 = 5min), long-only, basic params
- Result: acc=51.3%, Sharpe=0.99, very early stopping (38 iterations)
- Walk-forward inconsistent (Sharpe -0.72 to +4.08)
- **Lesson**: Model finds weak signal, needs tuning and more targets

### Iteration 2: Multi-target Exploration
Trained all 5 direction horizons + 3 trend models + big move model.

**Key findings per target:**

| Target | AUC | WF Sharpe (mean+-std) | WF All Positive? | Verdict |
|--------|-----|----------------------|-------------------|---------|
| direction_1 (5m) | 0.5208 | 1.33+-1.83 | No (1 negative) | Weak, inconsistent |
| **direction_6 (30m)** | **0.5291** | **1.91+-0.94** | **Yes (5/5)** | **Best model** |
| direction_12 (1h) | 0.5273 | 0.42+-1.71 | No (2 negative) | Inconsistent |
| direction_36 (3h) | 0.5364 | -1.20+-3.23 | No (4 negative) | Unreliable |
| direction_288 (24h) | 0.4846 | -6.56+-23.97 | No | Useless (barely learns) |
| trend_up_12 | 0.5330 | -0.95+-3.87 | No | Poor |
| trend_up_36 | 0.5352 | -1.54+-5.77 | No | Poor |
| trend_up_288 | 0.5311 | -8.37+-23.62 | No | Completely failed |
| big_move_12 | 0.8199 | -0.10+-3.61 | No | High AUC but class imbalance drift |

**Winner: target_direction_6 (30min)**
- Only model with ALL 5 walk-forward splits showing positive Sharpe
- Lowest Sharpe standard deviation (0.94) = most consistent
- Best walk-forward Sharpe (1.91)
- Iteration 2 overlapping backtest: long-short margin=0.04 showed +2274% return, Sharpe 4.16

**Top features for direction_6:**
1. volatility_kcp (Keltner Channel Percentage)
2. momentum_rsi
3. hour_of_day
4. trend_stc (Schaff Trend Cycle)
5. volatility_dcp (Donchian Channel Percentage)
6. momentum_tsi (True Strength Index)
7. momentum_pvo_hist
8. price_sma200_dist
9. volume_nvi
10. trend_adx

### Iteration 3: Realistic Backtesting (Critical Discovery)

Focused exclusively on direction_6. Introduced:
- Non-overlapping trades (every 6 candles = every 30min, no double-counting)
- Transaction costs (0.08% round-trip fee)
- 6 hyperparameter configs tested
- Feature selection (top-50, top-30, top-20)
- 5-seed ensemble

**The critical finding: Transaction costs destroy all profits.**

All 6 configs showed similar results:
- AUC: 0.528-0.530 (consistent, real predictive power)
- Prediction spread: IQR ~0.055-0.066 (tightly clustered around 0.5)
- With 22K+ non-overlapping trades and 0.08% fee: **compounded fee drag = 0.9992^22000 ~ 0** (total wipeout)

Best realistic scenario (A_baseline, long-only, margin=0.06):
- 353 trades over 16 months (~0.7/day)
- 44.48% win rate (after fees)
- Gross return: +22.21%
- Net return: **-7.85%** (fees of 353 x 0.08% = 28% exceed the 22% gross profit)

**Why iteration 2 results were misleading:**
1. Overlapping trades inflated results (each price move counted 6x)
2. No transaction costs applied
3. Compounding on every 5m candle (not realistic trading frequency)

---

## Key Lessons Learned

### 1. Overlapping backtests are dangerously misleading
The iteration 2 backtest applied 30min returns to every 5m candle, making each price movement count 6 times. This artificially inflated Sharpe ratios and returns. Always use non-overlapping trades for realistic evaluation.

### 2. Transaction costs are the dominant factor
At AUC ~0.53, the model's edge per trade is approximately 0.01-0.02%. With 0.08% round-trip fees, you need ~4-8x more edge than the model provides to break even. High-frequency trading on thin edges requires either:
- Much stronger signal (AUC > 0.55 at minimum)
- Much lower fees (maker limit orders at 0.02% instead of taker 0.04%)
- Much fewer trades (only trade highest-confidence predictions)

### 3. Fee math for this setup
- 0.08% RT fee per trade
- N trades compounded: portfolio multiplied by 0.9992^N
- 1,000 trades: 0.9992^1000 = 0.45 (55% fee drag)
- 5,000 trades: 0.9992^5000 = 0.018 (98% fee drag)
- 10,000 trades: effectively 0

### 4. The model does have real predictive power
- AUC 0.529 is consistently above 0.5 across all walk-forward splits
- Gross returns are often positive (before fees)
- Feature importance is stable across configs and seeds
- The signal is real but **too weak** for the current fee structure

### 5. Longer horizons don't help
- 24h direction: model barely learns (18 iterations, 74/114 zero-importance features)
- 3h direction: inconsistent walk-forward, compounding artifacts
- The 30min horizon is the sweet spot between signal strength and trade frequency

### 6. Python 3.14 compatibility issue
- scipy DLL load fails: `ImportError: DLL load failed while importing _superlu`
- sklearn depends on scipy, so sklearn cannot be used
- Workaround: implement metrics manually (AUC-ROC, log-loss, accuracy, precision, recall, F1)

### 7. DART boosting provides no advantage
- Early stopping doesn't work with DART mode
- Training takes 10x longer (98s vs 10s)
- Same AUC as regular GBDT (0.5297 vs 0.5291)
- Not worth the complexity

---

## What To Try Next (Iteration 4+)

### Most promising approaches
1. **Lower fee assumption**: Use maker limit orders (0.02% per side = 0.04% RT instead of 0.08%). This cuts fee drag in half and could flip the long-only high-confidence scenario to profitable.
2. **Larger horizon with non-overlapping**: Try predicting 4h (48 candles) or 8h (96 candles) direction with non-overlapping trades. Fewer trades = less fee drag, and each trade has more time to overcome the fee.
3. **Threshold return target**: Instead of predicting direction (>0%), predict if the move exceeds a threshold (>0.3% or >0.5%). This filters for larger moves where fees are proportionally smaller.
4. **Multi-model gating**: Only trade when both direction_6 AND big_move_12 (AUC 0.82) agree. This could increase per-trade edge at the cost of fewer trades.
5. **Position sizing**: Kelly criterion — bet proportional to confidence. This leverages the model's calibration where higher probabilities do correlate with higher actual rates.

### Less promising (but possible)
- Feature engineering: current 116 features seem sufficient (only 3-4 zero-importance)
- Hyperparameter tuning: all 6 configs gave nearly identical AUC (0.528-0.530)
- Ensemble: marginal improvement (AUC 0.530 vs 0.529 single model)
- LSTM/deep learning: unlikely to extract significantly more signal from same features

---

## File Structure

```
gitrich/
  btc_indicators.py    # Data download + feature engineering pipeline
  train_model.py       # LightGBM training pipeline (iteration 3)
  FINDINGS.md          # This file
  .gitignore           # Excludes btc_data/, __pycache__, venv, etc.
  btc_data/            # (gitignored)
    raw_spot_klines.csv       # 892K rows, ~113 MB
    raw_futures_klines.csv    # 677K rows, ~80 MB
    btc_features_5m.csv       # 892K x 143, ~1.9 GB
    training_results.txt      # Latest iteration results
    btc_indicators.log        # Download/feature engineering log
    models/                   # Saved LightGBM model files (.txt)
```

## Dependencies

```
pandas, numpy, lightgbm, ta, requests
```

Note: sklearn/scipy cannot be used due to Python 3.14 DLL load issue. All metrics are implemented manually.
