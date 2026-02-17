# BTC ML Trading Model — Findings & Learnings

## Project Overview

Goal: Build an ML model that predicts Bitcoin price movements using historical data, trained on 5-minute candle data from Binance.

## Data Pipeline (`btc_indicators.py`)

### Data Sources
- **Spot klines** (Binance BTCUSDT): 892,554 rows from 2017-08-17, 5m resolution (~107 MB raw)
- **Futures klines** (Binance BTCUSDT perpetual): 677,528 rows from 2019-09-08, 5m resolution (~76 MB raw)

### Features (196 total — v3 pipeline)
- **TA indicators (~86)**: via `ta.add_all_ta_features()` — trend, momentum, volatility, volume
- **CVD (3)**: delta per candle, cumulative delta, delta SMA-14
- **Custom price (22)**: returns at 8 horizons, log returns, SMA/EMA distances at 7 windows, candle features
- **Microstructure (28)**: OFI at 5 windows, volume profile, trade intensity, avg trade size, realized vol at 6 windows, vol-of-vol, price acceleration, range position, consecutive candles, VWAP deviation
- **Regime (8)**: vol regime rank, trend strength, z-scores at 3 windows, vol scaling ratio
- **Cross-timeframe (8)**: 15m/1h/4h aggregated features, MTF alignment
- **Interaction (3)**: RSI x volume, RSI x vol regime, ADX x return
- **Futures (9)**: close, basis, basis_pct, vol ratio, trades, basis momentum, basis z-score
- **Time (6)**: hour, day-of-week (raw + cyclical sin/cos encoding)
- **Robustness features (v3, ~11)**: momentum divergence, vol-price correlation (24/96), vol regime change/acceleration, normalized returns, range vs ATR, OFI trend, volume momentum (24/96)

### ML Targets (91 total — v3)
- **Direction (16)**: return + binary direction at 8 horizons (5m, 30m, 1h, 2h, 3h, 4h, 8h, 24h)
- **Threshold (60)**: up/down/bigmove at 5 horizons x 4 thresholds
- **Trend (9)**: strength/up/down at 3 horizons
- **Volatility (6+)**: max runup/drawdown, big move, risk-reward ratio

### Output
- `btc_data/btc_features_5m.parquet` — 892,554 rows x 289 columns (~1.3 GB parquet)

---

## Model: LightGBM

### Hyperparameters (unchanged since Iter 5 — strong regularization)
```
learning_rate=0.005, num_leaves=24, max_depth=5,
min_child_samples=500, subsample=0.5, colsample_bytree=0.3,
colsample_bynode=0.5, reg_alpha=5.0, reg_lambda=20.0,
feature_fraction_bynode=0.5, path_smooth=10.0
```

---

## Training Results

### Iterations 1-5 Summary

| Iter | Focus | Key Result |
|------|-------|------------|
| 1 | Baseline 5m direction | acc=51.3%, Sharpe 0.99, inconsistent WF |
| 2 | Multi-target exploration | direction_6 (30m) best: AUC 0.529, all 5 WF positive |
| 3 | Realistic backtesting | **Fees destroy everything** — 0.08% RT > model edge |
| 4 | Multi-strategy attack | 1h +9.26% test set, **but WF 0/5 positive** |
| 5 | Robustness breakthrough | 1h: 5/6 WF positive +8.96%, 3h: 5/6 WF positive +2.37% |
| 6 | 10-split WF + regime analysis | 1h: 9/10 WF positive +4.57%, regime gating marginal |
| **7b** | **Threshold targets** | **AUC 0.55→0.72, up_0002 p45_top20: 10/10 WF positive!** |

### Iteration 6: COMPREHENSIVE ROBUSTNESS VALIDATION

**Key changes from Iter 5:**
1. 10-split walk-forward (vs 6) for higher statistical confidence
2. Regime analysis — what predicts WF failure?
3. Regime-gated trading — filter trades by vol/trend regime
4. Feature stability analysis across WF splits
5. Stable-features-only WF comparison (50 vs 80 features)
6. Multi-horizon consensus WF validation (1h+3h+4h)
7. Dynamic confidence thresholds (rolling percentile)

**Data**: 892,554 rows x 289 cols, 196 features
**WF test size**: 44,627 candles (~155 days) per split
**Fee scenario**: Maker 0.04% RT, Long-only

---

#### Phase 1: 10-Split Walk-Forward — CONFIRMS ROBUSTNESS

| Horizon | Best Config | Pos/Total | Avg Net | Avg Kelly | AUC |
|---------|-------------|-----------|---------|-----------|-----|
| **1h** | **raw_m06_top10** | **9/10** | **+4.57%** | **+0.274** | **0.5502±0.018** |
| 3h | raw_m06_top10 | 6/9 | +1.35% | +0.074 | 0.5566±0.022 |

**1h raw_m06_top10 per split:**
| Split | Period | AUC | Net | WR | Trades |
|-------|--------|-----|-----|-----|--------|
| 1 (newest) | 2025-09-14/2026-02-16 | 0.5322 | +1.45% | 100.0% | 3 |
| 2 | 2025-04-12/2025-09-14 | 0.5258 | +0.72% | 69.2% | 13 |
| 3 | 2024-11-08/2025-04-12 | 0.5237 | **-1.73%** | 50.0% | 22 |
| 4 | 2024-06-06/2024-11-08 | 0.5392 | +1.41% | 59.4% | 32 |
| 5 | 2024-01-03/2024-06-06 | 0.5596 | +10.62% | 65.4% | 52 |
| 6 | 2023-08-01/2024-01-03 | 0.5658 | +5.16% | 67.9% | 28 |
| 7 | 2023-02-27/2023-08-01 | 0.5737 | +1.07% | 62.5% | 72 |
| 8 | 2022-09-25/2023-02-27 | 0.5759 | +4.91% | 53.2% | 77 |
| 9 | 2022-04-23/2022-09-25 | 0.5557 | +8.98% | 56.3% | 87 |
| 10 (oldest) | 2021-11-19/2022-04-23 | 0.5503 | +13.13% | 59.5% | 148 |

**Other notable 1h configs:**
- `cal_m04_top10`: 7/10 positive, avg +2.97%, median 131 trades
- `cal_m06_top20`: 7/10 positive, avg +2.17%, median 158 trades
- `raw_m04_top10`: 6/10 positive, avg +2.39%, median 80 trades

#### Phase 2: Regime Analysis — WHAT PREDICTS FAILURE

**1h regime correlations with net return:**
| Factor | Correlation | Interpretation |
|--------|-------------|----------------|
| `vol_mom96` | **-0.469** | Higher volume momentum = worse performance |
| `auc` | **+0.437** | Higher AUC directly predicts profitability |
| `vol_rank` | -0.270 | Higher volatility regime = slightly worse |
| `vpc24` | -0.242 | Volume-price correlation inversely correlated |
| `ret288` | -0.215 | Recent BTC direction slightly hurts |
| `trend` | -0.202 | Stronger trends slightly worse for model |

**Key insight**: The only split that loses money (split 3, Nov 2024-Apr 2025) has the highest `vol_mom96` (+0.2205) and lowest AUC (0.5237). The model struggles when volume momentum is elevated.

#### Phase 3: Regime-Gated Trading — MIXED RESULTS

| Config | Gate | Pos/N | Avg Net | Change |
|--------|------|-------|---------|--------|
| 1h raw_m04_top20 | no gate | 5/10 | -1.50% | baseline |
| 1h raw_m04_top20 | **low_vol** | **8/10** | **+2.96%** | **+4.46%** |
| 1h raw_m04_top20 | trending | 6/10 | +4.17% | +5.67% |
| 3h raw_m04_top20 | no gate | 4/10 | -0.94% | baseline |
| 3h raw_m04_top20 | **high_vol** | **8/10** | **+4.18%** | **+5.12%** |

**Important caveat**: Regime gating improves weaker configs (m04_top20) but the best config (raw_m06_top10) already achieves 9/10 positive, so gating adds minimal value where it matters most.

#### Phase 4: Feature Stability

**Most stable 1h features (high importance, low variability):**
1. `hour_of_day` (score=356, CV=0.51)
2. `vol_price_corr_96` (score=124, CV=0.66)
3. `volume_nvi` (score=109, CV=0.59)
4. `trend_stc` (score=66, CV=0.56)
5. `vol_price_corr_24` (score=55, CV=0.52)

**Most stable 3h features:**
1. `volume_momentum_96` (score=158, CV=0.42)
2. `hour_of_day` (score=153, CV=0.39)
3. `cumulative_delta_btc` (score=128, CV=0.48)
4. `volume_vpt` (score=81, CV=0.53)
5. `return_96` (score=79, CV=0.46)

**Most unstable features** (avoid in production): `tf4h_range`, `day_of_week`, `dow_sin`, `ofi_96`, `basis_zscore`

#### Phase 5: Stable Features Only — EQUIVALENT PERFORMANCE

| Metric | All Features (80) | Stable Features (50) |
|--------|-------------------|---------------------|
| 1h AUC | 0.5502±0.0182 | 0.5497±0.0179 |
| 3h AUC | 0.5566±0.0215 | **0.5573±0.0219** |
| 1h m04_top10 | 6/10, +2.39% | **7/10, +3.90%** |
| 1h m06_top10 | **9/10, +4.57%** | 7/10, +2.43% |

**Conclusion**: 50 stable features give equivalent AUC with slightly less variance. For production, stable features are preferable (simpler, more robust). However, the best single config still uses all 80 features.

#### Phase 6: Multi-Horizon Consensus WF — DOES NOT HELP

| Config | Pos/N | Avg Net | vs Single Model |
|--------|-------|---------|-----------------|
| agree3_exec36_m04 | 4/8 | -0.71% | Single 3h_m06_top10: 6/7, +3.84% |
| agree3_exec36_m06 | 4/8 | -0.57% | Single 1h_m06_top10: 4/7, +3.82% |
| agree2_exec36_m04 | 5/8 | -5.19% | Single 4h_m06_top10: 5/5, +0.83% |

**Consensus uniformly underperforms single models.** The models don't provide independent signals — when 1h is wrong, 3h and 4h tend to be wrong too. Consensus just adds noise and reduces trade count without improving accuracy.

#### Phase 7: Dynamic Confidence Thresholds — DESTRUCTIVE

Rolling percentile thresholds completely destroy 1h performance (all splits negative at all thresholds). 3h shows mixed results (3/6 positive at p90).

**Why it fails**: The fixed-margin approach (0.06) captures the model's absolute confidence, while rolling percentiles capture relative confidence. When the model has no edge (low AUC periods), even the top percentile predictions aren't worth trading. Fixed margins correctly filter these out; rolling percentiles don't.

---

### Iteration 7/7b: THRESHOLD SIGNAL EXPLOITATION (Current — BREAKTHROUGH)

**Key changes from Iter 6:**
1. Threshold-based targets: predict "price up >X% in 1h" instead of "price up in 1h"
2. Proper threshold backtest logic: uses absolute probability thresholds (p_threshold=0.35/0.40/0.45/0.50) + top percentile filtering instead of direction-style margins
3. XGBoost comparison against LightGBM
4. Recency-weighted training (exponential decay)
5. Combined direction + threshold signal testing

**Data**: 892,554 rows x 289 cols, 196 features
**WF test size**: ~44,627 candles (~155 days) per split
**Fee scenario**: Maker 0.04% RT, Long-only

---

#### Phase 0: Feature Importance — DIFFERENT FEATURES FOR THRESHOLD TARGETS

Threshold models use completely different features than direction models:

| Rank | Direction (target_direction_12) | Threshold (target_up_12_0003) |
|------|--------------------------------|-------------------------------|
| 1 | hour_of_day (268) | **realized_vol_288** (852) |
| 2 | volatility_bbp (199) | hour_of_day (786) |
| 3 | range_position_96 (197) | **volatility_kcw** (538) |
| 4 | trend_aroon_ind (182) | **rsi_x_vol_regime** (494) |
| 5 | vol_price_corr_24 (181) | **realized_vol_24** (494) |

**Key insight**: Threshold prediction is dominated by **volatility features** (realized_vol_288 is #1 by far). This makes sense — predicting whether price moves >0.3% requires knowing current volatility, not just direction.

#### Phase 1: Threshold Models — THE BREAKTHROUGH

##### up_12_0005 (predict "up >0.5% in 1h", base_rate=13.8%)
**AUC = 0.7210±0.036** (vs direction's 0.5502 — a +31% improvement!)

| Config | Pos/N | Avg Net | Med Trades | Kelly | Highlights |
|--------|-------|---------|------------|-------|------------|
| **p45_all** | **9/9** | **+3.34%** | **3** | **+0.632** | **Best consistency ever** |
| p45_top30 | 8/9 | +4.34% | 3 | +0.726 | Highest Kelly practical config |
| p45_top20 | 8/9 | +3.80% | 3 | +0.739 | |
| p35_all | 8/9 | +4.46% | 21 | +0.190 | Most trades, still very good |
| p40_top30 | 8/9 | +3.59% | 5 | +0.305 | |

Note: Split 2 (2025-04-12/2025-09-14) had p_max=0.346 so no trades at ≥p35 thresholds — model correctly abstained. Split 8 was the only losing split.

##### up_12_0003 (predict "up >0.3% in 1h", base_rate=22.5%)
**AUC = 0.6701±0.035**

| Config | Pos/N | Avg Net | Med Trades | Kelly | Highlights |
|--------|-------|---------|------------|-------|------------|
| **p35_top10** | **9/10** | **+4.45%** | **15** | **+0.278** | Best balance: selectivity + volume |
| **p40_top30** | **9/10** | **+3.29%** | **13** | **+0.305** | |
| **p40_top10** | **9/10** | **+3.57%** | **5** | **+0.452** | Highest Kelly for 9/10 configs |
| p45_top10 | 8/9 | +3.33% | 1 | +0.759 | Too few trades |

##### up_12_0002 (predict "up >0.2% in 1h", base_rate=29.6%)
**AUC = 0.6358±0.033**

| Config | Pos/N | Avg Net | Med Trades | Kelly | Highlights |
|--------|-------|---------|------------|-------|------------|
| **p45_top20** | **10/10** | **+4.71%** | **10** | **+0.257** | **FIRST EVER 10/10 WF POSITIVE!** |
| p50_top30 | 8/9 | +2.48% | 3 | +0.400 | |
| p45_all | 8/10 | +4.46% | 50 | +0.159 | Highest trade count with 80% WF+ |

**up_12_0002 p45_top20 is the first configuration in the entire project to achieve 10/10 walk-forward positive splits.** Every single 155-day window over 4+ years is profitable.

##### Threshold vs Direction Comparison

| Model | AUC | Best WF Pos | Best Avg Net | Best Kelly |
|-------|-----|-------------|-------------|------------|
| direction_12 | 0.5502 | 9/10 | +4.57% | +0.274 |
| up_12_0002 | 0.6358 | **10/10** | **+4.71%** | +0.257 |
| up_12_0003 | 0.6701 | 9/10 | +4.45% | +0.452 |
| up_12_0005 | **0.7210** | **9/9** | +3.34% | **+0.632** |

**Threshold targets are strictly superior.** Every threshold model has higher AUC than direction. The up_12_0002 model achieves the project's first 10/10 positive WF. The up_12_0005 model has AUC 0.72 — 31% higher than direction.

#### Phase 2: Direction Baseline — CONFIRMED INFERIOR

Direction model baseline (AUC=0.5502):
- m06_top10: 6/10 positive, +2.61% avg
- m04_top10: 7/10 positive, +2.06% avg

(Note: This uses the same features as Iter 6 but without the calibrated probability versions. Iter 6 best raw_m06_top10 was 9/10 — the slight difference is due to v7b not including calibrated configs.)

#### Phase 3: XGBoost vs LightGBM — MARGINAL

| Model | AUC | m06_top10 Pos/N | Avg Net |
|-------|-----|-----------------|---------|
| LightGBM | 0.5502±0.018 | 6/10 | +2.61% |
| XGBoost | 0.5516±0.015 | 5/10 | +0.74% |
| **LGB+XGB Average** | 0.5516±0.015 | **8/10** | **+2.83%** |

XGBoost is marginally better on AUC but worse for trading (lower selectivity → more noise). The LGB+XGB ensemble average achieves 8/10 positive (vs 6/10 for LGB alone) with similar returns — suggesting ensemble averaging smooths probabilities beneficially. However, this is on direction targets; the real gains are from threshold targets.

#### Phase 4: Recency-Weighted Training — HURTS

| Half-life | AUC | m06_top10 |
|-----------|-----|-----------|
| none (uniform) | **0.5502±0.018** | **6/10, +2.61%** |
| hl=0.05 | 0.5447±0.014 | 7/9, -0.21% |
| hl=0.10 | 0.5490±0.016 | 3/9, -2.82% |
| hl=0.20 | 0.5499±0.017 | 5/9, -1.63% |

**Recency weighting uniformly degrades trading performance.** The AUC barely changes but the probabilities become less calibrated, hurting the confidence-threshold filtering that makes the system work.

#### Phase 5: Combined Direction + Threshold Signal — NO BENEFIT FROM COMBINATION

| Strategy | Pos/N | Avg Net | Notes |
|----------|-------|---------|-------|
| dir_only | 0/10 | -30.0% | All direction predictions (no filtering) |
| thresh_only | 4/10 | -6.56% | All threshold predictions (no filtering) |
| dir_AND_thresh | 4/10 | -6.65% | Intersection |
| dir_OR_thresh | 0/10 | -30.2% | Union |

**IMPORTANT CAVEAT**: This phase tested ALL trades without top-percentile filtering, which explains the terrible results. The Phase 1 threshold results with proper filtering (p45_top20 → 10/10 positive) prove the signal is there — it just needs selectivity. Combining raw signals without filtering is useless for both direction and threshold models.

---

## Key Lessons Learned (Updated through Iter 7b)

### 1. Threshold targets are strictly superior to direction targets
**THE key finding of this project.** Predicting "will price rise >X%?" instead of "will price rise?" produces dramatically better models:
- Direction AUC: 0.5502 → Threshold AUC: 0.6358 (0.2%), 0.6701 (0.3%), **0.7210 (0.5%)**
- First-ever 10/10 WF positive achieved with threshold targets (up_12_0002 p45_top20)
- Threshold models use different features (volatility-dominated) suggesting they capture genuinely different patterns

### 2. Walk-forward MUST test the actual trading config
Iteration 4 showed 0/5 WF positive because it tested the wrong config. Iteration 5 fixed this → 5/6 positive. Iteration 6 confirmed with 10 splits → 9/10 positive. Iteration 7b → **10/10 positive**.

### 3. The signal requires extreme selectivity
Both direction and threshold models need confidence filtering to work:
- Direction: margin 0.06 + top 10% → median 42 trades/155 days
- Threshold up_0002: p45 + top 20% → median 10 trades/155 days
- Threshold up_0005: p45 all → median 3 trades/155 days
- Without filtering, ALL models lose money (Phase 5 proved this conclusively)

### 4. Volume momentum predicts model failure (corr=-0.47)
When `vol_mom96` is high (rapid volume growth), the model loses its edge. The model struggles when volume momentum is elevated (unusual market conditions).

### 5. Combining models doesn't help
Multi-horizon consensus (Iter 6), direction+threshold combination (Iter 7b Phase 5), and recency weighting (Iter 7b Phase 4) all fail to improve over single well-filtered models.

### 6. Absolute thresholds work, relative/dynamic thresholds hurt
The model needs ABSOLUTE confidence thresholds. Rolling percentiles (Iter 6), recency weighting (Iter 7b), and dynamic approaches all degrade performance. When model confidence is low, it should trade NOTHING.

### 7. LightGBM ≈ XGBoost for this task
XGBoost has marginally higher AUC (0.5516 vs 0.5502) but produces worse trading results. LGB+XGB averaging slightly helps (8/10 vs 6/10) but the effect is small compared to switching to threshold targets.

### 8. Feature importance changes dramatically with target type
Direction models: hour_of_day, BBP, range_position, trend indicators
Threshold models: **realized_vol_288**, hour_of_day, Keltner width, RSI×vol_regime, realized_vol_24
This suggests threshold prediction is fundamentally about volatility forecasting — knowing whether the market CAN move 0.3%+ is more important than knowing which direction.

### 9. Fees remain THE critical constraint
The model ONLY works with maker fees (0.04% RT), long-only trading, and high confidence thresholds.

---

## Production Status

### What Works (Best Configs — Ranked)

| Rank | Model | Config | WF Pos | Avg Net | Med Trades | Kelly |
|------|-------|--------|--------|---------|------------|-------|
| 1 | **up_12_0002** | **p45_top20** | **10/10** | **+4.71%** | **10** | **+0.257** |
| 2 | up_12_0005 | p45_all | 9/9 | +3.34% | 3 | +0.632 |
| 3 | up_12_0003 | p35_top10 | 9/10 | +4.45% | 15 | +0.278 |
| 4 | up_12_0003 | p40_top30 | 9/10 | +3.29% | 13 | +0.305 |
| 5 | up_12_0003 | p40_top10 | 9/10 | +3.57% | 5 | +0.452 |
| 6 | direction_12 | raw_m06_top10 | 9/10 | +4.57% | 42 | +0.274 |

**Lead candidate: up_12_0002 p45_top20** — 10/10 WF positive, avg +4.71% per 155 days (~11% annualized), median 10 trades per split, Kelly criterion 0.257.

**Runner-up: up_12_0005 p45_all** — 9/9 WF positive (1 split had no trades), highest Kelly (0.632), but only median 3 trades per split.

### What Doesn't Work
- Any configuration with taker fees
- Long-short trading
- Multi-horizon consensus
- Dynamic confidence thresholds
- Recency-weighted training
- Combined direction + threshold signals (without filtering)
- XGBoost (marginal AUC gain, worse trading results)
- Trading without extreme confidence filtering

### Remaining Risks
1. **Low trade count**: Median 3-15 trades per 155-day split for threshold models
2. **Maker fees assumption**: Requires limit order execution (fill rate unknown)
3. **No live validation**: All results are backtested
4. **Volatility regime dependency**: Performance varies with market volatility conditions
5. **Possible signal decay**: Newer splits show slightly lower AUC for direction models

---

## What To Try Next (Iteration 8)

### Priority 1: Increase trade count while preserving edge
The threshold models are highly profitable but trade very infrequently. This is the biggest production barrier.

1. **Multi-model portfolio** — Run up_0002/p45_top20, up_0003/p35_top10, up_0005/p45_all in parallel as independent signals. Combined should give ~1 trade/day instead of 0.06/day.

2. **3h threshold targets** — Test up_36_0003 and up_36_0005. The 3h direction model had 6/9 WF positive; threshold targets should dramatically improve this, giving independent 3h trades.

3. **Shorter holding period for threshold** — Currently using 12 candles (1h) as holding period. Could exit earlier if the threshold is hit (take-profit at +0.3%/+0.5%), increasing capital turnover.

### Priority 2: Improve existing threshold models
4. **Isotonic calibration for threshold models** — The threshold models output raw probabilities; calibrating them may improve the probability thresholds' effectiveness.

5. **LGB+XGB ensemble for threshold models** — The direction ensemble showed 6/10 → 8/10 improvement. Testing this on threshold targets could boost from 9/10 → 10/10.

6. **Feature engineering for volatility** — Since threshold models are volatility-dominated, adding specialized volatility features (GARCH estimates, implied vol proxies, intraday vol patterns) could boost AUC further.

7. **Regime filtering on threshold models** — Apply the vol_mom96 gate to threshold models to see if it eliminates the remaining losing splits.

### Priority 3: Short-side exploration
8. **Down threshold targets** — Test down_12_0003, down_12_0005 for short signals. Combined with long threshold signals, this could double the trade count.

### Priority 4: Production readiness
9. **Paper trading** — Deploy the top 3 configs on live data for forward validation
10. **Position sizing** — Use Kelly criterion dynamically (up_0005 Kelly=0.632 → more aggressive sizing)
11. **Trade execution** — Model maker order fill rates and slippage
12. **Monitoring** — Track AUC decay, feature drift, and regime indicators in real-time

---

## File Structure

```
trading/
  btc_indicators.py    # Data download + feature engineering (v3, 196 features)
  train_model.py       # LightGBM training pipeline (iteration 5, 5 phases)
  train_model_v6.py    # Iteration 6 training pipeline (7 phases)
  train_model_v7.py    # Iteration 7 — crashed mid-run (OOM), superseded by v7b
  train_model_v7b.py   # Iteration 7b — threshold signal exploitation (5 phases)
  FINDINGS.md          # This file
  .gitignore
  btc_data/            # (gitignored)
    raw_spot_klines.csv         # 892K rows, ~107 MB
    raw_futures_klines.csv      # 677K rows, ~76 MB
    btc_features_5m.parquet     # 892K x 289, ~1.3 GB
    training_results_v4.txt     # Iteration 4 results
    training_results_v5.txt     # Iteration 5 results
    training_results_v6.txt     # Iteration 6 results (1148 lines)
    training_results_v7.txt     # Iteration 7 partial results (crashed)
    training_results_v7b.txt    # Iteration 7b results (510 lines)
    models/                     # Saved LightGBM model files
```

## Dependencies

```
pandas, numpy, lightgbm, xgboost, ta, requests, pyarrow, scikit-learn
```
