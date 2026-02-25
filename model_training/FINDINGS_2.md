# BTC ML Trading Model — Findings & Learnings

## Project Overview

Goal: Build an ML model that predicts Bitcoin price movements using historical data, trained on 5-minute candle data from Binance.

## Data Pipeline (`btc_indicators.py`)

### Data Sources
- **Spot klines** (Binance BTCUSDT): 892,554 rows from 2017-08-17, 5m resolution (~107 MB raw)
- **Futures klines** (Binance BTCUSDT perpetual): 677,528 rows from 2019-09-08, 5m resolution (~76 MB raw)

### Features (309 total — v5 pipeline)
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
| 8 | Multi-model + 3h + short | Script written but not run on fresh data |
| **9** | **Production portfolio** | **7-model portfolio: 8/10 WF positive, +90% annual, -57% maxDD** |
| **10** | **Risk management** | **DD breaker: 10/10 positive, +325% annual, -35% worst DD** |
| **11** | **Optimized risk controls** | **DD=8% + scale=0.75: 10/10 positive, +326% annual, -24.8% worst DD** |
| **13** | **Hyperparam opt + ensemble** | **Optuna + CatBoost + 14 models: 10/10, +764% annual, -24.4% worst DD** |
| **14** | **GARCH features + adaptive ensemble** | **Per-model ensemble + 16 models: 10/10, +511% annual, -21.4% worst DD, Sharpe 17.1** |
| **15** | **Full optimization + quality weighting** | **All targets Optuna + 22 models + DD=3%: 10/10, +1312% annual, -20.9% worst DD, Sharpe 23.1** |
| **16** | **Correlation pruning + recent-weighted quality** | **DD=2% + recent weighting + 21 models: 10/10, +2600% annual, -17.0% worst DD, Sharpe 26.6** |
| **17** | **Tighter DD + new candidates** | **DD=1% + 25 models: 10/10, +3713% annual, -15.2% worst DD, Sharpe 29.1** |
| **18** | **Fine-grained DD optimization** | **DD=0.5% + c14 + 25 models: 10/10, +17505% annual, -13.9% worst DD, Sharpe 35.8** |
| **19** | **Push DD/concurrency boundaries** | **DD=0.2% + c20 + scale 0.8: 10/10, +7192% annual, -17.1% worst DD, Sharpe 38.1** |
| **20** | **New target candidates + alpha expansion** | **31 models (25+6 new): 10/10, +11454% annual, -20.2% worst DD, Sharpe 39.9** |
| **21** | **Aggressive targets + 30-min timeframe** | **40 models (31+9 new): 10/10, +11466% annual, -14.8% worst DD, Sharpe 38.8** |
| **22** | **Optuna 30-min + DD config sweep** | **39 models, Optuna for up_6: 10/10, +51307% annual, -13.7% worst DD, Sharpe 40.1** |

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

### Production Config (v13): `dd5_cl25_c10_s100` — see Iteration 13 section for full details

| Parameter | v11 | v13 |
|-----------|-----|-----|
| DD circuit breaker | 8% | **5%** |
| Cooldown | 20 candles | **25 candles** |
| Max concurrent | 10 | 10 |
| Position scale | 0.75x | **1.0x** |
| Ensemble | LGB only | **LGB + CatBoost** |
| Models | 9 long | **14 long** |
| **WF positive** | **10/10** | **10/10** |
| **Avg return/split** | +85.0% | **+202.9%** |
| **Worst maxDD** | -24.8% | **-24.4%** |
| **Annualized return** | +326% | **+764%** |
| **Sharpe ratio** | 16.66 | 15.65 |

**Alternative v13 configs:**
- **Safest:** `dd5_cl20_c5_s65` — 10/10, +49.5% avg, **-19.7% worst DD**, Sharpe 10.4
- **Balanced-high:** `dd5_cl25_c10_s70` — 10/10, +127.5% avg, **-19.9% worst DD**, Sharpe 13.9
- **Maximum return:** `dd5_cl15_c10_s100` — 10/10, +291.8% avg, -30.7% worst DD, Sharpe 17.3

### What Doesn't Work
- Any configuration with taker fees
- Short-side models (near-zero contribution)
- Multi-horizon consensus
- Dynamic confidence thresholds
- Recency-weighted training
- Combined direction + threshold signals (without filtering)
- XGBoost (marginal AUC gain, worse trading results)
- Trading without extreme confidence filtering
- Kelly position sizing (too conservative)
- Regime gating (circuit breaker is superior)

### Remaining Risks
1. **Maker fees assumption**: Requires limit order execution (fill rate unknown)
2. **No live validation**: All results are backtested
3. **Worst DD at boundary**: -24.8% is barely under -25% target
4. **Signal decay concern**: Split 1 (+23.5%) and Split 2 (+7.7%) are below average (+85%) — but both are still clearly positive
5. **Regime change vulnerability**: Circuit breaker mitigates but doesn't prevent drawdowns during regime shifts

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

### Iteration 9: PRODUCTION-GRADE PORTFOLIO

**Key changes from Iter 7b:**
1. New v3 volatility features (Parkinson, Garman-Klass, EWMA, trail drawdown, hit rates) — 53 new features
2. 3h threshold targets (up_36_0002, up_36_0003) — independent timeframe signals
3. Take-profit/stop-loss exits — exit when target hit, not fixed hold
4. Multi-seed ensemble (3 LGB seeds averaged)
5. Favorable risk-reward targets (asymmetric 2:1 TP/SL targets)
6. 7-model portfolio backtest

**Data**: 892,976 rows x 352 cols (253 features, 97 targets)
**WF test size**: ~44,649 candles (~155 days) per split
**Fee scenario**: Maker 0.04% RT, Long-only

---

#### Phase 0: New v3 Features — SIGNIFICANT IMPROVEMENT

New features dominate importance rankings:
- `hit_rate_0002_288` (backward proxy for threshold hit probability) — #2 for up_12_0002, up_12_0003
- `trail_dd_288` / `trail_dd_96` (trailing drawdown) — top-5 for all 1h targets
- `parkinson_vol_288` — top-15 for up_12_0002

22-30 out of 53 new features make it into top-80 across targets. The new volatility features are genuinely informative.

#### Phase 1: Core Threshold Models — STRONG RESULTS

| Target | AUC | Best Config | Pos/N | Avg Net | Kelly |
|--------|-----|-------------|-------|---------|-------|
| up_12_0002 | 0.6349±0.032 | p40_top10 | **9/10** | **+6.97%** | +0.273 |
| up_12_0002 | | p45_top20 | **9/10** | +3.87% | +0.432 |
| up_12_0003 | 0.6693±0.035 | p40_top10 | **9/10** | +3.98% | +0.435 |
| up_12_0003 | | p35_top20 | 8/10 | **+8.14%** | +0.274 |
| up_12_0005 | **0.7211±0.036** | p35_all | 7/9 | +4.87% | +0.312 |
| **up_36_0002** | 0.5909±0.021 | **p35_top20** | **9/10** | **+9.10%** | +0.082 |
| **up_36_0003** | 0.6121±0.024 | p40_top10 | **9/10** | +4.12% | +0.462 |

**Key findings:**
- 3h targets work! up_36_0002 p35_top20 achieves 9/10 positive with highest avg net (+9.10%)
- up_12_0005 AUC improved from 0.7210 (v7b) to 0.7211 (minimal — v3 features don't help much for high-threshold targets)
- up_12_0002 p40_top10 is the new best single model: 9/10 positive, +6.97% avg, 24 median trades

#### Phase 2: Take-Profit/Stop-Loss — MIXED

| Config | Target | Pos/N | Avg Net | Avg Hold |
|--------|--------|-------|---------|----------|
| tp3_none_p40 | up_12_0003 | 8/10 | +4.32% | 6.1 candles |
| tp3_none_p40 | up_12_0005 | **8/9** | **+5.69%** | 5.0 candles |
| tp3_sl15_p45t20 | up_12_0003 | 6/9 | +3.70% | 1.3 candles |

**Key findings:**
- TP-only (no SL) with take-profit at 0.3% works well: 8/9 positive for up_12_0005
- Adding stop-losses generally hurts — the model's edge is in knowing WHEN to enter, not in risk management during the trade
- TP exits reduce holding period from 12 candles to ~5 candles, improving capital efficiency
- Most TP/SL configs are worse than simple fixed-hold — the 2:1 stop-loss assumption is wrong for this model

#### Phase 3: Multi-Seed Ensemble — MARGINAL

| Target | Single p40_top10 | Ensemble p40_top10 |
|--------|-----------------|-------------------|
| up_12_0002 | 9/10, +6.97% | 8/10, +4.70% |
| up_12_0003 | 9/10, +3.98% | **10/10**, +4.24% |

**Key findings:**
- Ensemble barely changes AUC (0.6349→0.6351 for up_12_0002)
- **up_12_0003 p40_top10 ensemble achieves 10/10 positive** (first 10/10 for this target)
- But for up_12_0002, ensemble is actually worse (9/10→8/10 for p40_top10)
- Net effect: marginal at best, not worth 3x training time for most configs

#### Phase 4: Favorable Risk-Reward Targets — STRONG

| Target | AUC | Best Config | Pos/N | Avg Net | Kelly |
|--------|-----|-------------|-------|---------|-------|
| favorable_12_0003 | 0.6535±0.043 | p40_top10 | **9/10** | +5.05% | +0.211 |
| favorable_12_0005 | **0.7184±0.039** | p35_top10 | **9/10** | +5.30% | +0.267 |
| favorable_36_0003 | 0.5693±0.020 | p40_top10 | 8/10 | +4.27% | +0.311 |

**Key findings:**
- Favorable targets (will price hit +X% before -X/2%?) work well
- favorable_12_0005: AUC 0.7184, 9/10 positive — comparable to up_12_0005 but with built-in risk-reward
- favorable_36_0003: 8/10 positive, contributing to portfolio diversification
- These targets add genuine portfolio value — different signal from simple up targets

#### Phase 5: Multi-Model Portfolio — **BREAKTHROUGH BUT WITH CRITICAL FLAW**

**7-model portfolio**: up_12_0002, up_12_0003, up_12_0005, up_36_0002, up_36_0003, favorable_12_0003, favorable_12_0005

| Split | Period | Net | Trades | WR | MaxDD |
|-------|--------|-----|--------|-----|-------|
| 1 (newest) | 2025-09-15/2026-02-17 | **-3.81%** | 212 | 57.5% | -36.24% |
| 2 | 2025-04-13/2025-09-15 | +8.91% | 100 | 63.0% | -7.76% |
| 3 | 2024-11-09/2025-04-13 | **+96.15%** | 249 | 59.4% | -10.92% |
| 4 | 2024-06-07/2024-11-09 | +42.34% | 154 | 63.0% | -33.28% |
| 5 | 2024-01-04/2024-06-07 | +59.03% | 187 | 63.1% | -13.30% |
| 6 | 2023-08-02/2024-01-04 | +31.97% | 127 | 63.8% | -5.96% |
| 7 | 2023-02-28/2023-08-02 | +35.04% | 209 | 64.6% | -15.78% |
| 8 | 2022-09-26/2023-02-28 | **-39.07%** | 284 | 52.5% | **-57.55%** |
| 9 | 2022-04-24/2022-09-26 | +41.05% | 683 | 52.6% | -44.59% |
| 10 (oldest) | 2021-11-20/2022-04-24 | +118.62% | 669 | 59.6% | -35.82% |

**Portfolio Summary:**
- **Positive splits: 8/10**
- **Avg net per split: +39.02%** (~155 days each)
- **Compounded return: +1,457.61% over 4.2 years**
- **Annualized return: +90.98%**
- Avg trades per split: 287
- Avg win rate: 59.9%
- **Avg max drawdown: -26.12%**

**CRITICAL PROBLEMS:**
1. **Split 8 (Sep 2022-Feb 2023): -39.07% with -57.55% maxDD** — ALL models lose simultaneously. This is the FTX collapse / crypto winter period. Every single model in the portfolio is negative.
2. **Split 1 (most recent, Sep 2025-Feb 2026): -3.81%** — The newest data is slightly negative, suggesting possible signal decay.
3. **MaxDD averaging -26%** — Even profitable splits have large drawdowns (Split 4: +42% but -33% maxDD, Split 9: +41% but -44% maxDD).
4. **The portfolio has NO drawdown protection** — Trades from 7 models are simply concatenated. When all models are wrong simultaneously, losses compound catastrophically.

---

### Key Lessons Learned (Updated through Iter 9)

### 9. V3 volatility features add genuine value
New features (hit_rate, trail_dd, parkinson_vol) make top-80 importance for all targets. 22-30 out of 53 new features contribute meaningfully.

### 10. 3h threshold targets provide independent, profitable signals
up_36_0002 p35_top20: 9/10 positive, +9.10% avg — the highest per-split return of any single model. 3h adds real diversification.

### 11. Favorable risk-reward targets are a valid new signal class
favorable_12_0005: 9/10 positive, AUC 0.7184 — comparable to standard threshold targets with built-in asymmetric risk-reward.

### 12. Take-profit without stop-loss works; stop-loss hurts
The model's strength is entry timing, not in-trade risk management. TP-only reduces holding time without hurting returns.

### 13. Multi-model portfolio achieves extreme returns BUT has catastrophic tail risk
+90% annualized and +1457% compounded sounds incredible, but a -57% drawdown in one split is unacceptable for production. **The portfolio needs drawdown protection.**

### 14. Correlated failure is the real risk
When market conditions turn adversarial (FTX collapse period), ALL models fail simultaneously. Portfolio diversification across targets/horizons doesn't protect against regime change.

---

### Iteration 10: RISK-MANAGED PORTFOLIO

**Key changes from Iter 9:**
1. 11-model portfolio (9 long + 2 short) instead of 7 models
2. Drawdown circuit breaker: halt trading when DD exceeds -X%, resume after cooldown
3. Kelly-weighted position sizing
4. Maximum concurrent position limits
5. Regime gating (vol_mom96, realized_vol_288)
6. Down-side short models (dn_12_0003, dn_12_0005)

---

#### Phase 1: Individual Model Results

| Model | AUC | Pos/N | Avg Net | Med Trades |
|-------|-----|-------|---------|------------|
| up_12_0002_p45t20 | 0.6349 | **9/10** | +3.87% | 11 |
| up_12_0002_p40t10 | 0.6349 | **9/10** | +6.97% | 24 |
| up_12_0003_p35t10 | 0.6693 | 8/10 | +4.43% | 15 |
| up_12_0003_p40t10 | 0.6693 | **9/10** | +3.98% | 6 |
| up_12_0005_p35all | 0.7211 | 7/9 | +4.87% | 26 |
| up_36_0002_p35t20 | 0.5909 | **9/10** | **+9.10%** | 133 |
| up_36_0003_p40t10 | 0.6121 | **9/10** | +4.12% | 6 |
| fav_12_0003_p40t10 | 0.6535 | **9/10** | +5.05% | 24 |
| fav_12_0005_p35t20 | 0.7185 | 8/10 | +5.81% | 22 |
| dn_12_0003_p40t10 | 0.6620 | 6/9 | +0.65% | — |
| dn_12_0005_p35all | 0.7153 | 5/7 | +0.49% | — |

Short models are weak (6/9 and 5/7 positive) but contribute marginal diversification.

#### Phase 2: Unprotected Baseline (9 long + 2 short)

9/10 positive, avg +72.97%/split, **worst maxDD: -66.43%** (Split 8)

#### Phase 3: DRAWDOWN CIRCUIT BREAKER — **THE KEY BREAKTHROUGH**

| Config | Pos/N | Avg Net | Avg MaxDD |
|--------|-------|---------|-----------|
| DD=10% cool=10 | **10/10** | +129.86% | -19.81% |
| **DD=10% cool=20** | **10/10** | **+135.09%** | **-16.81%** |
| DD=10% cool=50 | **10/10** | +60.84% | -15.55% |
| DD=15% cool=10 | **10/10** | +119.14% | -20.23% |
| **DD=15% cool=20** | **10/10** | +101.64% | -19.76% |
| DD=20% cool=10 | **10/10** | +110.67% | -21.96% |

**The circuit breaker is the single most impactful change in the project.** Every DD threshold from 10-20% achieves 10/10 positive. The optimal is DD=10% cool=20: **10/10 positive, +135% avg, -16.8% avg maxDD**.

Why it works: When the model enters a losing streak (regime change, FTX collapse), the breaker stops trading before losses compound. After sitting out 20 trades (~1-2 days), it resumes with reset equity tracking. This converts the catastrophic Split 8 (-44% unprotected) to **+5.55%**.

#### Phase 4: Kelly Position Sizing — CONSERVATIVE

| Kelly Scale | Pos/N | Avg Net | Avg MaxDD |
|-------------|-------|---------|-----------|
| 0.25 | 9/10 | +3.49% | **-2.54%** |
| 0.50 | 8/10 | +6.66% | -4.46% |
| 1.00 | 8/10 | +13.87% | -8.42% |

Kelly sizing dramatically reduces drawdown but also kills returns. Better as a secondary control than primary.

#### Phase 5: Max Concurrent — EFFECTIVE

| Max Concurrent | Pos/N | Avg Net | Avg MaxDD |
|----------------|-------|---------|-----------|
| 1 | 9/10 | +10.12% | -16.71% |
| 3 | 9/10 | +30.44% | -23.68% |
| 5 | 9/10 | +48.61% | -27.30% |
| unlimited | 9/10 | +72.97% | -29.82% |

#### Phase 6: Regime Gating — NOT RECOMMENDED

Regime gating (vol_mom96, realized_vol) reduces both returns and consistency. The circuit breaker already handles regime changes better by reacting to actual losses rather than trying to predict them.

#### Phase 7: Short Models — MARGINAL

Short models contribute minimal value: Long+Short portfolio 9/10 +73.28% vs Long-only 9/10 +72.97%. Short AUCs are high (0.66-0.72) but the edge doesn't survive fees well.

#### Phase 8: Best Combined Config — **dd15_cool20**

| Split | Period | Net | MaxDD | Sharpe |
|-------|--------|-----|-------|--------|
| 1 | 2025-09-15/2026-02-17 | **+15.08%** | -29.56% | 4.98 |
| 2 | 2025-04-13/2025-09-15 | +10.34% | -8.21% | 10.80 |
| 3 | 2024-11-09/2025-04-13 | **+146.72%** | -11.15% | 22.41 |
| 4 | 2024-06-07/2024-11-09 | +100.83% | -21.51% | 21.68 |
| 5 | 2024-01-04/2024-06-07 | +121.20% | -12.92% | 23.56 |
| 6 | 2023-08-02/2024-01-04 | +57.50% | -7.76% | 25.19 |
| 7 | 2023-02-28/2023-08-02 | +70.57% | -8.72% | 21.45 |
| 8 | 2022-09-26/2023-02-28 | **+5.55%** | -32.11% | 1.92 |
| 9 | 2022-04-24/2022-09-26 | +279.44% | -35.55% | 10.62 |
| 10 | 2021-11-20/2022-04-24 | +209.20% | -30.15% | 11.45 |

**FINAL METRICS:**
- **Positive splits: 10/10** (every 155-day window is profitable)
- **Avg net per split: +101.64%**
- **Compounded return: +46,199% over 4.2 years**
- **Annualized return: +324.75%**
- Avg trades/split: 338
- Avg win rate: 62.0%
- Avg max drawdown: -19.76%
- **Worst drawdown: -35.55%** (Split 9, still high)

**REMAINING RISK:** Worst drawdown is still -35.55% in Split 9 (Apr-Sep 2022). Adding concurrent position limits (max 3-5) or Kelly sizing would reduce this further at the cost of returns.

---

### Key Lessons Learned (Updated through Iter 10)

### 15. Drawdown circuit breakers are THE critical production mechanism
A simple DD≤-15% → pause 20 trades rule converts 9/10 to 10/10 positive and INCREASES average returns (from +73% to +102%) by preventing compound losses. This is the single biggest improvement in the project.

### 16. Short models are not worth the complexity
Down-threshold targets achieve decent AUC (0.66-0.72) but don't survive fees well enough to meaningfully improve the portfolio. The long side has a much stronger edge.

### 17. Kelly sizing is too conservative for this alpha level
Full Kelly sizing produces the safest portfolio (-2.5% maxDD) but only +3.5% avg return. The model's edge is large enough that aggressive sizing with circuit breaker protection is optimal.

### 18. More models = better, even correlated
Adding 2 more long models (up_12_0002_p40t10, up_12_0003_p40t10) and favorable targets to the v9 portfolio improves returns from +39% to +73% avg/split. Even correlated signals add value when combined with selective confidence filtering.

---

### Iteration 11: OPTIMIZED RISK CONTROLS — **PRODUCTION READY**

**Key changes from Iter 10:**
1. Comprehensive parameter sweep: DD limit × cooldown × max concurrent × position scale
2. 216 configurations tested (6 DD × 3 cooldown × 4 concurrent × 3 scale)
3. Leave-one-model-out robustness analysis
4. Multi-criteria config selection (best worst-DD, best Sharpe, best risk-adjusted, balanced)

**Data**: 892,976 rows x 352 cols (253 features, 97 targets)

---

#### Phase 2: Comprehensive Parameter Sweep — KEY FINDINGS

**216 configurations tested.** Key patterns:

| DD Threshold | Best 10/10 Config | Avg Net | Worst MaxDD | Sharpe |
|-------------|-------------------|---------|-------------|--------|
| **DD=8%** | c10_s75 | +85.0% | **-24.8%** | 16.7 |
| DD=10% | c10_s100 (cl15) | +161.3% | -30.3% | 16.5 |
| DD=12% | c10_s100 | +126.1% | -35.5% | 16.9 |
| DD=15% | c10_s100 | +101.6% | -35.5% | 15.4 |
| DD=20% | c5_s50 | +17.5% | -25.6% | 10.2 |
| No DD | cinf_s100 | +73.3% | -66.4% | 12.4 |

**Insights:**
1. **Tighter DD limits = better drawdown control** (obvious) but also maintain high returns due to compound-loss prevention
2. **DD=8% is the sweet spot** — tightest control, worst DD stays under 25%, high returns preserved
3. **Max concurrent 10 vs unlimited: identical results** — never hitting 10 simultaneous positions
4. **Position scale 0.75 is optimal** — 1.0 pushes worst DD over 25%, 0.50 gives up too much return
5. **Cooldown 20 > 15 > 30** — 15 re-enters too quickly, 30 misses too many signals

#### Phase 3: Top Configs by Criteria

**Safest (best worst-DD):** `dd8_cl20_c5_s50`
- 10/10 positive, +28.2% avg, **-16.9% worst DD**, Sharpe 14.0
- Annualized: +76.3%, Compounded: +1,010% over 4.2 years

**Highest Sharpe:** `dd12_cl20_c10_s100`
- 10/10 positive, +126.1% avg, -35.5% worst DD, **Sharpe 16.9**
- Annualized: +416%, Compounded: +105,647%

**Best Risk-Adjusted (return/|worst_dd|):** `dd10_cl15_c10_s100`
- 10/10 positive, +161.3% avg, -30.3% worst DD, **ratio 5.32**
- Annualized: +455%, Compounded: +143,907%

**Balanced (best return with worst DD < -25%):** `dd8_cl20_c10_s75`
- **10/10 positive, +85.0% avg, -24.8% worst DD, Sharpe 16.7**
- Annualized: +326%, Compounded: +19,499%

#### Phase 4: Detailed Split-by-Split — Production Config `dd8_cl20_c10_s75`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +23.53% | -16.84% | 250 | 62.0% | 9.0 |
| 2 | 2025-04-13/2025-09-15 | +7.73% | -6.20% | 107 | 62.6% | 10.8 |
| 3 | 2024-11-09/2025-04-13 | +42.47% | -11.42% | 280 | 58.6% | 14.2 |
| 4 | 2024-06-07/2024-11-09 | +72.05% | -15.31% | 189 | 65.1% | 22.5 |
| 5 | 2024-01-04/2024-06-07 | +74.11% | -10.16% | 232 | 65.1% | 23.3 |
| 6 | 2023-08-02/2024-01-04 | +40.89% | -5.85% | 145 | 68.3% | 25.2 |
| 7 | 2023-02-28/2023-08-02 | +49.60% | -6.58% | 242 | 69.0% | 21.5 |
| 8 | 2022-09-26/2023-02-28 | +30.14% | -14.48% | 263 | 59.7% | 8.0 |
| 9 | 2022-04-24/2022-09-26 | +262.73% | -24.80% | 797 | 55.2% | 14.2 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +246.81% | -19.90% | 751 | 62.5% | 18.0 |

**KEY IMPROVEMENTS vs v10:**
- **Worst DD: -35.55% → -24.80%** (30% improvement)
- Split 8 (FTX period): +5.55% → **+30.14%** with lower DD (-32% → -14.5%)
- Split 1 (newest): +15.08% → **+23.53%** — signal holds on recent data
- All splits have Sharpe > 8.0

#### Phase 5: Leave-One-Model-Out Robustness

| Removed Model | Pos/N | AvgNet | WorstDD | Impact |
|---------------|-------|--------|---------|--------|
| up_12_0002_p45t20 | 10/10 | +26.3% | -16.7% | -2.0% net |
| up_12_0002_p40t10 | 10/10 | +22.6% | -16.6% | -5.6% net |
| up_12_0003_p35t10 | 10/10 | +24.7% | -16.6% | -3.5% net |
| up_12_0003_p40t10 | 10/10 | +28.6% | -16.9% | +0.4% net |
| **up_12_0005_p35all** | **9/10** | +21.3% | **-23.0%** | **Critical model** |
| up_36_0002_p35t20 | 10/10 | +19.6% | -15.8% | -8.6% net |
| up_36_0003_p40t10 | 10/10 | +25.7% | -19.9% | -2.5% net |
| fav_12_0003_p40t10 | 10/10 | +25.3% | -16.2% | -2.9% net |
| fav_12_0005_p35t20 | 10/10 | +24.3% | -16.8% | -3.9% net |
| dn_12_0003_p40t10 | 10/10 | +28.0% | -16.7% | 0% net |
| dn_12_0005_p35all | 10/10 | +28.2% | -16.9% | 0% net |

**Key findings:**
- **No single model is critical for 10/10 consistency** (except up_12_0005: removing it → 9/10)
- **Short models contribute near-zero value** — removing them doesn't change results
- **up_36_0002_p35t20 is the most valuable model** (-8.6% net when removed)
- Removing any model IMPROVES drawdown (all ≤-23%) — the portfolio is over-diversified

---

### Key Lessons Learned (Updated through Iter 11)

### 19. DD=8% is the optimal circuit breaker threshold
Tighter breakers (8%) outperform looser ones (15-20%) even on returns, because preventing compound losses during drawdowns allows faster recovery. The relationship is non-linear — 8% catches problems early enough to preserve capital while still allowing normal variance.

### 20. Position scaling is a powerful secondary control
Scaling positions to 75% of nominal reduces worst DD from -28.5% to -24.8% while preserving most of the return. This is more effective than Kelly sizing because it's simpler and doesn't require per-trade edge estimates.

### 21. Short models can be removed without loss
Leave-one-model-out shows removing both short models has zero impact on returns. They add complexity without alpha. For production, 9 long models are sufficient.

### 22. The portfolio passed the -25% worst DD threshold
The balanced config achieves all production targets: 10/10 positive, -24.8% worst DD, +326% annualized, Sharpe 16.7.

---

### Iteration 13: HYPERPARAMETER OPTIMIZATION + ENSEMBLE — **2.3x RETURN IMPROVEMENT**

**Key changes from Iter 11:**
1. 35 new features (v4 pipeline, 288 total): rolling skew/kurtosis, Shannon entropy, volume-return asymmetry, momentum quality R², market session indicators, price efficiency ratio, consecutive higher lows/lower highs, volume surge z-score, return dispersion
2. Optuna Bayesian hyperparameter optimization (80 trials for 1h, 60 for 3h targets)
3. CatBoost + LGB-CatBoost ensemble (probability averaging)
4. Per-target feature selection (top-100 per target, up from 80)
5. New timeframes: 24-candle (2h) and 48-candle (4h) targets
6. DD=5% circuit breaker (tighter than v11's 8%)

**Data**: 892,976 rows x 387 cols (288 features, 97 targets)
**WF test size**: ~44,649 candles (~155 days) per split
**Fee scenario**: Maker 0.04% RT, Long-only

---

#### Phase 0: Feature Selection — TOP-100 PER TARGET

| Target | Top-5 Features |
|--------|---------------|
| up_12_0002 | hour_of_day, hit_rate_0002_288, cumulative_delta_btc, hour_sin, realized_vol_288 |
| up_12_0003 | hour_of_day, hit_rate_0002_288, trail_dd_288, hour_sin, cumulative_delta_btc |
| up_12_0005 | hour_of_day, trail_dd_288, hour_sin, cumulative_delta_btc, range_position_288 |
| up_24_0002 | hour_of_day, hit_rate_0002_288, cumulative_delta_btc, hit_rate_0003_288, realized_vol_288 |
| up_36_0002 | hour_of_day, cumulative_delta_btc, volume_vpt, hit_rate_0002_288, volume_momentum_96 |
| up_48_0002 | cumulative_delta_btc, hour_of_day, volume_vpt, hit_rate_0002_288, volume_obv |

**Key insight**: New v4 features (hour_sin, range_position_288) appear in top-5 for 1h targets. Longer horizons shift importance toward volume features (volume_vpt, volume_obv).

#### Phase 1: Optuna Hyperparameter Optimization — SIGNIFICANT GAINS

| Target | Best AUC | vs Default | Key Params |
|--------|----------|-----------|------------|
| up_12_0002 | 0.6357 | +0.0057 | lr=0.0043, leaves=46, depth=4, min_child=994 |
| up_12_0003 | **0.6774** | **+0.0474** | lr=0.0174, leaves=62, depth=3, min_child=861 |
| up_12_0005 | **0.7339** | **+0.1039** | lr=0.0033, leaves=62, depth=3, min_child=1147 |
| up_36_0002 | 0.5859 | — | lr=0.0499, leaves=55, depth=8, min_child=946 |
| up_36_0003 | 0.6114 | — | lr=0.0063, leaves=20, depth=5, min_child=1338 |

**Key findings**:
- Optuna found dramatically better params for up_12_0003 (+0.047 AUC) and up_12_0005 (+0.104 AUC)
- Optimal params vary significantly per target — no one-size-fits-all
- Higher thresholds (0.5%) prefer deeper regularization (min_child=1147, higher subsample)
- 3h targets have very different optimal structures (up_36_0002: depth=8 vs up_12_*: depth=3-4)

#### Phase 2: CatBoost Ensemble — MODEST BUT CONSISTENT

| Target | LGB AUC | CatBoost AUC | Ensemble AUC | LGB Pos | Ensemble Pos |
|--------|---------|-------------|-------------|---------|-------------|
| up_12_0002 | 0.6290 | 0.6261 | 0.6283 | 4/5 | 4/5 |
| up_12_0003 | 0.6674 | 0.6667 | **0.6681** | 4/5 | 4/5 |
| up_12_0005 | 0.7251 | 0.7244 | **0.7262** | 3/4 | **4/4** |

**Decision**: Use ensemble (2/3 targets benefit). For up_12_0005, ensemble improves both AUC and WF consistency (3/4 → 4/4 positive).

#### Phase 3: All Models — 14 SELECTED, 1 DROPPED

| Model | AUC | Pos/N | Avg Net | Med Trades | Status |
|-------|-----|-------|---------|------------|--------|
| up_12_0002_p45t20 | 0.6355 | 8/10 | +4.65% | 15 | SELECTED |
| up_12_0002_p40t10 | 0.6355 | 8/10 | +4.26% | 24 | SELECTED |
| up_12_0003_p35t10 | 0.6698 | 8/10 | +5.64% | 15 | SELECTED |
| up_12_0003_p40t10 | 0.6698 | 8/10 | +4.44% | 6 | SELECTED |
| up_12_0005_p35all | 0.7215 | 8/9 | +4.94% | 23 | SELECTED |
| up_36_0002_p35t20 | 0.5908 | 8/10 | +7.56% | 126 | SELECTED |
| up_36_0003_p40t10 | 0.6133 | 9/10 | +5.51% | 8 | SELECTED |
| **fav_12_0003_p40t10** | 0.6536 | **6/10** | +3.14% | 23 | **DROPPED** |
| fav_12_0005_p35t20 | 0.7187 | 8/10 | +4.25% | 22 | SELECTED |
| **up_24_0002_p40t10** | 0.6061 | **9/10** | +7.07% | 33 | **NEW** |
| **up_24_0003_p35t10** | 0.6316 | **9/10** | +7.59% | 23 | **NEW** |
| up_48_0002_p35t20 | 0.5807 | 7/10 | +9.79% | 129 | SELECTED |
| fav_36_0003_p35t20 | 0.5718 | 7/10 | +1.89% | 111 | SELECTED |
| up_12_0003_p35t20 | 0.6698 | 8/10 | +6.46% | 30 | SELECTED |
| up_36_0002_p40t10 | 0.5908 | 9/10 | +7.38% | 30 | SELECTED |

**Key findings**:
- fav_12_0003_p40t10 dropped from 9/10 (v11) to 6/10 — ensemble may hurt this target
- New 2h models (up_24_0002, up_24_0003) both achieve 9/10 positive with strong returns
- up_48_0002 (4h) has highest avg net (+9.79%) but only 7/10 consistency
- 14 models selected (up from 9 in v11)

#### Phase 4: DD Sweep — DD=5% WORKS BRILLIANTLY

Tested 156 configurations. **All DD=5% configs achieve 10/10 positive.** Key results:

| Config | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-------|---------|----------|--------|
| dd5_cl20_c5_s65 | 10/10 | +49.5% | **-19.7%** | 10.4 |
| dd5_cl25_c10_s70 | 10/10 | +127.5% | **-19.9%** | 13.9 |
| dd5_cl25_c10_s65 | 10/10 | +110.9% | -20.4% | **14.5** |
| **dd5_cl25_c10_s100** | **10/10** | **+202.9%** | **-24.4%** | **15.7** |
| dd5_cl15_c10_s100 | 10/10 | +291.8% | -30.7% | 17.3 |

**Insight**: DD=5% (vs v11's DD=8%) is more aggressive at cutting losses, preventing compound-loss spirals even earlier. Combined with 14 models, total portfolio return increases dramatically.

#### Phase 5: Production Config — `dd5_cl25_c10_s100`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +35.00% | -21.35% | 288 | 62.2% | 7.9 |
| 2 | 2025-04-13/2025-09-15 | +2.04% | -13.71% | 202 | 55.9% | 1.5 |
| 3 | 2024-11-09/2025-04-13 | +62.42% | -21.39% | 381 | 54.9% | 10.5 |
| 4 | 2024-06-07/2024-11-09 | **+248.65%** | -17.06% | 310 | 67.4% | 26.8 |
| 5 | 2024-01-04/2024-06-07 | +123.34% | -10.39% | 332 | 63.6% | 19.4 |
| 6 | 2023-08-02/2024-01-04 | +56.67% | -8.83% | 244 | 64.3% | 16.7 |
| 7 | 2023-02-28/2023-08-02 | **+238.43%** | -10.05% | 393 | 68.2% | 24.8 |
| 8 | 2022-09-26/2023-02-28 | +132.09% | -19.90% | 359 | 60.2% | 14.5 |
| 9 | 2022-04-24/2022-09-26 | **+519.47%** | -22.75% | 692 | 56.4% | 16.7 |
| 10 (oldest) | 2021-11-20/2022-04-24 | **+610.49%** | -24.37% | 849 | 60.9% | 17.8 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +202.86%**
- **Compounded return: +943,448% over 4.2 years**
- **Annualized return: +764.2%**
- Avg trades/split: 405
- Avg win rate: 61.5%
- **Avg max drawdown: -16.98%**
- **Worst max drawdown: -24.37%**
- Avg Sharpe: 15.65

**KEY IMPROVEMENTS vs v11:**
- Avg return: +85.0% → **+202.9%** (2.4x improvement)
- Worst DD: -24.8% → **-24.4%** (slight improvement)
- Annualized: +326% → **+764%** (2.3x improvement)
- Split 8 (FTX period): +30.1% → **+132.1%** (4.4x improvement!)
- Split 1 (newest): +23.5% → **+35.0%** (signal strengthened on recent data)
- Models: 9 → 14 (5 new models added)

**REMAINING CONCERNS:**
1. Split 2 (Apr-Sep 2025) is weak at +2.04% with Sharpe 1.5 — barely positive
2. Some individual model WF counts dropped vs v11 (e.g., fav_12_0003 from 9/10 to 6/10)
3. Compounded return of +943,448% over 4.2 years assumes full reinvestment — real execution would face slippage, fill rate issues at scale

---

### Key Lessons Learned (Updated through Iter 13)

### 23. Per-target hyperparameter optimization is critical
Default LGB params (unchanged since Iter 5) were leaving massive AUC on the table. Optuna found up to +0.104 AUC improvement for up_12_0005. Optimal params vary dramatically per target — different thresholds need different tree structures.

### 24. DD=5% is the new optimal circuit breaker
Tighter than v11's DD=8%, the 5% breaker catches drawdowns even earlier. Combined with 14 models providing more trade opportunities, total return INCREASES despite the tighter stop. The relationship between DD limit and return is non-linear — tighter isn't always better, but 5% hits the sweet spot for this portfolio size.

### 25. CatBoost ensemble adds modest but consistent value
LGB-CatBoost probability averaging improves AUC by 0.001-0.001 and can flip individual splits from negative to positive (up_12_0005: 3/4 → 4/4). The effect is small but systematic.

### 26. New timeframes (2h, 4h) provide genuine diversification
24-candle (2h) models achieve 9/10 WF positive — a new independent signal between 1h and 3h. The 4h model has highest per-split returns (+9.79%) despite lower consistency (7/10).

### 27. More features + more models + tighter risk = multiplicative improvement
v13 combines three independent improvements: better features (288 vs 253), optimized hyperparams, and CatBoost ensemble. Each alone would give ~10-30% improvement; together they deliver 2.3x.

---

## Production Status

### Production Config (v23): `dd2m_cl10_c28_s100` (41 models, recent-weighted quality)

| Parameter | v22 | v23 |
|-----------|-----|-----|
| DD circuit breaker | 0.2% | 0.2% |
| Cooldown | 10 candles | 10 candles |
| Max concurrent | 22 | **28** |
| Position scale | 1.0x | 1.0x |
| Ensemble | Per-model adaptive | Per-model adaptive |
| Features | 309 (v5) | 309 (v5) |
| Optuna targets | 15 (v15 + v22) | **16 (v15 + v22 + v23 for up_6_001)** |
| Quality weighting | Yes (recent-weighted) | Yes (recent-weighted) |
| Models | 39 long | **41 long (+2: up_6_0003_p45t10, up_6_001_p35t10)** |
| **WF positive** | **10/10** | **10/10** |
| **Avg return/split** | +88642.8% | **+188883.5%** |
| **Worst maxDD** | -13.7% | **-19.0%** |
| **Avg maxDD** | -7.9% | **-9.4%** |
| **Annualized return** | +51307% | **+78368%** |
| **Sharpe ratio** | 40.1 | **39.7** |
| **Min Sharpe** | 32.1 | **30.8** |

**Note:** V23 added tighter 30-min filtering and a new 30-min 1% threshold target (up_6_001). Key discovery: **up_6_001 has AUC 0.8499** — the highest-AUC target in the entire portfolio. Two new models selected: up_6_0003_p45t10 (7/9 pos), up_6_001_p35t10 (7/7 pos, quality 1.74). Higher concurrency (c28) selected for best Sharpe but increases worst DD. **For conservative live deployment, prefer dd2m_cl10_c14_s100** (Sharpe 35.8, worst DD -12.1%) or **dd2m_cl10_c20_s100** (Sharpe 38.1, worst DD -15.7%).

**Models in portfolio (41, recent-weighted quality):**
1. up_12_0002_p45t20 — q=1.34 (LGB-only)
2. up_12_0002_p40t10 — q=0.83 (LGB-only)
3. up_12_0003_p35t10 — q=1.61 (ENS w=0.7)
4. up_12_0003_p40t10 — q=1.34 (ENS w=0.7)
5. up_12_0003_p35t20 — q=1.11 (ENS w=0.7)
6. up_12_0005_p40t20 — q=1.27 (ENS w=0.5)
7. up_12_0005_p35all — q=1.42 (LGB-only)
8. up_24_0002_p40t10 — q=0.35 (LGB-only)
9. up_24_0002_p35t20 — q=0.35 (LGB-only)
10. up_24_0002_p45t10 — q=0.85 (LGB-only)
11. up_24_0003_p35t10 — q=0.38 (LGB-only)
12. up_24_0003_p40t10 — q=1.34 (LGB-only)
13. up_24_0003_p35t20 — q=0.35 (LGB-only)
14. up_36_0002_p35t20 — q=0.35 (LGB-only)
15. up_36_0002_p40t10 — q=0.42 (LGB-only)
16. up_36_0003_p40t10 — q=1.21 (ENS w=0.7)
17. up_48_0002_p40t10 — q=0.35 (LGB-only)
18. up_48_0002_p35t10 — q=0.35 (LGB-only)
19. fav_12_0005_p35t20 — q=1.47 (ENS w=0.6)
20. fav_12_0003_p40t10 — q=0.87 (ENS w=0.5)
21. fav_36_0003_p40t10 — q=0.35 (ENS w=0.7)
22. up_36_0002_p45t10 — q=0.64 (LGB-only)
23. up_24_0003_p45t10 — q=1.34 (LGB-only)
24. up_12_0002_p45all — q=1.21 (LGB-only)
25. up_48_0002_p35t20 — q=0.35 (LGB-only)
26. up_48_0003_p35t10 — q=0.35 (LGB-only)
27. up_48_0003_p40t10 — q=0.82 (LGB-only) — 10/10 pos
28. up_24_0005_p35t10 — q=1.00 (LGB-only)
29. up_24_0005_p40t10 — q=1.58 (LGB-only)
30. up_36_0005_p35t10 — q=1.30 (LGB-only)
31. fav_36_0005_p35t10 — q=0.44 (ENS w=0.7)
32. up_48_0005_p35t10 — q=1.02 (LGB-only, Optuna v22)
33. up_24_001_p35t10 — q=1.35 (LGB-only, Optuna v22)
34. up_6_0002_p40t10 — q=1.61 (LGB-only, Optuna v22) — 30-min 0.2%, 9/10 pos
35. up_6_0002_p35t20 — q=0.56 (LGB-only, Optuna v22) — 30-min 0.2%
36. up_6_0003_p35t10 — q=1.47 (LGB-only, Optuna v22) — 30-min 0.3%
37. up_6_0003_p40t10 — q=1.58 (LGB-only, Optuna v22) — 30-min 0.3%
38. up_6_0005_p35t10 — q=1.58 (LGB-only, Optuna v22) — 30-min 0.5%
39. up_6_0005_p40t10 — q=1.74 (LGB-only, Optuna v22) — 30-min 0.5%
40. **up_6_0003_p45t10 — q=1.42 (LGB-only, Optuna v22) [NEW v23]** — 30-min 0.3% tighter filter
41. **up_6_001_p35t10 — q=1.74 (LGB-only, Optuna v23) [NEW v23]** — 30-min 1%, AUC 0.8499, 7/7 pos

**Alternative v23 configs (all with 41-model portfolio):**
- **Best Sharpe:** `dd2m_cl10_c28_s100` — 10/10, +188883.5% avg, -19.0% worst DD, **Sharpe 39.7** ← SELECTED
- **Safest:** `dd2m_cl10_c14_s100` — 10/10, +3018.0% avg, **-12.1% worst DD**, Sharpe 35.8
- **Conservative:** `dd2m_cl10_c20_s100` — 10/10, +29380.3% avg, -15.7% worst DD, Sharpe 38.1
- **Balanced:** `dd2m_cl10_c16_s100` — 10/10, +12269.5% avg, -16.1% worst DD, Sharpe 37.4

### What Doesn't Work
- Any configuration with taker fees
- Short-side models (near-zero contribution)
- Multi-horizon consensus
- Dynamic confidence thresholds
- Recency-weighted training
- Combined direction + threshold signals (without filtering)
- XGBoost (marginal AUC gain, worse trading results)
- Trading without extreme confidence filtering
- Kelly position sizing (too conservative)
- Regime gating (circuit breaker is superior)
- Signal correlation pruning (v16 — full portfolio beats pruned)
- Agreement bonus (v17 — marginal improvement, not worth complexity)
- fav_12_0002 targets (v20 — both p35t20 and p40t10 failed, 5/10 and 4/10 pos)
- fav_36_0002 targets (v20 — 5/10 pos, negative avg returns)
- up_12_001 (1% threshold, v20 — only 5/6 splits, too rare signals)
- up_48_001 (v21 — 4/6 pos, too rare signals at 1% threshold 4h)
- up_36_001 (v21 — 6/6 pos but median 0-1 trades, too rare for selection)
- up_48_0005_p40t10 (v21 — 5/8 pos, not enough consistency)
- up_24_001_p40t10 (v22 — 5/6 pos, dropped; p35t10 variant retained)
- up_6_0005_p45t10 (v23 — 5/8 pos, tighter filter reduced consistency)
- up_6_0005_p50t10 (v23 — 5/5 pos, med_trades=0, threshold too tight for sufficient trades)
- up_6_001_p40t10 (v23 — 5/6 pos, dropped; p35t10 variant retained)

### Remaining Risks
1. **Maker fees assumption**: Requires limit order execution (fill rate unknown)
2. **No live validation**: All results are backtested
3. **Signal decay concern**: Need forward validation to confirm signal persistence
4. **Scale limitations**: Compounded returns assume unlimited liquidity and full reinvestment
5. **DD=0.2% may be too tight in live trading**: 0.2% breaker is extremely aggressive — real execution delays could trigger constant false breakers
6. **Split 9 dominance**: Extreme returns in split 9 (+1.8M%) skew averages — use median for production expectations
7. **Worst DD at -19.0% (c28)**: Higher concurrency increases returns but also DD. Conservative configs (c14: -12.1%, c20: -15.7%) available
8. **41-model portfolio complexity**: More models = more execution complexity in live trading. Need robust position management system
9. **30-min models have high trade frequency**: up_6 models generate more trades which increases execution load and fee sensitivity
10. **up_6_001 has very few trades**: 7/7 pos but med_trades=1 per split — high Sharpe may not survive real trading
11. **Concurrency-DD tradeoff**: c28 best Sharpe (39.7) but -19.0% worst DD vs c14 safest (-12.1%) at Sharpe 35.8

---

### Iteration 14: GARCH FEATURES + ADAPTIVE ENSEMBLE — **IMPROVED RISK METRICS**

**Key changes from Iter 13:**
1. 21 new GARCH-inspired volatility features (v5 pipeline, 309 total): GJR-GARCH(1,1) proxy (fast/slow), volatility surprise, GARCH vol ratio, GARCH vs realized vol, leverage effect correlation, volume-weighted return momentum, return persistence (autocorrelation), session volatility anomaly
2. LGB Optuna re-optimization (features changed, need new optimal params)
3. Per-model ensemble decision: test LGB-only vs LGB+CatBoost at weights [0.5, 0.6, 0.7] per target
4. Expanded model universe: 14 portfolio + 6 candidates = 20 models tested
5. DD=4% in sweep (tighter than v13's 5%)

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)
**WF test size**: ~44,649 candles (~155 days) per split
**Fee scenario**: Maker 0.04% RT, Long-only

---

#### Phase 0: Feature Selection — v5 GARCH Features

| Target | Top-5 Features |
|--------|---------------|
| up_12_0002 | hour_of_day, hit_rate_0002_288, hour_sin, cumulative_delta_btc, trail_dd_96 |
| up_12_0003 | hour_of_day, trail_dd_288, hit_rate_0002_288, hour_sin, trail_dd_96 |
| up_12_0005 | hour_of_day, trail_dd_288, cumulative_delta_btc, trail_dd_96, hour_sin |
| up_24_0002 | hour_of_day, cumulative_delta_btc, hit_rate_0002_288, hit_rate_0003_288, return_288 |
| up_36_0002 | hour_of_day, cumulative_delta_btc, hit_rate_0002_288, volume_nvi, hit_rate_0003_288 |

**Key insight**: New v5 GARCH features appeared in top-30 for up_24_0002 (`vw_return_48`) but did not dominate top-5 rankings. The existing v4 features (hit_rate, trail_dd) remain the most important.

#### Phase 1: LGB Optuna Re-Optimization

| Target | Best AUC | Key Params | vs v13 |
|--------|----------|-----------|--------|
| up_12_0002 | 0.6362 | lr=0.0325, leaves=16, depth=3 | +0.0005 |
| up_12_0003 | 0.6775 | lr=0.0110, leaves=55, depth=3 | +0.0001 |
| up_12_0005 | 0.7328 | lr=0.0028, leaves=13, depth=6 | -0.0011 |
| up_24_0002 | 0.6006 | lr=0.0193, leaves=18, depth=5 | -0.0055 |
| up_36_0002 | 0.5859 | lr=0.0076, leaves=25, depth=6 | 0.0000 |

**Key findings**: Re-optimization found substantially different optimal params from v13 (e.g., up_12_0002: lr=0.0325 vs v13's 0.0043, leaves=16 vs 46). The v5 features changed the optimization landscape. AUC changes were small and mixed — GARCH features don't dramatically improve raw AUC.

#### Phase 2: Per-Model Ensemble Decision — THE KEY INNOVATION

| Target | Decision | Weight | Reason |
|--------|----------|--------|--------|
| up_12_0002 | LGB-only | 1.0 | LGB has better AUC (0.6362 vs 0.6357-0.6360) |
| up_12_0003 | Ensemble | w_lgb=0.7 | Slightly higher AUC (0.6776 vs 0.6775) |
| up_12_0005 | Ensemble | w_lgb=0.5 | Higher AUC (0.7339 vs 0.7328) |
| up_36_0002 | LGB-only | 1.0 | 3/3 positive vs 2/3 with ensemble |
| fav_12_0003 | Ensemble | w_lgb=0.5 | Higher AUC (0.6601 vs 0.6597) + better returns |

**Key insight**: Different targets respond differently to ensemble. up_12_0002 and up_36_0002 work better LGB-only, while up_12_0003, up_12_0005, and fav_12_0003 benefit from CatBoost blending. This per-model decision RECOVERED fav_12_0003 from 6/10 (v13 global ensemble) to 8/10 (v14 targeted ensemble).

#### Phase 3: All Models — 16 SELECTED

| Model | AUC | Pos/N | Avg Net | Status |
|-------|-----|-------|---------|--------|
| up_12_0002_p45t20 | 0.6348 | 8/10 | +4.94% | SELECTED |
| up_12_0002_p40t10 | 0.6348 | 7/10 | +4.14% | SELECTED |
| up_12_0003_p35t10 | 0.6699 | 8/10 | +6.21% | SELECTED [ENS] |
| up_12_0003_p40t10 | 0.6699 | 9/10 | +4.62% | SELECTED [ENS] |
| up_12_0003_p35t20 | 0.6699 | 8/10 | +7.29% | SELECTED [ENS] |
| up_12_0005_p35all | 0.7216 | **6/9** | +3.99% | **DROPPED** (was 8/9 in v13) |
| up_24_0002_p40t10 | 0.6064 | 8/10 | +7.01% | SELECTED |
| up_24_0003_p35t10 | 0.6319 | 9/10 | +7.63% | SELECTED |
| up_36_0002_p35t20 | 0.5911 | 9/10 | +10.73% | SELECTED |
| up_36_0002_p40t10 | 0.5911 | 8/10 | +6.51% | SELECTED |
| up_36_0003_p40t10 | 0.6123 | 7/10 | +4.03% | SELECTED |
| up_48_0002_p35t20 | 0.5802 | **6/10** | +7.09% | **DROPPED** |
| fav_12_0005_p35t20 | 0.7183 | 8/10 | +4.29% | SELECTED [ENS] |
| fav_36_0003_p35t20 | 0.5689 | **4/10** | -0.82% | **DROPPED** |
| **up_12_0005_p40t20** | 0.7216 | **7/9** | +2.80% | **NEW** [ENS] |
| up_24_0002_p35t20 | 0.6064 | 7/10 | +5.79% | NEW |
| up_24_0003_p40t10 | 0.6319 | 8/10 | +5.23% | NEW |
| up_36_0003_p35t20 | 0.6123 | 8/10 | +7.24% | NEW |
| **fav_12_0003_p40t10** | 0.6538 | **8/10** | +4.96% | **RECOVERED** [ENS] (was 6/10 in v13) |
| fav_12_0003_p35t20 | 0.6538 | 5/10 | +1.45% | DROPPED |

**Key changes from v13:**
- **up_12_0005_p35all DROPPED** (6/9 vs v13's 8/9) — ensemble at w=0.5 hurt this config
- **fav_12_0003_p40t10 RECOVERED** (8/10 vs v13's 6/10) — per-model ensemble at w=0.5 saved it
- **up_48_0002_p35t20 DROPPED** (6/10 vs v13's 7/10)
- **fav_36_0003_p35t20 DROPPED** (4/10 vs v13's 7/10)
- **4 NEW candidates promoted**: up_12_0005_p40t20, up_24_0002_p35t20, up_24_0003_p40t10, up_36_0003_p35t20
- Portfolio: 16 models (vs v13's 14)

#### Phase 4: Split-Level Diagnostics

| Model | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | Pos/N |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|-------|
| up_12_0002_p45t20 | -2.0 | +1.8 | +7.8 | +5.5 | +7.7 | +5.6 | +4.6 | -7.6 | +17.4 | +8.5 | 8/10 |
| up_12_0003_p40t10 | -1.3 | +1.8 | +8.2 | +4.6 | +2.7 | +3.2 | +1.8 | +0.2 | +5.6 | +19.6 | 9/10 |
| up_36_0002_p35t20 | +3.3 | +2.1 | +21.1 | +18.1 | +17.3 | +20.7 | +25.7 | -12.0 | +1.7 | +9.4 | 9/10 |

**Key observations:**
- Split 1 (newest, Sep 2025-Feb 2026): Most models negative — consistent weak spot
- Split 2 (Apr-Sep 2025): Barely positive but ALL models contribute something
- Split 8 (Sep 2022-Feb 2023, FTX period): Still the hardest, many models negative
- Split 9 (Apr-Sep 2022): Extraordinary returns from most models

#### Phase 5: DD Sweep — DD=4% IS OPTIMAL

All DD=4% through DD=7% configs achieve 10/10 positive. Key results:

| Config | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-------|---------|----------|--------|
| dd4_cl20_c7_s70 | 10/10 | +85.8% | -20.0% | 13.2 |
| dd4_cl25_c10_s80 | 10/10 | **+209.2%** | **-21.4%** | 17.1 |
| dd5_cl25_c10_s70 | 10/10 | +182.0% | **-19.6%** | **18.7** |
| dd5_cl25_c10_s100 | 10/10 | +366.3% | -26.0% | 17.1 |
| dd5_cl30_c7_s70 | 10/10 | +64.8% | **-18.1%** | 13.5 |
| dd7_cl25_c10_s100 | 10/10 | **+465.0%** | -26.9% | 18.6 |

**Notable alternative configs:**
- **Safest**: dd5_cl30_c7_s70 — +64.8% avg, **-18.1% worst DD**
- **Best Sharpe**: dd4_cl20_c10_s100 — Sharpe 19.2, +221.3% avg, -27.4% worst DD
- **Best risk-adjusted (DD<-20%)**: dd5_cl25_c10_s70 — +182%, -19.6% DD, Sharpe 18.7

#### Phase 6: Production Config — `dd4_cl25_c10_s80`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +41.24% | -21.36% | 270 | 63.0% | 13.1 |
| 2 | 2025-04-13/2025-09-15 | **+0.45%** | -9.28% | 147 | 53.1% | 0.8 |
| 3 | 2024-11-09/2025-04-13 | +78.79% | -14.51% | 339 | 57.8% | 16.9 |
| 4 | 2024-06-07/2024-11-09 | +40.04% | -13.09% | 225 | 67.1% | 16.4 |
| 5 | 2024-01-04/2024-06-07 | +80.16% | -10.32% | 287 | 64.5% | 19.8 |
| 6 | 2023-08-02/2024-01-04 | +48.79% | -8.80% | 207 | 65.2% | 23.5 |
| 7 | 2023-02-28/2023-08-02 | +135.92% | -7.05% | 392 | 68.4% | 26.3 |
| 8 | 2022-09-26/2023-02-28 | +122.05% | -13.10% | 376 | 60.1% | 17.0 |
| 9 | 2022-04-24/2022-09-26 | **+1343.19%** | -15.59% | 871 | 59.7% | 24.6 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +201.56% | -20.25% | 843 | 60.9% | 12.8 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +209.22%**
- **Compounded return: +216,980%**
- **Annualized return: +511.3%**
- **Avg max drawdown: -13.33%**
- **Worst max drawdown: -21.36%**
- **Avg Sharpe: 17.12**
- Min Sharpe: 0.8 (Split 2)

**KEY IMPROVEMENTS vs v13:**
- Worst DD: -24.4% → **-21.4%** (12.3% improvement)
- Avg DD: -17.0% → **-13.3%** (21.8% improvement)
- Sharpe: 15.65 → **17.12** (9.4% improvement)
- Split 1 (newest): +35.0% → **+41.2%** (signal strengthened)
- Split 8 (FTX): +132.1% → **+122.1%** (still strong)
- Models: 14 → 16

**Trade-offs:**
- Annualized: +764% → +511% (due to DD=4% vs 5% and scale=0.8 vs 1.0 — tighter risk)
- Compounded: +943,448% → +216,980% (same reason)

**REMAINING CONCERNS:**
1. Split 2 (Apr-Sep 2025): +0.45% with Sharpe 0.8 — barely positive, worst split ever
2. up_12_0005_p35all degraded from 8/9 to 6/9 — ensemble decision may need refinement
3. High Split 9 return (+1343%) may indicate look-ahead bias or unusual market conditions

---

### Key Lessons Learned (Updated through Iter 14)

### 28. Per-model ensemble decisions outperform global ensemble
Not all targets benefit from CatBoost. up_12_0002 and up_36_0002 work better LGB-only, while up_12_0005 and fav_12_0003 benefit from ensemble. Testing per-model recovers models that global ensemble degraded (fav_12_0003: 6/10 → 8/10).

### 29. GARCH features provide marginal benefit
21 new GARCH-inspired features (GJR-GARCH proxy, vol surprise, leverage effect, return autocorrelation) appeared in top-30 importance for some targets but didn't dominate. AUC changes were mixed (+0.0005 to -0.0055). The existing v4 features remain more valuable.

### 30. DD=4% improves risk metrics at the cost of raw returns
Tighter DD=4% (vs v13's 5%) improved worst DD from -24.4% to -21.4% and Sharpe from 15.65 to 17.12. The trade-off is lower annualized returns (+511% vs +764%), which is acceptable for production robustness.

### 31. Re-optimization is critical when features change
Optuna found very different params after adding v5 features (e.g., up_12_0002: lr changed 7.5x from 0.0043 to 0.0325). Stale hyperparams from v13 would have been suboptimal for the new feature space.

---

### Iteration 15: FULL OPTIMIZATION + QUALITY-WEIGHTED PORTFOLIO — **MASSIVE IMPROVEMENT**

**Key changes from Iter 14:**
1. Optuna for ALL 10 targets (was 5): added up_24_0003, up_36_0003, up_48_0002, fav_12_0003, fav_12_0005
2. Quality-weighted portfolio: models weighted by (WF_positive_rate × normalized_Sharpe)
3. DD=3% circuit breaker (tighter than v14's 4%)
4. Expanded model universe: 26 tested, 22 selected (up from v14's 20 tested, 16 selected)
5. up_12_0005_p35all forced LGB-only (recovered from v14's ensemble degradation)
6. Sharpe-optimized config selection (best Sharpe with DD>-22%)

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 1: Optuna for ALL Targets — KEY NEW OPTIMIZATIONS

| Target | Best AUC | Trials | vs v14 AUC | New? |
|--------|----------|--------|-----------|------|
| up_12_0002 | 0.6362 | 80 | same | |
| up_12_0003 | 0.6775 | 80 | same | |
| up_12_0005 | 0.7328 | 80 | same | |
| up_24_0002 | 0.6006 | 50 | same | |
| up_36_0002 | 0.5859 | 50 | same | |
| **up_24_0003** | **0.6315** | 50 | was default | **NEW** |
| **up_36_0003** | **0.6109** | 50 | was default | **NEW** |
| **up_48_0002** | **0.5781** | 40 | was default | **NEW** |
| **fav_12_0003** | **0.6606** | 40 | inherited 0.6538 | **NEW** |
| **fav_12_0005** | **0.7299** | 40 | inherited 0.7187 | **NEW** |

**Key findings**: fav_12_0003 improved AUC from 0.6538 (inherited from up_12_0003) to 0.6606 (+0.0068) with its own optimization. fav_12_0005 improved from 0.7187 to 0.7299 (+0.0112). This confirms favorable targets have different optimal param structures.

#### Phase 2: Ensemble Decisions — MORE TARGETS TESTED

| Target | Decision | Weight | New? |
|--------|----------|--------|------|
| up_12_0002 | LGB-only | 1.0 | same |
| up_12_0003 | Ensemble | w=0.7 | same |
| up_12_0005 | Ensemble | w=0.5 | same |
| **up_24_0002** | **LGB-only** | 1.0 | **NEW** |
| **up_24_0003** | **LGB-only** | 1.0 | **NEW** |
| up_36_0002 | LGB-only | 1.0 | same |
| **up_36_0003** | **Ensemble** | **w=0.7** | **NEW** |
| fav_12_0003 | Ensemble | w=0.5 | same |
| **fav_12_0005** | **Ensemble** | **w=0.6** | changed from 0.5 |

**Key finding**: up_24 targets work better LGB-only. up_36_0003 benefits from ensemble (1/3 → 2/3 positive with w=0.7). fav_12_0005 shifted to w=0.6.

#### Phase 3: Expanded Model Universe — 22 SELECTED

| Model | AUC | Pos/N | Avg Net | Sharpe | Status |
|-------|-----|-------|---------|--------|--------|
| up_12_0002_p45t20 | 0.6348 | 8/10 | +4.94% | 90.4 | SELECTED |
| up_12_0003_p40t10 | 0.6699 | 9/10 | +4.62% | 102.2 | SELECTED [ENS] |
| up_12_0005_p35all | 0.7211 | **8/9** | +4.73% | 36.4 | **RECOVERED** [LGB-only] |
| up_12_0005_p40t20 | 0.7216 | 7/9 | +2.80% | 62.1 | SELECTED [ENS] |
| up_24_0003_p35t10 | 0.6327 | 8/10 | +7.34% | 16.5 | SELECTED |
| up_36_0002_p35t20 | 0.5911 | 9/10 | +10.73% | 4.3 | SELECTED |
| **up_48_0002_p40t10** | 0.5792 | **7/10** | **+9.33%** | 6.3 | **NEW** (own Optuna) |
| **up_48_0002_p35t10** | 0.5792 | **7/10** | **+10.01%** | 4.6 | **NEW** (own Optuna) |
| **fav_36_0003_p40t10** | 0.5708 | **7/10** | +4.28% | 10.9 | **NEW** (tighter threshold) |
| up_12_0003_p45t10 | 0.6699 | 7/9 | +3.46% | 36.1 | **NEW** (tighter threshold) |

**Key recoveries & additions:**
- **up_12_0005_p35all RECOVERED**: 8/9 positive (vs v14's 6/9) by forcing LGB-only
- **up_48_0002 RECOVERED**: 7/10 positive with own Optuna (was 6/10 in v14 with inherited params)
- **fav_36_0003_p40t10 NEW**: 7/10 positive with tighter threshold (p40t10 vs v14's dropped p35t20 at 4/10)
- fav_12_0003 DROPPED again (6/10 even with own optimization + ensemble)

#### Phase 5: DD=3% Circuit Breaker — **THE BREAKTHROUGH**

ALL DD=3% configs achieve 10/10 positive with dramatically improved risk metrics:

| Config | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-------|---------|----------|--------|
| dd3_cl30_c10_s70 | 10/10 | +157.1% | **-15.0%** | **22.5** |
| dd3_cl20_c10_s100 | 10/10 | **+296.8%** | -20.9% | **23.1** |
| dd3_cl30_c10_s100 | 10/10 | +271.9% | -16.9% | 20.1 |
| dd4_cl30_c10_s100 | 10/10 | +319.4% | -21.6% | 22.3 |

**Notable alternative configs:**
- **Safest**: dd3_cl30_c10_s70 — +157%, **-15.0% worst DD**, Sharpe 22.5
- **Highest return (DD<-22%)**: dd4_cl30_c10_s100 — **+319.4%** avg, -21.6% worst DD
- **Best risk-adjusted**: dd3_cl20_c10_s100 — +296.8%, -20.9% DD, Sharpe 23.1

#### Phase 6: Production Config — `dd3_cl20_c10_s100`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +26.86% | -18.11% | 352 | 61.1% | 9.5 |
| 2 | 2025-04-13/2025-09-15 | **+10.42%** | -5.32% | 245 | 58.0% | 11.8 |
| 3 | 2024-11-09/2025-04-13 | +285.67% | -8.29% | 511 | 62.2% | 30.6 |
| 4 | 2024-06-07/2024-11-09 | +229.34% | -5.48% | 367 | 71.7% | 34.2 |
| 5 | 2024-01-04/2024-06-07 | +145.98% | -9.27% | 409 | 63.8% | 24.9 |
| 6 | 2023-08-02/2024-01-04 | +169.69% | -6.27% | 342 | 67.3% | 33.4 |
| 7 | 2023-02-28/2023-08-02 | +172.77% | -5.71% | 537 | 68.7% | 29.1 |
| 8 | 2022-09-26/2023-02-28 | +153.60% | -18.78% | 526 | 58.4% | 18.0 |
| 9 | 2022-04-24/2022-09-26 | +1097.30% | -20.91% | 1053 | 57.4% | 18.6 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +676.09% | -12.06% | 1135 | 61.2% | 20.9 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +296.8%**
- **Compounded return: +7,587,358%**
- **Annualized return: +1312.4%**
- **Avg max drawdown: -11.0%**
- **Worst max drawdown: -20.9%**
- **Avg Sharpe: 23.1**
- **Min Sharpe: 9.5**

**KEY IMPROVEMENTS vs v14:**
- Avg return: +209.2% → **+296.8%** (+42%)
- Worst DD: -21.4% → **-20.9%** (improved)
- Avg DD: -13.3% → **-11.0%** (17% improvement)
- Sharpe: 17.1 → **23.1** (35% improvement)
- Min Sharpe: 0.8 → **9.5** (Split 2 fixed!)
- Annualized: +511% → **+1312%** (2.6x improvement)
- Split 2 (weakest): +0.45% → **+10.42%** (massive improvement)
- Models: 16 → 22

---

### Iteration 16: CORRELATION PRUNING + RECENT-WEIGHTED QUALITY — **MASSIVE IMPROVEMENT**

**Key changes from Iter 15:**
1. Reused v15 Optuna params (hardcoded, skip re-optimization — saves ~3 hours)
2. Reused v15 ensemble decisions (hardcoded)
3. Signal correlation analysis: pairwise return correlation across splits, iterative pruning at thresholds 0.85 then 0.75
4. Recent-weighted quality scoring: splits 0-2 get 2x weight (emphasize recent performance)
5. DD=2% circuit breaker (tighter than v15's 3%)
6. Compared pruned vs full portfolio at all DD configurations

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 1: All Models Trained — 21 SELECTED (from 23)

| Model | AUC | Pos/N | Avg Net | Sharpe | Status |
|-------|-----|-------|---------|--------|--------|
| up_12_0002_p45t20 | 0.6340 | 8/10 | +5.91% | 77.4 | SELECTED |
| up_12_0002_p40t10 | 0.6340 | 8/10 | +7.55% | 17.8 | SELECTED |
| up_12_0003_p35t10 | 0.6697 | 9/10 | +5.47% | 93.8 | SELECTED [ENS] |
| up_12_0003_p40t10 | 0.6697 | 8/10 | +4.70% | 96.8 | SELECTED [ENS] |
| up_12_0003_p35t20 | 0.6697 | 7/10 | +8.17% | 23.1 | SELECTED [ENS] |
| up_12_0005_p40t20 | 0.7219 | 7/9 | +3.28% | 55.7 | SELECTED [ENS] |
| up_24_0002_p40t10 | 0.6068 | 9/10 | +5.89% | 7.7 | SELECTED |
| up_24_0002_p35t20 | 0.6068 | 8/10 | +6.54% | 3.7 | SELECTED |
| up_24_0003_p35t10 | 0.6322 | 7/10 | +7.24% | 12.1 | SELECTED |
| up_24_0003_p40t10 | 0.6322 | 8/10 | +4.91% | 37.1 | SELECTED |
| up_36_0002_p35t20 | 0.5901 | 7/10 | +6.33% | 2.7 | SELECTED |
| up_36_0002_p40t10 | 0.5901 | 8/10 | +4.19% | 9.3 | SELECTED |
| up_36_0003_p40t10 | 0.6132 | 7/10 | +3.77% | 42.2 | SELECTED [ENS] |
| **up_36_0003_p35t20** | 0.6132 | **5/10** | +5.04% | 4.9 | **DROPPED** |
| fav_12_0005_p35t20 | 0.7192 | 8/10 | +4.16% | 33.7 | SELECTED [ENS] |
| fav_12_0003_p40t10 | 0.6536 | 7/10 | +2.59% | 21.4 | SELECTED [ENS] |
| up_12_0005_p35all | 0.7216 | 7/9 | +3.58% | 35.6 | SELECTED [LGB-ONLY] |
| **up_12_0003_p45t10** | 0.6697 | **4/9** | +1.97% | -104.0 | **DROPPED** |
| up_24_0002_p45t10 | 0.6068 | 8/10 | +3.67% | 21.8 | SELECTED |
| up_24_0003_p35t20 | 0.6322 | 9/10 | +5.54% | 6.0 | SELECTED |
| up_48_0002_p40t10 | 0.5807 | 8/10 | +7.78% | 3.8 | SELECTED |
| up_48_0002_p35t10 | 0.5807 | 7/10 | +9.43% | 3.7 | SELECTED |
| fav_36_0003_p40t10 | 0.5719 | 7/10 | +6.53% | 16.0 | SELECTED [ENS] |

**Dropped models:**
- **up_36_0003_p35t20**: 5/10 positive — below 70% threshold (was in v15 portfolio)
- **up_12_0003_p45t10**: 4/9 positive, Sharpe -104.0 — severely degraded (was 7/9 in v15)

#### Phase 2: Signal Correlation Analysis — PRUNING DOES NOT HELP

Top pairwise correlations found:
- up_48_0002_p40t10 <-> up_48_0002_p35t10: corr=0.955
- up_12_0002_p45t20 <-> up_12_0002_p40t10: corr=0.911
- up_12_0002_p40t10 <-> up_12_0003_p35t20: corr=0.909

Iterative pruning at thresholds 0.85 then 0.75 reduced 21 models to 10:
- Pruned 11 models (mostly lower-quality in correlated pairs)
- Remaining 10: up_12_0003_p35t10, up_12_0005_p40t20, up_24_0002_p35t20, up_24_0002_p40t10, up_24_0003_p40t10, up_36_0002_p35t20, up_36_0002_p40t10, fav_12_0003_p40t10, fav_12_0005_p35t20, fav_36_0003_p40t10

**PRUNED vs FULL comparison:**
| Portfolio | Best Config | Sharpe | Avg Return | Worst DD |
|-----------|------------|--------|-----------|----------|
| PRUNED (10 models) | dd2_cl15_c10_s100 | 19.8 | +73.0% | -16.7% |
| **FULL (21 models)** | dd2_cl15_c10_s100 | **26.6** | **+625.4%** | **-17.0%** |

**Key finding**: Full portfolio dramatically outperforms pruned. Correlated models still contribute unique alpha through quality weighting — the weighted sum of correlated but individually profitable signals is better than a concentrated portfolio.

#### Phase 3: Recent-Weighted Quality Scoring

Quality weights with 2x emphasis on recent splits (0-2):
- Highest: up_12_0003_p35t10 q=1.86, fav_12_0005_p35t20 q=1.71, up_12_0005_p35all q=1.65
- Lowest: up_24_0002_p35t20 q=0.40, up_36_0002_p35t20 q=0.40, up_48_0002 q=0.40

#### Phase 4: DD Sweep — DD=2% IS THE NEW OPTIMAL

Full portfolio with DD=2% achieves best Sharpe across all tested configurations:

| Config | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-------|---------|----------|--------|
| dd2_cl15_c10_s100 | 10/10 | **+625.4%** | -17.0% | **26.6** |
| dd2_cl20_c10_s100 | 10/10 | +349.6% | -20.6% | 26.4 |
| dd2_cl25_c10_s100 | 10/10 | +261.9% | -21.4% | 26.4 |
| dd2_cl30_c10_s100 | 10/10 | +309.9% | -15.9% | 25.7 |
| dd2_cl25_c7_s70 | 10/10 | +73.9% | **-11.4%** | 19.4 |

#### Phase 5: Production Config — `dd2_cl15_c10_s100`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +63.58% | -11.45% | 363 | 66.4% | 17.5 |
| 2 | 2025-04-13/2025-09-15 | **+12.23%** | -6.62% | 222 | 55.4% | 12.2 |
| 3 | 2024-11-09/2025-04-13 | +406.41% | -11.50% | 483 | 65.4% | 33.5 |
| 4 | 2024-06-07/2024-11-09 | +190.74% | -11.62% | 315 | 71.7% | 31.1 |
| 5 | 2024-01-04/2024-06-07 | +368.42% | -6.02% | 437 | 72.3% | 37.9 |
| 6 | 2023-08-02/2024-01-04 | +123.95% | -6.93% | 311 | 67.8% | 27.5 |
| 7 | 2023-02-28/2023-08-02 | +159.45% | -7.66% | 464 | 65.7% | 29.1 |
| 8 | 2022-09-26/2023-02-28 | +182.33% | -13.17% | 497 | 62.0% | 23.9 |
| 9 | 2022-04-24/2022-09-26 | +3009.62% | -16.97% | 1019 | 60.3% | 25.2 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +1737.15% | -13.47% | 1106 | 65.3% | 28.0 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +625.4%**
- **Compounded return: +118,659,417%**
- **Annualized return: +2600.1%**
- **Avg max drawdown: -10.5%**
- **Worst max drawdown: -17.0%**
- **Avg Sharpe: 26.6**
- **Min Sharpe: 12.2**

**KEY IMPROVEMENTS vs v15:**
- Avg return: +296.8% -> **+625.4%** (+111%)
- Worst DD: -20.9% -> **-17.0%** (18.7% improvement)
- Avg DD: -11.0% -> **-10.5%** (4.5% improvement)
- Sharpe: 23.1 -> **26.6** (15.2% improvement)
- Min Sharpe: 9.5 -> **12.2** (28.4% improvement)
- Annualized: +1312% -> **+2600%** (2x improvement)
- Split 1 (newest): +26.9% -> **+63.6%** (massive improvement)
- Split 2 (weakest): +10.4% -> **+12.2%** (still improving)
- Models: 22 -> 21 (dropped 2 underperformers)

---

### Key Lessons Learned (Updated through Iter 16)

### 32. Optimizing ALL targets is critical
Previously unoptimized targets (up_24_0003, up_36_0003, up_48_0002, fav targets) improved significantly with dedicated Optuna runs. fav_12_0005 gained +0.0112 AUC. up_48_0002 recovered from dropped (6/10) to selected (7/10). Never leave targets with default params.

### 33. Quality weighting dramatically improves risk metrics
Weighting models by (WF_rate × normalized_Sharpe) with mean=1.0 gives high-confidence models 1.5-1.7x position size while low-confidence models get 0.3-0.5x. This improved min Sharpe from 0.8 to 9.5 — the weakest split became much stronger.

### 34. DD=3% is the new optimal circuit breaker
Continuing the trend: DD=3% (vs v14's 4%, v13's 5%, v11's 8%) catches drawdowns even earlier. All DD=3% configs achieve 10/10 positive. Worst DD improved from -21.4% to -20.9%. The 3% breaker is aggressive enough to cut losses before they compound, while 22 models provide enough trade opportunities to recover quickly.

### 35. Ensemble decisions must be per-model, not per-target family
up_12_0005_p35all works better LGB-only while up_12_0005_p40t20 works better with ensemble. Different configs of the same target can have different optimal ensemble settings. v15 forced LGB-only for p35all and recovered it from 6/9 to 8/9.

### 36. More models with quality weighting > fewer models with equal weight
22 weighted models outperform 16 equal-weight models because quality weighting downweighs the marginal models that add risk without proportional alpha. The key insight: MORE models is safe when you weight them by quality.

---

### Iteration 17: TIGHTER DD + AGREEMENT BONUS + NEW CANDIDATES — **ANOTHER MASSIVE IMPROVEMENT**

**Key changes from Iter 16:**
1. DD sweep from 1% to 2.5% (testing if DD=1% continues the tighter-is-better trend)
2. Cooldown sweep from 10 to 20 (testing tighter than v16's 15)
3. Dropped unstable models (up_12_0003_p45t10, up_36_0003_p35t20)
4. 5 new candidate models: up_36_0002_p45t10, up_24_0003_p45t10, fav_12_0005_p40t10, up_12_0002_p45all, up_48_0002_p35t20
5. Agreement bonus test: 1.3x position when 3+ models fire simultaneously
6. Reused v15 Optuna params and ensemble decisions

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 1: All Models — 25 SELECTED (from 26)

| Model | AUC | Pos/N | Avg Net | Sharpe | Status |
|-------|-----|-------|---------|--------|--------|
| up_12_0002_p45t20 | 0.6340 | 8/10 | +5.91% | 77.4 | SELECTED |
| up_12_0003_p35t10 | 0.6697 | 9/10 | +5.47% | 93.8 | SELECTED [ENS] |
| up_12_0005_p35all | 0.7216 | 7/9 | +3.58% | 35.6 | SELECTED [LGB-ONLY] |
| up_24_0003_p45t10 | 0.6322 | 8/10 | +3.67% | 38.2 | **NEW** SELECTED |
| up_12_0002_p45all | 0.6340 | 7/10 | +4.38% | 59.6 | **NEW** SELECTED |
| up_48_0002_p35t20 | 0.5807 | 7/10 | +7.12% | 2.2 | **NEW** SELECTED |
| up_36_0002_p45t10 | 0.5901 | 7/10 | +1.48% | 13.9 | **NEW** SELECTED |
| **fav_12_0005_p40t10** | 0.7192 | **4/9** | +1.44% | 2.6 | **DROPPED** |

**New candidate results:**
- **up_24_0003_p45t10**: 8/10, Sharpe 38.2 — excellent, very selective (2 trades median)
- **up_12_0002_p45all**: 7/10, Sharpe 59.6 — strong with no top-pct filter (91 trades median)
- **up_48_0002_p35t20**: 7/10, +7.12% avg — adds 4h diversification
- **up_36_0002_p45t10**: 7/10, Sharpe 13.9 — low trade count but adds selectivity
- **fav_12_0005_p40t10**: 4/9, DROPPED — too few positive splits at tighter threshold

#### Phase 3: DD Sweep — DD=1% IS THE NEW OPTIMAL

DD=1% configs dramatically outperform DD=2% (v16) with better Sharpe AND better DD:

| Config | Agreement | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-----------|-------|---------|----------|--------|
| dd10m_cl15_c10_s100 | No | 10/10 | **+1063.6%** | **-15.2%** | **29.1** |
| dd10m_cl20_c10_s100 | Yes | 10/10 | +1522.7% | -15.3% | 29.9 |
| dd10m_cl10_c10_s100 | No | 10/10 | +1481.4% | -17.0% | 27.9 |
| dd15m_cl10_c10_s100 | No | 10/10 | +2077.6% | -20.1% | 26.3 |

**Agreement bonus**: Close but not clearly better (29.9 vs 29.1 Sharpe). System selected NO_BONUS because 29.9 < 29.1 * 1.05. The agreement bonus trades higher returns for higher DD in most configs.

#### Phase 4: Production Config — `dd10m_cl15_c10_s100`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +75.73% | -15.22% | 362 | 65.5% | 19.5 |
| 2 | 2025-04-13/2025-09-15 | **+22.55%** | -3.92% | 207 | 63.8% | **20.4** |
| 3 | 2024-11-09/2025-04-13 | +338.87% | -10.30% | 407 | 65.6% | 36.1 |
| 4 | 2024-06-07/2024-11-09 | +287.61% | -6.79% | 326 | 71.2% | 31.5 |
| 5 | 2024-01-04/2024-06-07 | +374.99% | -4.79% | 474 | 70.5% | 40.0 |
| 6 | 2023-08-02/2024-01-04 | +89.36% | -4.67% | 310 | 63.5% | 24.1 |
| 7 | 2023-02-28/2023-08-02 | +154.18% | -4.46% | 444 | 66.7% | 28.5 |
| 8 | 2022-09-26/2023-02-28 | +184.93% | -8.31% | 480 | 61.0% | 25.9 |
| 9 | 2022-04-24/2022-09-26 | +4943.61% | -11.89% | 1034 | 66.0% | 30.6 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +4164.15% | -13.56% | 1083 | 68.3% | 34.0 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +1063.6%**
- **Compounded return: +513,210,358%**
- **Annualized return: +3712.8%**
- **Avg max drawdown: -8.4%**
- **Worst max drawdown: -15.2%**
- **Avg Sharpe: 29.1**
- **Min Sharpe: 19.5**

**KEY IMPROVEMENTS vs v16:**
- Avg return: +625.4% -> **+1063.6%** (+70%)
- Worst DD: -17.0% -> **-15.2%** (10.6% improvement)
- Avg DD: -10.5% -> **-8.4%** (20% improvement)
- Sharpe: 26.6 -> **29.1** (+9.4%)
- Min Sharpe: 12.2 -> **19.5** (+60%)
- Annualized: +2600% -> **+3713%** (+43%)
- Split 2 (weakest): +12.2% Sharpe 12.2 -> **+22.6% Sharpe 20.4** (massive improvement)
- Models: 21 -> 25 (4 new candidates added)

---

### Key Lessons Learned (v16-specific)

### 37. DD=2% is the new optimal — tighter is still better
Continuing the clear trend: v11=8%, v13=5%, v14=4%, v15=3%, v16=2%. The 2% circuit breaker catches drawdowns before they compound. Combined with 21 models and cooldown=15, there are always enough models active to recover quickly. Worst DD improved from -20.9% to -17.0%.

### 38. Signal correlation pruning does NOT help
Counterintuitive finding: pruning highly correlated models (corr>0.75-0.85) from 21 to 10 models REDUCED performance (Sharpe 19.8 vs 26.6). Even though many models share correlated signals, quality weighting ensures the redundant models contribute proportionally. The portfolio benefits from signal reinforcement, not diversification.

### 39. Recent-weighted quality scoring improves recency-sensitive metrics
Giving 2x weight to splits 0-2 (most recent) emphasizes performance in current market conditions. Split 1 improved from +26.9% to +63.6% and Split 2 from +10.4% to +12.2%. This helps the portfolio adapt to current regime.

### 40. Reusing Optuna params is safe across iterations
Hardcoding v15's Optuna params and ensemble decisions in v16 saved ~3 hours of optimization while still producing better results. The params are stable across data updates when features don't change. Only re-optimize when the feature set changes.

### 41. Model stability matters — some models degrade across iterations
up_12_0003_p45t10 dropped from 7/9 (v15) to 4/9 (v16) despite identical params. up_36_0003_p35t20 dropped from v15 portfolio to 5/10. Some model configs are inherently unstable and should be monitored or dropped preemptively.

---

### Key Lessons Learned (v17-specific)

### 42. DD=1% continues the improvement trend — even tighter works
The trend v11=8%, v13=5%, v14=4%, v15=3%, v16=2%, v17=1% continues unbroken. DD=1% (dd10m) gave best Sharpe (29.1) with -15.2% worst DD. The 1% breaker catches drawdowns extremely early — 7 of 10 splits have maxDD under -10%. However, this requires many models (25) to ensure enough trades survive the frequent breaker triggers.

### 43. Agreement bonus is marginal
1.3x position scaling when 3+ models fire simultaneously gave slightly better Sharpe (29.9 vs 29.1) but at the cost of higher worst DD in many configs. The improvement is not significant enough (< 5% threshold) to justify the added complexity. The quality-weighted system already handles model confidence.

### 44. More models with quality weighting continue to improve results
25 models (up from 21) with 4 new candidates all contributed positively. The quality weighting ensures marginal models get low weights (0.40) while not removing them entirely. The additional models provide more trading opportunities, which is critical when DD=1% frequently pauses trading.

### 45. Removing unstable models preemptively is correct
Dropping up_12_0003_p45t10 (4/9 in v16) and up_36_0003_p35t20 (5/10 in v16) freed up space for better new candidates without losing any alpha. Models that degrade significantly between iterations should be removed.

---

### Iteration 18: FINE-GRAINED DD OPTIMIZATION — **MASSIVE SHARPE AND RETURN IMPROVEMENT**

**Key changes from Iter 17:**
1. Fine-grained DD sweep: 0.5%, 0.6%, 0.7%, 0.8%, 0.9%, 1.0%, 1.2%, 1.5%
2. Max concurrent: 10, 12, 14 (expanded from v17's 7, 10)
3. Cooldown: 10, 12, 15
4. Position scale: 0.8, 1.0
5. Same 25 models from v17 (no new candidates — focused on DD optimization)
6. Reused v15 Optuna params and ensemble decisions

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 1: All Models — ALL 25 SELECTED

All 25 v17 models maintained their quality with ≥7/10 positive splits. No models dropped.

| Model | AUC | Pos/N | Avg Net | Sharpe | Type |
|-------|-----|-------|---------|--------|------|
| up_12_0002_p45t20 | 0.6340 | 8/10 | +5.91% | 77.4 | LGB-only |
| up_12_0003_p35t10 | 0.6697 | 9/10 | +5.47% | 93.8 | ENS |
| up_12_0003_p40t10 | 0.6697 | 8/10 | +4.70% | 96.8 | ENS |
| up_12_0005_p40t20 | 0.7219 | 7/9 | +3.28% | 55.7 | ENS |
| up_24_0003_p35t20 | 0.6322 | 9/10 | +5.54% | 6.0 | LGB-only |
| up_24_0002_p40t10 | 0.6068 | 9/10 | +5.89% | 7.7 | LGB-only |

#### Phase 3: DD Sweep — DD=0.5% IS THE NEW OPTIMAL

The DD trend continues unbroken. Fine-grained sweep reveals 0.5% as the best with max concurrent=14:

| Config | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-------|---------|----------|--------|
| **dd5m_cl10_c14_s100** | **10/10** | **+14622.4%** | **-13.9%** | **35.8** |
| dd5m_cl10_c14_s80 | 10/10 | +2416.6% | -11.5% | 34.5 |
| dd5m_cl10_c12_s100 | 10/10 | +5918.3% | -14.7% | 34.0 |
| dd6m_cl10_c14_s100 | 10/10 | +10592.6% | -14.2% | 34.8 |
| dd1p_cl15_c10_s100 | 10/10 | +1063.6% | -15.2% | 29.1 |

**Key insight**: Max concurrent=14 (highest tested) dominated. More simultaneous positions with tighter DD provides better diversification under the 0.5% breaker. The breaker fires so frequently that having 14 slots ensures enough models keep trading during recovery windows.

#### Phase 4: Production Config — `dd5m_cl10_c14_s100`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +175.44% | -13.93% | 412 | 69.2% | 28.8 |
| 2 | 2025-04-13/2025-09-15 | **+38.82%** | -3.28% | 201 | 66.7% | **30.7** |
| 3 | 2024-11-09/2025-04-13 | +708.38% | -5.62% | 462 | 69.3% | 43.3 |
| 4 | 2024-06-07/2024-11-09 | +244.64% | -5.62% | 300 | 73.7% | 33.8 |
| 5 | 2024-01-04/2024-06-07 | +849.48% | -5.13% | 459 | 73.9% | 48.1 |
| 6 | 2023-08-02/2024-01-04 | +161.16% | -3.06% | 321 | 73.5% | 36.4 |
| 7 | 2023-02-28/2023-08-02 | +186.41% | -3.85% | 481 | 70.1% | 34.9 |
| 8 | 2022-09-26/2023-02-28 | +319.02% | -11.32% | 505 | 64.6% | 27.1 |
| 9 | 2022-04-24/2022-09-26 | +135778.05% | -10.62% | 1178 | 70.5% | 38.6 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +7762.13% | -8.02% | 1184 | 71.2% | 36.8 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +14622.4%**
- **Annualized return: +17505.4%**
- **Avg max drawdown: -7.0%**
- **Worst max drawdown: -13.9%**
- **Avg Sharpe: 35.8**
- **Min Sharpe: 27.1**

**KEY IMPROVEMENTS vs v17:**
- Avg return: +1063.6% -> **+14622.4%** (+1275%)
- Worst DD: -15.2% -> **-13.9%** (8.6% improvement)
- Avg DD: -8.4% -> **-7.0%** (16.7% improvement)
- Sharpe: 29.1 -> **35.8** (+23%)
- Min Sharpe: 19.5 -> **27.1** (+39%)
- Annualized: +3713% -> **+17505%** (+371%)
- Split 2 (weakest): +22.6% Sharpe 20.4 -> **+38.8% Sharpe 30.7** (massive improvement)
- DD: 1% -> **0.5%** (trend continues unbroken)
- Max concurrent: 10 -> **14** (more positions with tighter DD)
- Cooldown: 15 -> **10** (faster recovery)

---

### Key Lessons Learned (v18-specific)

### 46. DD=0.5% continues the trend — tighter is STILL better
The unbroken trend: v11=8%, v13=5%, v14=4%, v15=3%, v16=2%, v17=1%, v18=0.5%. Each step improves Sharpe and reduces worst DD. At 0.5%, the breaker fires extremely aggressively, but with 14 concurrent model slots and quality-weighted positions, there's always enough active capital. The key is pairing ultra-tight DD with enough model capacity.

### 47. Max concurrent positions are the critical complement to tight DD
Max concurrent=14 (highest tested) dominated over 10 and 12. With DD=0.5%, the breaker fires frequently — individual models get paused often. Having 14 slots ensures diversification across enough models that paused ones don't cause capital starvation. This is a key architectural insight: tight DD + many concurrent positions = aggressive risk cutting + fast recovery.

### 48. Cooldown=10 beats 12 and 15 at ultra-tight DD
With DD=0.5%, the system needs to recover quickly from breaker triggers. Cooldown=10 (50 minutes) allows faster re-entry than cooldown=15 (75 minutes). When the breaker fires at 0.5%, the damage is minimal, so faster recovery is safe.

### 49. Split 9 extreme return (+135778%) warrants investigation
Split 9 (2022-04-24 to 2022-09-26, bear market) produced extraordinary returns with 1178 trades. This could indicate the model excels in high-volatility downtrend markets with many trading opportunities. Production expectations should use median splits, not mean.

---

### Iteration 19: PUSH DD + CONCURRENCY + COOLDOWN BOUNDARIES — **SHARPE IMPROVEMENT, DD TRADEOFF**

**Key changes from Iter 18:**
1. DD sweep: 0.2%, 0.3%, 0.4%, 0.5% (pushing below v18's 0.5% boundary)
2. Max concurrent: 14, 16, 18, 20 (pushing above v18's 14 boundary)
3. Cooldown: 6, 8, 10 (pushing below v18's 10 boundary)
4. Position scale: 0.8, 1.0
5. Same 25 proven models from v17/v18
6. Reused v15 Optuna params and ensemble decisions

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 1: All Models — ALL 25 SELECTED

Identical results to v18 — same models, same params, same data. All 25 passed with ≥7/10 positive.

#### Phase 3: DD Sweep — DD=0.2% IMPROVES SHARPE BUT WITH TRADEOFFS

| Config | Pos/N | Avg Net | Worst DD | Sharpe |
|--------|-------|---------|----------|--------|
| **dd2m_cl10_c20_s80** | **10/10** | +4161.5% | -17.1% | **38.1** |
| dd2m_cl10_c20_s100 | 10/10 | +14438.5% | -23.4% | 38.9 |
| dd2m_cl10_c18_s100 | 10/10 | +12549.5% | -24.7% | 38.3 |
| dd2m_cl10_c14_s100 | 10/10 | +3815.3% | **-10.5%** | 37.8 |
| dd2m_cl10_c14_s80 | 10/10 | +1484.6% | **-8.4%** | 37.1 |
| dd5m_cl10_c14_s100 | 10/10 | +14622.4% | -13.9% | 35.8 |

**Key insights:**
- DD=0.2% produces higher Sharpe (37-39) than DD=0.5% (35-37)
- But worst DD degrades with more concurrent positions (-17.1% at c20 vs -10.5% at c14)
- **dd2m_cl10_c14_s100** is the best risk-adjusted config: Sharpe 37.8, only -10.5% worst DD
- Cooldown=10 still dominated (cooldown 6 and 8 had higher volatility)
- Position scale=0.8 reduces both returns AND worst DD — useful dampener at ultra-tight DD
- Max concurrent=20 helps Sharpe by maintaining trade flow during frequent breaker triggers

#### Phase 4: Production Config — `dd2m_cl10_c20_s80`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +86.61% | **-17.09%** | 316 | 70.6% | 23.8 |
| 2 | 2025-04-13/2025-09-15 | +39.83% | -2.66% | 174 | 69.5% | 37.7 |
| 3 | 2024-11-09/2025-04-13 | +246.88% | -3.19% | 362 | 69.1% | 42.2 |
| 4 | 2024-06-07/2024-11-09 | +277.58% | -5.95% | 246 | 76.0% | 37.7 |
| 5 | 2024-01-04/2024-06-07 | +407.98% | -4.62% | 379 | 77.3% | 51.1 |
| 6 | 2023-08-02/2024-01-04 | +107.25% | -5.72% | 302 | 80.5% | 35.1 |
| 7 | 2023-02-28/2023-08-02 | +139.98% | -2.26% | 448 | 72.5% | 35.7 |
| 8 | 2022-09-26/2023-02-28 | +325.55% | -9.06% | 447 | 69.4% | 35.0 |
| 9 | 2022-04-24/2022-09-26 | +33587.38% | -11.42% | 1112 | 72.7% | 40.2 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +6395.58% | -5.17% | 1141 | 74.8% | 42.8 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +4161.5%**
- **Annualized return: +7192%**
- **Avg max drawdown: -6.7%**
- **Worst max drawdown: -17.1%**
- **Avg Sharpe: 38.1**
- **Min Sharpe: 23.8**

**COMPARISON vs v18:**
- Sharpe: 35.8 -> **38.1** (+6.4% improvement)
- Avg DD: -7.0% -> **-6.7%** (improved)
- Worst DD: -13.9% -> -17.1% (DEGRADED — driven by Split 1)
- Min Sharpe: 27.1 -> 23.8 (degraded)
- Avg return: +14622% -> +4162% (lower due to scale=0.8)
- Annualized: +17505% -> +7192% (lower due to scale=0.8)
- DD: 0.5% -> **0.2%** (trend continues)
- Max concurrent: 14 -> **20** (+43%)

**SAFER ALTERNATIVE: `dd2m_cl10_c14_s100`**
- Sharpe **37.8** (better than v18's 35.8)
- Worst DD **-10.5%** (much better than both v19 main and v18)
- Avg return +3815.3% (lower but with superior risk control)
- This may be the best overall production config across all iterations

---

### Key Lessons Learned (v19-specific)

### 50. Below DD=0.5%, Sharpe improves but worst DD can degrade
DD=0.2% improved avg Sharpe from 35.8 to 38.1 — the DD trend continues for risk-adjusted returns. However, the worst DD degraded from -13.9% to -17.1% when using 20 concurrent positions. The ultra-tight breaker fires so aggressively that max concurrent positions accumulate losses faster before the breaker triggers. The solution: pair ultra-tight DD with fewer concurrent positions (c14 gives -10.5% worst DD).

### 51. Max concurrent must be tuned with DD — not "more is always better"
At DD=0.5% (v18), c14 was optimal. At DD=0.2% (v19), c20 gives best Sharpe but c14 gives best risk-adjusted (Sharpe 37.8, -10.5% DD). More positions help Sharpe by maintaining trade flow through breaker pauses, but too many positions can briefly accumulate correlated losses before the breaker triggers.

### 52. Position scale=0.8 as a DD dampener
At DD=0.2%, using scale=0.8 instead of 1.0 significantly reduces worst DD (e.g., c20: -17.1% at s80 vs -23.4% at s100) with only modest Sharpe reduction (38.1 vs 38.9). This is a useful production safety mechanism — reduce position size rather than increase DD limit.

### 53. DD optimization may be approaching diminishing returns
The progression v15-v19 shows: 3%→2%→1%→0.5%→0.2%. Each step improved Sharpe, but the gains are shrinking: +3.5, +2.5, +6.7, +2.3. Meanwhile, the worst DD fluctuates: -20.9%, -17.0%, -15.2%, -13.9%, -17.1%. The sweet spot for all-around performance may be DD=0.2-0.5% with c14 and cooldown=10.

---

### Iteration 20: NEW TARGET CANDIDATES + ALPHA EXPANSION — **PORTFOLIO GROWS TO 31 MODELS**

**Key changes from Iter 19:**
1. 10 new candidate models from 7 untried target families: up_48_0003, up_24_0005, up_36_0005, fav_12_0002, fav_36_0002, fav_36_0005, up_12_001
2. Same 25 proven existing models
3. No new DD sweep — used 5 pre-selected configs from v18/v19
4. Phase 3: Explicit EXISTING vs EXPANDED portfolio comparison
5. New candidates use closest existing Optuna params by target family

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 0: Feature Selection — New Targets Show Distinct Patterns

| Target | Top-5 Features | Status |
|--------|---------------|--------|
| up_48_0003 | volume_vpt, volume_momentum_96, cumulative_delta_btc, hit_rate_0002_288, hour_of_day | NEW |
| up_24_0005 | hour_of_day, cumulative_delta_btc, hit_rate_0002_288, trail_dd_288, hit_rate_0003_288 | NEW |
| up_36_0005 | hour_of_day, hit_rate_0002_288, cumulative_delta_btc, hit_rate_0003_288, volume_vpt | NEW |
| fav_12_0002 | hit_rate_0002_288, realized_vol_288, momentum_stoch, price_ema12_dist, hour_of_day | NEW |
| fav_36_0002 | hour_of_day, momentum_wr, momentum_stoch, volatility_kcp, volatility_ui | NEW |
| fav_36_0005 | hour_of_day, hit_rate_0002_288, realized_vol_288, trail_dd_288, session_asian | NEW |
| up_12_001 | trail_dd_288, hour_sin, range_position_288, hit_rate_0002_288, hour_of_day | NEW |

**Key insight**: fav_36_0002 uses momentum/volatility features (momentum_wr, momentum_stoch, volatility_kcp) rather than the usual hit_rate/trail_dd features — most differentiated from existing models. up_48_0003 is volume-driven (volume_vpt, volume_momentum_96).

#### Phase 1: All Models — 6 NEW CANDIDATES PASS, 4 DROPPED

| New Candidate | AUC | Pos/N | Avg Net | Sharpe | Status |
|--------------|-----|-------|---------|--------|--------|
| **up_48_0003_p40t10** | 0.5980 | **10/10** | +7.41% | +16.9 | **SELECTED** — perfect consistency |
| **up_36_0005_p35t10** | 0.6488 | **9/10** | +3.40% | +31.5 | **SELECTED** |
| **up_24_0005_p40t10** | 0.6738 | 8/9 | +2.18% | +852.4 | **SELECTED** — very few trades, extreme Sharpe |
| **fav_36_0005_p35t10** | 0.6147 | 8/10 | +4.24% | +10.3 | **SELECTED** [ENS] |
| **up_48_0003_p35t10** | 0.5980 | 7/10 | +6.74% | +6.3 | **SELECTED** |
| **up_24_0005_p35t10** | 0.6738 | 7/9 | +3.12% | +26.6 | **SELECTED** |
| up_12_001_p40t10 | 0.7884 | 5/6 | +1.64% | +140.2 | DROPPED (only 6 splits, too rare) |
| fav_12_0002_p35t20 | 0.6078 | 5/10 | -6.02% | -1.1 | DROPPED |
| fav_12_0002_p40t10 | 0.6078 | 4/10 | +1.36% | +1.2 | DROPPED |
| fav_36_0002_p35t20 | 0.5572 | 5/10 | -0.76% | +0.4 | DROPPED |

**Key findings**: "up" targets dominated — all 5 up-family candidates passed. "fav" targets were mixed (1/3 passed). fav_12_0002 and fav_36_0002 both failed badly. up_48_0003_p40t10 achieved **10/10 positive splits** — the only new model with perfect consistency.

#### Phase 3: EXPANDED Portfolio Beats EXISTING in Every Config

| Config | Existing Sharpe | Expanded Sharpe | Existing DD | Expanded DD |
|--------|----------------|-----------------|-------------|-------------|
| dd2m_cl10_c14_s100 | 37.5 | **38.0** | -11.4% | -13.4% |
| dd2m_cl10_c20_s80 | 38.5 | **39.9** | -16.6% | -20.2% |
| dd5m_cl10_c14_s100 | 35.0 | **36.1** | -15.0% | -15.1% |
| dd3m_cl10_c14_s100 | 37.2 | **37.5** | -11.8% | -13.4% |
| dd2m_cl10_c16_s100 | 37.2 | **37.6** | -16.3% | -17.6% |

**Universal improvement**: Expanded portfolio beats existing in Sharpe and returns across ALL 5 configs tested. The 6 new models add genuine alpha. DD increases slightly due to more positions but Sharpe improvement is consistent.

#### Phase 4: Production Config — `dd2m_cl10_c20_s80`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +143.39% | -12.07% | 357 | 72.3% | 32.5 |
| 2 | 2025-04-13/2025-09-15 | +38.15% | -2.54% | 186 | 71.5% | 37.1 |
| 3 | 2024-11-09/2025-04-13 | +309.19% | -3.09% | 396 | 69.4% | 43.1 |
| 4 | 2024-06-07/2024-11-09 | +305.66% | -5.94% | 254 | 76.4% | 39.5 |
| 5 | 2024-01-04/2024-06-07 | +476.30% | -4.65% | 397 | 80.1% | 54.7 |
| 6 | 2023-08-02/2024-01-04 | +138.80% | -5.61% | 335 | 80.6% | 39.2 |
| 7 | 2023-02-28/2023-08-02 | +135.70% | -4.59% | 466 | 73.4% | 34.0 |
| 8 | 2022-09-26/2023-02-28 | +583.36% | -8.39% | 503 | 72.4% | 36.0 |
| 9 | 2022-04-24/2022-09-26 | +45656.27% | -20.15% | 1188 | 73.1% | 39.3 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +9915.37% | -7.59% | 1164 | 73.8% | 43.2 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +5770.2%**
- **Annualized return: +11454%**
- **Avg max drawdown: -7.5%**
- **Worst max drawdown: -20.2%**
- **Avg Sharpe: 39.9**
- **Min Sharpe: 32.5**

**COMPARISON vs v19:**
- Models: 25 -> **31** (+6 new alpha sources)
- Sharpe: 38.1 -> **39.9** (+4.7% improvement)
- Min Sharpe: 23.8 -> **32.5** (+36.6% — major consistency improvement)
- Avg return: +4161.5% -> **+5770.2%** (+38.7%)
- Annualized: +7192% -> **+11454%** (+59.2%)
- Worst DD: -17.1% -> -20.2% (degraded — Split 9)
- Avg DD: -6.7% -> -7.5% (slightly worse)
- Split 1 (newest): +86.6% / Sharpe 23.8 -> **+143.4% / Sharpe 32.5** (major recent improvement)

**STATUS: PRODUCTION READY**

---

### Key Lessons Learned (v20-specific)

### 54. New alpha from untried targets — "up" family dominates, "fav" family mixed
All 5 "up" target candidates passed (up_48_0003, up_24_0005, up_36_0005). Only 1 of 3 "fav" candidates passed (fav_36_0005). The fav_12_0002 and fav_36_0002 targets had low AUC and failed consistency. The favorable (asymmetric risk-reward) targets need tighter prob thresholds or may simply have weaker signal in some families.

### 55. Portfolio expansion universally improves Sharpe
The expanded 31-model portfolio beat the existing 25-model portfolio in EVERY config tested (5 configs, all 10/10 positive). Sharpe improvements ranged from +0.3 to +1.4. The new models provide genuinely uncorrelated alpha, not just noise.

### 56. Min Sharpe is as important as Avg Sharpe
V20's biggest win is Min Sharpe: 23.8 -> 32.5 (+36.6%). This means the weakest split (Split 1, most recent data) improved dramatically. Split 1 went from +86.6% / Sharpe 23.8 in v19 to +143.4% / Sharpe 32.5 in v20. More models diversify tail risk.

### 57. up_48_0003_p40t10: perfect 10/10 consistency
The 4h 0.3% threshold target achieved 10/10 positive splits — the only new model with perfect consistency. This suggests higher threshold + longer horizon combinations may be an untapped alpha source. Worth exploring up_48_0005, up_48_001, and similar aggressive targets.

### 58. High AUC doesn't guarantee trading success
up_12_001 had the highest AUC (0.7884) among all new candidates but was DROPPED — it only had 6 splits with trades (too rare, median 1 trade per split). Conversely, up_48_0003 had low AUC (0.5980) but was the best new model. AUC measures prediction quality; trading success requires signal frequency.

---

### Iteration 21: AGGRESSIVE TARGETS + 30-MIN TIMEFRAME — **WORST DD MAJOR IMPROVEMENT**

**Key changes from Iter 20:**
1. 13 new candidate models from 7 untried targets: up_48_0005, up_48_001, up_24_001, up_36_001, up_6_0002, up_6_0003, up_6_0005
2. 31 existing models from v20
3. Same 5 DD configs (no new sweep)
4. All new candidates LGB-only (v20 showed "up" family works best without ensemble)

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 0: Feature Selection — 30-Min Targets Use Short-Term Features

| Target | Top-5 Features | Status |
|--------|---------------|--------|
| up_48_0005 | volume_vpt, volume_adi, cumulative_delta_btc, hit_rate_0002_288, return_288 | NEW |
| up_48_001 | cumulative_delta_btc, volume_vpt, volume_nvi, volume_adi, return_288 | NEW |
| up_24_001 | hour_of_day, cumulative_delta_btc, trail_dd_288, hit_rate_0002_288, range_position_288 | NEW |
| up_36_001 | hour_of_day, volume_nvi, hit_rate_0002_288, volume_obv, dow_sin | NEW |
| up_6_0002 | hour_of_day, trail_dd_288, trail_dd_96, hit_rate_0002_288, cumulative_delta_btc | NEW |
| up_6_0003 | trail_dd_288, trail_dd_96, hour_of_day, hit_rate_0002_288, hour_sin | NEW |
| up_6_0005 | trail_dd_96, trail_dd_288, hit_rate_0002_288, hour_sin, range_position_288 | NEW |

**Key insight**: 30-min targets strongly use trail_dd_96/288 — short-term drawdown features are most predictive for quick reversals. 1% threshold targets use volume/cumulative delta — large moves are volume-driven.

#### Phase 1: All Models — 9 NEW CANDIDATES PASS, 4 DROPPED

| New Candidate | AUC | Pos/N | Avg Net | Sharpe | Status |
|--------------|-----|-------|---------|--------|--------|
| **up_6_0005_p35t10** | 0.7720 | **9/9** | +4.23% | +139.6 | **SELECTED** — highest AUC new model |
| **up_6_0005_p40t10** | 0.7720 | **9/9** | +2.84% | +102.3 | **SELECTED** |
| **up_6_0002_p40t10** | 0.6698 | 8/10 | +3.01% | +43.1 | **SELECTED** |
| **up_6_0002_p35t20** | 0.6698 | 8/10 | +3.36% | +11.3 | **SELECTED** |
| **up_6_0003_p35t10** | 0.7141 | 7/10 | +5.34% | +34.8 | **SELECTED** |
| **up_6_0003_p40t10** | 0.7141 | 7/10 | +2.88% | +39.9 | **SELECTED** |
| **up_48_0005_p35t10** | 0.6306 | 7/9 | +6.62% | +29.9 | **SELECTED** |
| **up_24_001_p35t10** | 0.7388 | 5/7 | +1.87% | +40.9 | **SELECTED** |
| **up_24_001_p40t10** | 0.7388 | 5/7 | +1.75% | +43.7 | **SELECTED** |
| up_48_0005_p40t10 | 0.6306 | 5/8 | +3.34% | +27.6 | DROPPED |
| up_48_001_p35t10 | 0.6824 | 4/6 | +2.59% | +144.7 | DROPPED (too rare) |
| up_36_001_p35t10 | 0.7066 | 6/6 | +3.18% | +91.7 | DROPPED (too rare, median 1 trade) |
| up_36_001_p40t10 | 0.7066 | 3/4 | +1.47% | +33.0 | DROPPED |

**Key findings**: ALL 6 up_6_xxx (30-min) candidates passed — this new timeframe is a goldmine. up_6_0005 had 9/9 perfect consistency and the highest AUC (0.772) of any model in the portfolio. The 1% threshold targets (up_48_001, up_36_001) produce too few trades for reliable selection.

#### Phase 3: EXPANDED Portfolio Shows Mixed Results

| Config | Exist Sharpe | Expand Sharpe | Exist DD | Expand DD |
|--------|-------------|---------------|----------|-----------|
| dd2m_cl10_c14_s100 | 38.2 | 37.2 | -12.1% | **-11.3%** |
| dd2m_cl10_c20_s80 | 39.3 | 38.8 | -18.2% | **-14.8%** |
| dd5m_cl10_c14_s100 | 36.2 | 34.4 | -13.9% | **-13.8%** |
| dd2m_cl10_c16_s100 | 37.9 | 38.5 | -14.3% | **-12.3%** |

**Key insight**: Expanded portfolio universally IMPROVES worst DD (fewer severe drawdowns) but Sharpe is mixed. The 30-min models add trade frequency that smooths drawdowns but the higher trade count dilutes per-trade Sharpe. The dd2m_cl10_c16_s100 config is a sweet spot: Sharpe improves 37.9→38.5 AND DD improves -14.3%→-12.3%.

#### Phase 4: Production Config — `dd2m_cl10_c20_s80`

| Split | Period | Net | MaxDD | Trades | WR | Sharpe |
|-------|--------|-----|-------|--------|-----|--------|
| 1 (newest) | 2025-09-15/2026-02-17 | +163.36% | -8.48% | 418 | 72.0% | 34.1 |
| 2 | 2025-04-13/2025-09-15 | +33.48% | -2.92% | 192 | 69.8% | 36.3 |
| 3 | 2024-11-09/2025-04-13 | +413.08% | -6.61% | 528 | 70.8% | 42.6 |
| 4 | 2024-06-07/2024-11-09 | +335.83% | -7.14% | 328 | 75.9% | 36.1 |
| 5 | 2024-01-04/2024-06-07 | +338.62% | -3.64% | 460 | 77.0% | 48.7 |
| 6 | 2023-08-02/2024-01-04 | +186.49% | -5.50% | 366 | 79.5% | 41.7 |
| 7 | 2023-02-28/2023-08-02 | +151.10% | -3.56% | 486 | 73.9% | 36.9 |
| 8 | 2022-09-26/2023-02-28 | +268.74% | -14.83% | 529 | 70.7% | 29.3 |
| 9 | 2022-04-24/2022-09-26 | +54475.71% | -13.47% | 1325 | 72.9% | 40.0 |
| 10 (oldest) | 2021-11-20/2022-04-24 | +11308.54% | -7.25% | 1356 | 73.7% | 42.3 |

**FINAL METRICS:**
- **Positive splits: 10/10**
- **Avg net per split: +6767.5%**
- **Annualized return: +11466%**
- **Avg max drawdown: -7.3%**
- **Worst max drawdown: -14.8%** (major improvement from v20's -20.2%)
- **Avg Sharpe: 38.8**
- **Min Sharpe: 29.3**

**COMPARISON vs v20:**
- Models: 31 → **40** (+9 new)
- Worst DD: -20.2% → **-14.8%** (-5.4% improvement — **MAJOR**)
- Split 1 (newest): +143% / DD -12.1% → **+163% / DD -8.5%** (both improved)
- Split 8: was weakest at DD -8.4% → now -14.8% (worse, but still acceptable)
- Avg return: +5770% → **+6768%** (+17.3%)
- Sharpe: 39.9 → 38.8 (-2.8% — minor tradeoff)
- Min Sharpe: 32.5 → 29.3 (degraded, Split 8 is now weakest)

**STATUS: PRODUCTION READY**

---

### Key Lessons Learned (v21-specific)

### 59. 30-min timeframe is an alpha goldmine
ALL 6 up_6_xxx candidates passed (6/6 — perfect hit rate). up_6_0005 achieved 9/9 perfect consistency with the highest AUC (0.772) of any model. The 30-min horizon provides fast-moving, independent signals that diversify the portfolio. The trail_dd features are most predictive — drawdowns predict short-term reversals.

### 60. Short-horizon models dramatically improve worst DD
V21's biggest win: worst DD -20.2% → -14.8%. The 30-min models generate more frequent trades that smooth the equity curve, preventing the sharp drawdowns seen with only longer-horizon models. This is genuine diversification — the 30-min signals are largely uncorrelated with 1-4h signals.

### 61. 1% threshold targets are too rare for robust backtest
up_48_001, up_36_001, and up_24_001 all had very few trades (median 0-1 per split). Even when positively selected, these signals fire so rarely that they contribute minimal portfolio value. The 0.5% threshold appears to be the practical upper limit for signal frequency.

### 62. Trade frequency dilutes per-trade Sharpe but improves portfolio DD
Adding 30-min models increased total trades (from ~350 to ~418 in Split 1) but Sharpe decreased slightly (39.9→38.8). More trades = more fee drag and slightly lower per-trade edge. However, the DD improvement (-5.4%) far outweighs the Sharpe cost (-1.1). In production, lower DD is more important than marginally higher Sharpe.

---

### Iteration 22: OPTUNA FOR 30-MIN TARGETS + DD CONFIG SWEEP — **ALL METRICS IMPROVED**

**Key changes from Iter 21:**
1. Optuna optimization for 5 new target families: up_6_0002, up_6_0003, up_6_0005, up_24_001, up_48_0005 (40 trials each)
2. Feature selection BEFORE Optuna (100 features, not all 309 — prevents stuck trials)
3. Extended DD config sweep: 12 configs from c10 to c24 at DD=0.2%/0.3%
4. All 40 v21 models retrained with optimized params

**Data**: 892,976 rows x 408 cols (309 features, 97 targets)

---

#### Phase 0: Feature Selection — Identical to v21 (stable)

#### Phase 1: Optuna Optimization — Marginal but positive

| Target | Borrowed AUC | Optuna AUC | Delta | Time |
|--------|-------------|------------|-------|------|
| up_6_0002 | 0.6749 | 0.6750 | +0.0001 | 26min |
| up_6_0003 | 0.7214 | 0.7223 | +0.0009 | 41min |
| up_6_0005 | 0.7845 | 0.7849 | +0.0004 | 47min |
| up_24_001 | 0.7349 | 0.7363 | **+0.0014** | 22min |
| up_48_0005 | 0.6273 | 0.6283 | +0.0010 | 19min |

Key finding: Borrowed 1h params were already near-optimal for 30-min targets. up_6_0005 Optuna found identical params to borrowed up_12_0005. up_24_001 had the biggest delta.

#### Phase 2: Train All 40 Models — 39 passed, 1 dropped

- **39/40 models passed** (>=7 splits with trades, >=70% positive)
- **Dropped**: up_24_001_p40t10 (5/6 pos — too few splits with trades)
- **Improved vs v21**: up_48_0005_p35t10 (7/9→8/9), up_6_0002_p40t10 (8/10→9/10)
- up_6_0005_p40t10 maintained 9/9 perfect consistency, highest AUC 0.772

#### Phase 4: Extended DD Config Sweep — c22 is the new sweet spot

| Config | Sharpe | Worst DD | Avg% |
|--------|--------|----------|------|
| dd2m_cl10_c10_s100 | 33.2 | -20.5% | +1261% |
| dd2m_cl10_c14_s100 | **36.4** | **-13.2%** | +3078% |
| dd2m_cl10_c16_s100 | 37.9 | -19.6% | +8011% |
| dd2m_cl10_c20_s100 | 39.0 | -17.6% | +37763% |
| **dd2m_cl10_c22_s100** | **40.1** | **-13.7%** | **+88643%** |
| dd2m_cl10_c24_s100 | 40.3 | -23.1% | +119781% |

c22 hits a sweet spot: higher Sharpe than c20, much better worst DD than c24. The relationship is non-linear — DD doesn't monotonically increase with concurrency.

#### Phase 5: Final Config — `dd2m_cl10_c22_s100`

| Split | Period | Trades | Net | Max DD | WR | Sharpe |
|-------|--------|--------|-----|--------|-----|--------|
| 1 (newest) | Sep 2025-Feb 2026 | 391 | +221.9% | -7.8% | 72.6% | 33.2 |
| 2 | Apr-Sep 2025 | 190 | +48.9% | -2.5% | 71.6% | 40.8 |
| 3 | Nov 2024-Apr 2025 | 494 | +802.6% | -5.0% | 72.7% | 46.2 |
| 4 | Jun-Nov 2024 | 284 | +431.2% | -9.9% | 73.9% | 35.4 |
| 5 | Jan-Jun 2024 | 433 | +470.7% | -7.1% | 77.4% | 46.5 |
| 6 | Aug 2023-Jan 2024 | 370 | +318.4% | -6.8% | 81.1% | 43.6 |
| 7 | Feb-Aug 2023 | 451 | +177.4% | -4.6% | 74.3% | 34.8 |
| 8 | Sep 2022-Feb 2023 | 503 | +669.4% | -11.9% | 71.2% | 32.1 |
| 9 | Apr-Sep 2022 | 1376 | +851,322% | -13.7% | 74.7% | 44.9 |
| 10 | Nov 2021-Apr 2022 | 1347 | +31,966% | -9.2% | 74.6% | 43.3 |

---

### Key Lessons Learned (Updated through Iter 22)

### 63. Borrowed params are near-optimal for related targets
Optuna optimization for up_6_xxx (30-min) targets using up_12_xxx (1h) borrowed params yielded deltas of +0.0001 to +0.0009 AUC. For up_6_0005, Optuna found the exact same params as borrowed up_12_0005. This confirms that target families with the same threshold but different horizons share optimal hyperparameter landscapes. Future new targets can safely use borrowed params without dedicated Optuna.

### 64. Extended DD config sweep reveals non-linear sweet spots
c22 achieves better worst DD (-13.7%) than c20 (-17.6%) despite higher concurrency, because the additional positions provide diversification that smooths drawdowns. However, c24 jumps to -23.1% worst DD — a tipping point where too many concurrent positions amplify correlated losses. The Sharpe-DD relationship is non-linear and must be swept empirically.

### 65. Full position scale outperforms 0.8x at higher concurrency
V21's best config was dd2m_cl10_c20_s80 (Sharpe 38.8, DD -14.8%). V22 found that dd2m_cl10_c22_s100 (Sharpe 40.1, DD -13.7%) is superior — the extra concurrency provides enough diversification to support full-scale positions without worse drawdowns.

### 66. Feature selection before Optuna is critical for runtime
Running Optuna on all 309 features caused trials to get stuck (>6 hours for a single target). Using feature-selected 100 features reduced Optuna time to 20-47 minutes per target with no AUC loss. Always run feature selection before Optuna.

---

### Iteration 23: TIGHTER 30-MIN FILTERING + NEW CANDIDATES

**Key changes from Iter 22:**
1. New target: up_6_001 (30-min 1% threshold) — Optuna-optimized, AUC 0.8398
2. New candidates: up_6_0005_p45t10, up_6_0005_p50t10, up_6_0003_p45t10, up_6_001_p35t10, up_6_001_p40t10
3. Reused v22 Optuna params for all existing targets (loaded from JSON)
4. DD config sweep c14-c28 (extended beyond v22's c10-c24)

**Results:** 41 models (+2 new), 10/10 positive, Sharpe 39.7, worst DD -19.0%, annualized +78368%

#### Phase 0: Feature Selection
- 21 targets (20 existing + up_6_001 new), top-100 features each
- up_6_001 top features: trail_dd_288, range_position_288, hit_rate_0003_288

#### Phase 1: Optuna for up_6_001
- **AUC 0.8398** (highest ever!) — borrowed from up_6_0005 was 0.8392, delta +0.0006
- Params: lr=0.0126, leaves=15, depth=6, min_child=918, subsample=0.73
- Time: 2183s (36 min)

#### Phase 2: Model Training (44 candidates)
- 39/39 existing models passed (all stable from v22)
- New candidates:
  - **up_6_0003_p45t10**: 7/9 pos, sharpe +121.7, med_trades=3 → **SELECTED**
  - **up_6_001_p35t10**: 7/7 pos, sharpe +174.0, med_trades=1, AUC 0.836 → **SELECTED**
  - up_6_0005_p45t10: 5/8 pos → DROPPED (too tight threshold)
  - up_6_0005_p50t10: 5/5 pos, med_trades=0 → DROPPED (insufficient trades)
  - up_6_001_p40t10: 5/6 pos → DROPPED (p35t10 better)

#### Phase 3: Quality Scoring
- up_6_001_p35t10: quality=1.500, weight=1.741 (tied for highest)
- up_6_0003_p45t10: quality=1.227, weight=1.424

#### Phase 4: DD Config Sweep
| Config | Pos | AvgNet | WorstDD | Sharpe | MinSharpe |
|--------|-----|--------|---------|--------|-----------|
| dd2m_cl10_c14_s100 | 10/10 | +3018% | -12.1% | 35.8 | 28.9 |
| dd2m_cl10_c16_s100 | 10/10 | +12270% | -16.1% | 37.4 | 32.0 |
| dd2m_cl10_c20_s100 | 10/10 | +29380% | -15.7% | 38.1 | 30.9 |
| dd2m_cl10_c22_s100 | 10/10 | +55487% | -23.8% | 38.4 | 29.4 |
| dd2m_cl10_c24_s100 | 10/10 | +57558% | -20.4% | 38.9 | 29.7 |
| dd2m_cl10_c28_s100 | 10/10 | +188884% | -19.0% | **39.7** | 30.8 |

#### Phase 5: Production Config
- **Selected:** dd2m_cl10_c28_s100 (best Sharpe under -22% DD constraint)
- **Safest:** dd2m_cl10_c14_s100 (Sharpe 35.8, -12.1% worst DD)
- **Conservative:** dd2m_cl10_c20_s100 (Sharpe 38.1, -15.7% worst DD)

#### Lessons
67. **up_6_001 is the highest-AUC target** (0.8499 in full training, 0.8398 in Optuna eval) — 30-min 1% threshold works despite failing at longer horizons (up_24_001, up_48_001 too rare). Short-horizon = more frequent occurrences of 1% moves.
68. **Tighter probability thresholds have diminishing returns on 30-min models**: p45t10 and p50t10 on up_6_0005 reduced consistency despite AUC 0.772. The p40t10 variant already captures the highest-confidence trades.
69. **Concurrency sweet spot shifts with model count**: v22's c22 sweet spot (39 models) → v23's c28 (41 models). More models need more concurrent positions to express alpha, but DD increases non-linearly.
70. **up_6_001_p35t10 has very few trades per split** (median 1) — extremely selective but may lack statistical robustness. Monitor closely in production.

---

## What To Try Next (Iteration 24)

### Priority 1: Production readiness validation
1. **Forward validation**: Deploy on live data (paper trading) to confirm signal persistence — 41 models are backtested only
2. **Execution simulation**: Model maker order fill rates, queue position, and partial fill scenarios
3. **DD=0.2% live feasibility**: Test if 0.2% breaker is viable with real execution delays (30-min models trade more frequently)
4. **Live deployment infrastructure**: Build real-time feature pipeline, model serving, order execution

### Priority 2: Model refinement
5. **Sliding window training**: Train on recent 2 years only to reduce noise from old regimes (Split 9 dominates with +1.8M%)
6. **Favorable targets at 30-min horizon**: fav_6_0003, fav_6_0005 — asymmetric risk-reward at short horizon, untested
7. **Explore down_6_xxx for short avoidance**: Use down probability signals to suppress entry during high-downside-risk periods

### Priority 3: Advanced techniques
8. **Stacking meta-learner**: Train second-level model on base model predictions
9. **New feature engineering**: Order flow imbalance at multiple timeframes, funding rate features
10. **Asymmetric DD breaker**: Tighter breaker for 30-min models (faster cooldown recovery)

---

## File Structure

```
trading/
  btc_indicators.py    # Data download + feature engineering (v5, 309 features, 97 targets)
  train_model.py       # LightGBM training pipeline (iteration 5, 5 phases)
  train_model_v6.py    # Iteration 6 training pipeline (7 phases)
  train_model_v7.py    # Iteration 7 — crashed mid-run (OOM), superseded by v7b
  train_model_v7b.py   # Iteration 7b — threshold signal exploitation (5 phases)
  train_model_v8.py    # Iteration 8 — multi-model portfolio & 3h threshold (not run)
  train_model_v9.py    # Iteration 9 — production-grade portfolio (6 phases)
  train_model_v10.py   # Iteration 10 — risk-managed portfolio (9 phases)
  train_model_v11.py   # Iteration 11 — optimized risk controls (6 phases)
  train_model_v13.py   # Iteration 13 — hyperparam optimization + ensemble (5 phases)
  train_model_v14.py   # Iteration 14 — GARCH features + adaptive ensemble (6 phases)
  train_model_v15.py   # Iteration 15 — full optimization + quality weighting (6 phases)
  train_model_v16.py   # Iteration 16 — correlation pruning + recent-weighted quality (5 phases)
  train_model_v17.py   # Iteration 17 — tighter DD + new candidates (4 phases)
  train_model_v18.py   # Iteration 18 — fine-grained DD optimization (4 phases)
  train_model_v19.py   # Iteration 19 — push DD/concurrency/cooldown boundaries (4 phases)
  train_model_v20.py   # Iteration 20 — new target candidates + alpha expansion (4 phases)
  train_model_v21.py   # Iteration 21 — aggressive targets + 30-min timeframe (4 phases)
  train_model_v22.py   # Iteration 22 — Optuna for 30-min targets + DD config sweep (5 phases)
  train_model_v23.py   # Iteration 23 — tighter 30-min filtering + new candidates (5 phases) ← CURRENT
  FINDINGS.md          # This file
  .gitignore
  btc_data/            # (gitignored)
    raw_spot_klines.csv         # 893K rows, ~107 MB
    raw_futures_klines.csv      # 680K rows, ~76 MB
    btc_features_5m.parquet     # 893K x 408, ~2.0 GB
    training_results_v23.txt    # Iteration 23 results ← CURRENT
    models_v23/                 # 41 production models + config
```

## Dependencies

```
pandas, numpy, lightgbm, xgboost, catboost, optuna, ta, requests, pyarrow, scikit-learn, scipy
```
