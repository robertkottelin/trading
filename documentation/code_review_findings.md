# Code Review Findings — BTC Trading System

> Two comprehensive code reviews performed on the full repository.
> Covers: execution safety, backtesting realism, data pipeline integrity, LLM agent logic.
> Issues marked **FIXED** have been resolved; remaining issues are open.

---

## 1. EXECUTION LAYER

### 1.1 No Orphan Order Cleanup on Startup
**File:** `execution/dydx_executor.py` | **Severity:** MEDIUM

**Problem:** If the pipeline crashes or is interrupted after placing an entry but before placing TP/SL (or after placing TP but before SL), orphan orders can remain on the dYdX order book. No cleanup mechanism runs on pipeline startup.

**Recommendation:** Add a startup check in `dydx_executor.py` or `run_pipeline.py` that queries open orders and cancels any that don't correspond to active positions.

**Status:** FIXED (2026-02-26) — Added `get_open_orders()` and `cancel_order()` to `DydxClient`, `cleanup_orphan_orders()` to `DydxExecutor`, and Stage 0 startup cleanup in `reasoning_agent.py` (live mode only, non-fatal on failure).

---

### 1.2 No Position Close Monitoring
**File:** `execution/dydx_executor.py` | **Severity:** MEDIUM

**Problem:** After placing TP/SL conditional orders, the executor has no mechanism to check if they actually filled. If a TP or SL order gets cancelled by the exchange (e.g., insufficient margin, self-trade prevention), the position continues unmonitored until the next pipeline cycle.

**Recommendation:** Add a position monitoring loop or a startup check that verifies positions still have active TP/SL orders.

**Status:** FIXED (2026-02-28) — Added `verify_position_protection()` to `DydxExecutor`: checks every open position has TP+SL orders, emergency-closes positions missing SL. Called from Stage 0 in `reasoning_agent.py` (live mode only, after orphan cleanup).

---

## 2. BACKTESTING & ML TRAINING

### 2.1 Backtest vs Live Signal Threshold Mismatch
**File:** `model_training/train_model_v23.py:335-351` vs `llm_agent/signal_generator.py:246` | **Severity:** HIGH

**Problem:** Backtest uses expanding percentile filtering:
```python
expanding_threshold = np.quantile(past_probs, 1.0 - top_pct)
if t["prob"] >= expanding_threshold:
    trades.append(t)
```
Live uses a fixed probability threshold:
```python
firing = final_prob >= prob_threshold  # e.g., 0.40
```
The expanding percentile adapts to the probability distribution within each test fold — in high-signal periods the threshold rises (fewer, higher-quality trades), in low-signal periods it falls (more trades). This inflates backtest Sharpe and reduces drawdown compared to the fixed threshold used in production. Models with `top_pct=0.10` are especially affected.

**Recommendation:** Either:
1. Use the same fixed `prob_threshold` in backtesting (set `top_pct=None`)
2. Or implement expanding percentile in the live `signal_generator.py`

**Impact:** Backtest results with expanding percentile do not accurately predict live performance. Must retrain and re-evaluate models after aligning thresholds.

**Status:** FIXED (2026-02-26) — Set `top_pct=None` for all 44 models in `ALL_MODELS`. Backtest now uses fixed `prob_threshold` only, matching live behavior. Models need retraining to get honest metrics.

---

### 2.2 Portfolio DD Breaker Doesn't Unwind Positions
**File:** `model_training/train_model_v23.py:381-403` | **Severity:** MEDIUM

**Problem:** When the portfolio drawdown breaker triggers in backtesting, it skips new trades but doesn't close existing ones. Equity tracking assumes positions just stop opening — no mark-to-market on open trades during the cooldown. The cooldown resets after N skipped trades and resumes without verifying that conditions improved.

**Recommendation:** Either close existing positions when DD breaker fires, or at minimum mark open positions to market during the cooldown period.

**Status:** FIXED (2026-02-28) — When DD breaker triggers, all still-active positions (trades where `idx + horizon > trigger_idx`) are force-closed by zeroing their `net_ret`. Cumulative equity and peak are recomputed from scratch after the adjustment.

---

### 2.3 Recent-Weighted Quality Scoring Creates Regime Bias
**File:** `model_training/train_model_v23.py` (~line 858) | **Severity:** LOW

**Problem:**
```python
weight = 2.0 if s <= 2 else 1.0  # Recent splits weighted 2x
```
The most recent 2 of 10 walk-forward splits get double weight in the quality scoring. If the recent market regime is favorable (strong trend, low volatility), scores are inflated and models that fit the recent regime are promoted over more robust ones.

**Recommendation:** Consider equal weighting across all splits, or explicitly test robustness across different market regimes.

**Status:** FIXED (2026-02-28) — Changed to equal weighting (`weight = 1.0`) for all walk-forward splits. Phase header updated to "equal-weighted".

---

### 2.4 Reported Backtest Performance Is Unrealistic
**Severity:** HIGH (requires retrain to validate)

**Problem:**
- v23: Sharpe 3.1+, 10/10 positive WF splits, worst DD -0.3%, annualized +461%
- v22: Sharpe 40.1, +524,139% annualized

These results were inflated by multiple issues that have since been fixed (Optuna test-set leakage, low fees, no slippage, insufficient purge) plus the still-unfixed expanding percentile threshold advantage (item 2.1).

**Action required:** Retrain models with all fixes applied and compare new Sharpe/DD/win-rate against originals. Realistic expectations for a well-built crypto momentum system: Sharpe 0.5-1.5, max DD 5-15%.

---

## 3. DATA PIPELINE

### 3.1 Sentiment Data Lag May Be Insufficient
**Files:** `features/sentiment.py`, `downloaders/sentiment_hist.py` | **Severity:** LOW

**Problem:** Fear & Greed index uses `lag_days=0`. Verify that the Alternative.me API publishes the index in real-time rather than as a daily aggregate. If it's published once daily (e.g., at midnight UTC), using `lag_days=0` on intraday data could leak ~24h of future information.

**Recommendation:** Verify Alternative.me publication timing. If daily, use `lag_days=1`.

**Status:** FIXED (2026-02-28) — Changed `lag_days=0` to `lag_days=1` for all three daily sources in `sentiment.py` (Fear & Greed, Google Trends, CoinGecko market data). Daily values now become available at D+1 00:00 UTC, preventing look-ahead bias.

---

### 3.2 618+ Features Create Overfitting Risk
**Severity:** LOW (design consideration)

**Problem:** ~618 features for a binary classification task with ~240K rows creates high dimensionality risk. The per-target top-100 feature selection helps but doesn't eliminate the risk of spurious correlations, especially with only ~27 months of data (Nov 2023 – present) covering predominantly bull market conditions.

**Recommendation:** Monitor out-of-sample decay after deployment. Consider feature importance analysis to reduce to a more parsimonious set. Consider collecting more data spanning different market regimes (bear market, ranging).

---

### 3.3 Downloader Pagination Minor Issues
**File:** `downloaders/base.py` | **Severity:** LOW

**Problem:**
- `base.py` pagination loop may have off-by-one at boundaries
- dYdX and OKX downloaders lack incremental download logic (always re-download from start)
- Some magic numbers in rate limiting and retry backoff

**Recommendation:** For live operation these are non-critical (market_context.py only fetches last 24h). For historical re-download efficiency, add incremental logic by reading the latest timestamp from existing CSV before requesting new data.

**Status:** PARTIALLY FIXED (2026-02-28) — Fixed pagination boundary issues: `_paginate_by_ms` now uses `<=` for end boundary to avoid missing data at exact end timestamp; `_paginate_backward_iso` stop condition changed to `<=` to avoid re-fetching boundary rows. Incremental download logic not added (low priority).

---

## 4. LLM AGENT

### 4.1 Signal Generator Feature Warmup Gap
**File:** `llm_agent/signal_generator.py:219-220` | **Severity:** MEDIUM

**Problem:** Live inference uses only the last ~24h of data from `market_context_data/`. Rolling windows of 288 candles need a full 24h warmup. On fresh startup, or if market_context.py fetches less than 24h of data, feature quality degrades for the early candles. The system accepts models even with up to 50% missing features.

**Recommendation:** Either:
1. Ensure `market_context.py` always fetches at least 288 candles (24h of 5-min data)
2. Or add a warmup check: if fewer than 288 candles available, log a warning and skip ML inference for that cycle

**Status:** FIXED (2026-02-26) — `_build_features()` now returns empty DataFrame (skips ML inference) if fewer than 350 candles available. Added NaN check on latest feature row: >30% NaN raises ValueError, caught per-model.

---

### 4.2 Grok JSON Parsing Relies on Fragile Fallbacks
**File:** `llm_agent/grok_client.py` | **Severity:** LOW

**Problem:** Grok's JSON output is parsed with 3 fallback strategies (direct parse → regex extract → manual parse). While robust in practice, malformed JSON from the LLM could silently produce partial decisions with missing fields.

**Recommendation:** Add explicit validation of all required fields (`direction`, `confidence`, `entry_price`, `take_profit`, `stop_loss`, `duration_minutes`, `position_size_pct`) after JSON parsing, rejecting decisions with missing critical fields before they reach the risk manager.

**Status:** FIXED (2026-02-26) — Added value-range validation in `_validate_decision()`: `entry_price > 0`, `position_size_pct` in `(0, 1]`, `duration_minutes > 0`, and LONG/SHORT price ordering checks (TP > entry > SL for LONG, SL > entry > TP for SHORT).

---

## 5. OPERATIONAL

### 5.1 Pipeline Has No Health Check / Heartbeat
**File:** `run_pipeline.py` | **Severity:** LOW

**Problem:** In `--loop` mode, the pipeline runs indefinitely but has no health check mechanism. If the pipeline silently stalls (e.g., hung HTTP request, deadlock), there's no alert.

**Recommendation:** Add a heartbeat file (e.g., write timestamp to `state_data/heartbeat.txt` each cycle) and monitor externally.

**Status:** FIXED (2026-02-28) — Added `_write_heartbeat()` to `reasoning_agent.py`: writes `state_data/heartbeat.json` (timestamp, stage, status) at the start, after each stage, and at completion. Uses atomic write (tmp + rename) for safe external reads.

---

### 5.2 Subprocess Timeout May Leave Orphan Processes
**File:** `run_pipeline.py` | **Severity:** LOW

**Problem:** `run_pipeline.py` launches subprocesses with 10-minute (data) and 5-minute (reasoning) timeouts via `subprocess.run(timeout=...)`. If a timeout fires, the subprocess is killed, but child processes spawned by it may not be cleaned up.

**Recommendation:** Use `subprocess.Popen` with process group management, or ensure each subprocess handles SIGTERM gracefully.

**Status:** N/A (2026-02-28) — `run_pipeline.py` does not exist in the codebase. No subprocess usage found. The pipeline uses direct function calls and `asyncio.run()`. This issue is moot.

---

## 6. SUMMARY TABLE

| # | Issue | Severity | Status | File(s) |
|---|-------|----------|--------|---------|
| 1.1 | No orphan order cleanup on startup | MEDIUM | **FIXED** | `dydx_executor.py`, `dydx_client.py` |
| 1.2 | No position close monitoring | MEDIUM | **FIXED** | `dydx_executor.py` |
| 2.1 | Backtest/live threshold mismatch | HIGH | **FIXED** | `train_model_v23.py` |
| 2.2 | DD breaker doesn't unwind positions | MEDIUM | **FIXED** | `train_model_v23.py` |
| 2.3 | Recent-weighted quality scoring bias | LOW | **FIXED** | `train_model_v23.py` |
| 2.4 | Reported performance unrealistic (retrain needed) | HIGH | OPEN | `train_model_v23.py` |
| 3.1 | Sentiment data lag may be insufficient | LOW | **FIXED** | `sentiment.py` |
| 3.2 | 618+ features overfitting risk | LOW | OPEN | design |
| 3.3 | Downloader pagination minor issues | LOW | **PARTIAL** | `base.py`, downloaders |
| 4.1 | Signal generator warmup gap | MEDIUM | **FIXED** | `signal_generator.py` |
| 4.2 | Grok JSON fragile fallback parsing | LOW | **FIXED** | `grok_client.py` |
| 5.1 | No pipeline health check | LOW | **FIXED** | `reasoning_agent.py` |
| 5.2 | Subprocess orphan processes | LOW | **N/A** | no subprocess usage |

---

## 7. PRIORITY — BEFORE LIVE TRADING

### Must Do
1. **Retrain all models** with corrected fees, slippage, purge, and Optuna fix — compare new metrics to originals
2. ~~**Align backtest/live thresholds** (item 2.1)~~ — **FIXED** (2026-02-26): set `top_pct=None` for all models
3. ~~**Add orphan order cleanup** (item 1.1)~~ — **FIXED** (2026-02-26): startup cleanup in live mode

### Should Do
4. ~~Validate sentiment data publication timing (item 3.1)~~ — **FIXED** (2026-02-28): `lag_days=1` for all daily sources
5. ~~Add feature warmup check in signal generator (item 4.1)~~ — **FIXED** (2026-02-26)
6. ~~Add position monitoring for TP/SL fill verification (item 1.2)~~ — **FIXED** (2026-02-28): `verify_position_protection()` in Stage 0
7. ~~Add decision field validation after Grok JSON parsing (item 4.2)~~ — **FIXED** (2026-02-26)

### Nice to Have
8. ~~Pipeline heartbeat monitoring (item 5.1)~~ — **FIXED** (2026-02-28): `state_data/heartbeat.json`
9. ~~Pagination boundary fixes (item 3.3)~~ — **FIXED** (2026-02-28): off-by-one fixes in `base.py`
10. Feature importance analysis and dimensionality reduction (item 3.2)
11. Test across bear market data when available (item 3.2)
