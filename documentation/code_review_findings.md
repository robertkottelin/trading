# Code Review Findings — BTC Trading System

> **Round 1:** Two comprehensive code reviews performed on 2026-02-26 and 2026-02-28.
> **Round 2:** Third code review performed on 2026-02-28.
> **Round 3:** Fourth code review performed on 2026-02-28 (full codebase audit).
> Covers: execution safety, backtesting realism, data pipeline integrity, LLM agent logic.
> Fixed issues from Round 1 have been archived to the summary table below.

---

## 1. OPEN ISSUES FROM ROUND 1

### 1.1 Reported Backtest Performance Is Unrealistic
**Severity:** HIGH (requires retrain to validate)

**Problem:**
- v23: Sharpe 3.1+, 10/10 positive WF splits, worst DD -0.3%, annualized +461%
- v22: Sharpe 40.1, +524,139% annualized

These results were inflated by multiple issues that have since been fixed (Optuna test-set leakage, low fees, no slippage, insufficient purge) plus the now-fixed expanding percentile threshold advantage.

**Action required:** Retrain models with all fixes applied and compare new Sharpe/DD/win-rate against originals. Realistic expectations for a well-built crypto momentum system: Sharpe 0.5-1.5, max DD 5-15%.

---

### 1.2 618+ Features Create Overfitting Risk
**Severity:** LOW (design consideration)

**Problem:** ~618 features for a binary classification task with ~240K rows creates high dimensionality risk. The per-target top-100 feature selection helps but doesn't eliminate the risk of spurious correlations, especially with only ~27 months of data (Nov 2023 – present) covering predominantly bull market conditions.

**Recommendation:** Monitor out-of-sample decay after deployment. Consider feature importance analysis to reduce to a more parsimonious set. Consider collecting more data spanning different market regimes (bear market, ranging).

---

### 1.3 Downloader Pagination Minor Issues
**File:** `downloaders/base.py` | **Severity:** LOW

**Problem:**
- dYdX and OKX downloaders lack incremental download logic (always re-download from start)
- Some magic numbers in rate limiting and retry backoff

**Status:** PARTIALLY FIXED (2026-02-28) — Fixed pagination boundary issues: `_paginate_by_ms` now uses `<=` for end boundary; `_paginate_backward_iso` stop condition changed to `<=`. Incremental download logic not added (low priority for live operation — `market_context.py` only fetches last 24h).

---

## 2. ROUND 2 CODE REVIEW (2026-02-28) — ALL FIXED

### 2.1 Fill Confirmation Fallback Matches Wrong Fill
**File:** `execution/dydx_executor.py` — `_wait_for_fill()` | **Severity:** CRITICAL

**Problem:** `_wait_for_fill()` fell back to returning the most recent fill from the Indexer if no `clientId` match was found. This could misattribute a fill from a different order (e.g., a TP/SL fill, or a fill from a previous crash), leading to incorrect `fill_price`, wrong `fee_usd`, and proceeding with TP/SL placement for a potentially unfilled entry.

**Fix:** Removed the fallback. The method now only returns fills matched by `clientId`. If no match is found after all polling attempts, returns `None`. The caller already handles this as `UNVERIFIED` status and performs an on-chain position check before deciding whether to place TP/SL.

**Status:** FIXED (2026-02-28)

---

### 2.2 Daily Loss Circuit Breaker Fails Open on Corrupted State
**File:** `execution/risk_manager.py` — `_check_daily_loss()` | **Severity:** MEDIUM

**Problem:** When `portfolio.jsonl` existed but was corrupted or unreadable (`OSError` or `json.JSONDecodeError`), the daily loss circuit breaker returned `(True, "")` — allowing the trade to proceed. A financial safety check should fail-closed: if you cannot verify daily losses are within limits, reject the trade.

**Fix:** Changed the exception handler to return `(False, "daily loss check failed: could not read portfolio history")`. Note: the "file does not exist" case (line 171) still correctly returns `(True, "")` since there's no history to check on first run.

**Status:** FIXED (2026-02-28)

---

### 2.3 Emergency Close Not Verified
**File:** `execution/dydx_executor.py` — `_emergency_close()` | **Severity:** MEDIUM

**Problem:** `_emergency_close()` placed a market close order and logged it as "SUBMITTED" without verifying the fill. If the order was rejected by the exchange (e.g., insufficient margin, self-trade prevention), the position would remain open but the system would believe it was closed.

**Fix:** After submitting the close order, the method now calls `_wait_for_fill()` with the close order's `clientId`. If confirmed, logs at INFO with fill price and records status `FILLED`. If unconfirmed, logs at CRITICAL level ("position may still be open — manual intervention required") and records status `UNVERIFIED`.

**Status:** FIXED (2026-02-28)

---

### 2.4 No Market Data Staleness Check
**File:** `llm_agent/signal_generator.py` — `generate_signals()` | **Severity:** MEDIUM

**Problem:** The signal generator logged the latest candle timestamp but never checked how old it was. If the data pipeline failed to update (e.g., exchange API outage, downloader crash), ML signals would be generated on stale data without any warning.

**Fix:** Added staleness check after computing `latest_ts`:
- **>30 minutes stale:** Logs WARNING ("signals may be unreliable") but continues inference.
- **>2 hours stale:** Logs ERROR and returns empty signals (same format as warmup gap check), preventing trading on data that's too old.

**Status:** FIXED (2026-02-28)

---

### 2.5 Orphan Cleanup Misses Duplicate TP/SL Orders
**File:** `execution/dydx_executor.py` — `cleanup_orphan_orders()` | **Severity:** LOW-MEDIUM

**Problem:** `cleanup_orphan_orders()` only checked whether orders existed in markets without active positions. It would not detect duplicate TP or SL orders for the same position — e.g., if the system crashed after placing a TP order but before recording it, then placed another TP on restart.

**Fix:** Added a second pass after the orphan-by-market cleanup. For each active position's market, groups orders by type (TP vs SL). If more than 1 TP or more than 1 SL order exists, cancels the extras. Re-fetches open orders before the second pass if any were cancelled in the first pass to avoid operating on stale state.

**Status:** FIXED (2026-02-28)

---

### 2.6 Target Threshold Parsing Crash on Malformed Names
**File:** `llm_agent/signal_generator.py` — `_run_single_model()` | **Severity:** LOW

**Problem:** The threshold parsing code `thresh_str[0] + "." + thresh_str[1:]` would crash with `IndexError` if the target name had an unexpected format (empty or single-character threshold component).

**Fix:** Wrapped in `try/except (IndexError, ValueError)`. On parse failure, defaults to `threshold_decimal = 0.0` and logs a warning. This value is only used for the human-readable signal description — the actual model threshold comparison uses `prob_threshold` from config, so trading correctness is unaffected.

**Status:** FIXED (2026-02-28)

---

## 3. ROUND 3 CODE REVIEW (2026-02-28) — ALL FIXED

### 3.1 Grok JSON Extraction Uses `rfind("}"}` — Matches Wrong Brace
**File:** `llm_agent/grok_client.py` — `_parse_decision()` | **Severity:** HIGH

**Problem:** The fallback JSON extraction used `raw_text.rfind("}")` to find the closing brace. This finds the LAST `}` in the entire response, not the one matching the opening `{`. If Grok's response contained additional JSON-like structures after the decision (e.g., in web search results or rationale), the parser would extract a superset of the intended JSON, causing parse failure or extracting the wrong object.

**Fix:** Replaced `rfind` with a balanced brace-matching algorithm that tracks depth. The loop increments on `{` and decrements on `}`, finding the first `}` that balances the opening `{`. This correctly extracts the outermost JSON object.

**Status:** FIXED (2026-02-28)

---

### 3.2 Confidence, Entry Price, TP, SL Accept NaN/Infinity
**File:** `llm_agent/grok_client.py` — `_validate_decision()` | **Severity:** MEDIUM

**Problem:** The validation only checked `0 <= confidence <= 1` and `entry_price > 0`, but Python's comparison operators return `False` for NaN, so `0 <= NaN <= 1` is `False` — however `NaN > 0` is also `False`, meaning the entry_price check would catch NaN but the confidence check would coincidentally reject it. The real gap was: `float('inf')` would pass the confidence check (`0 <= inf <= 1` is `False`, but `inf > 0` is `True` for entry_price), and there was no explicit NaN/inf check for TP or SL prices.

**Fix:** Added explicit `math.isnan()` and `math.isinf()` checks for confidence, entry_price, take_profit, and stop_loss. Also added type checks (`isinstance(x, (int, float))`) to reject string or None values early.

**Status:** FIXED (2026-02-28)

---

### 3.3 Paper Executor Accepts Zero/NaN/Negative Prices
**File:** `execution/paper_executor.py` — `_fetch_current_price()` | **Severity:** MEDIUM

**Problem:** `_fetch_current_price()` returned `float(data["candles"][0]["close"])` without validating the value. If the Indexer returned a corrupted candle with close=0 or close=NaN, the paper trade would proceed with an invalid price, causing division by zero in position sizing (`size_btc = (equity * size_pct) / entry_price`).

**Fix:** Added price validation: rejects values that are `<= 0` or NaN (using the `x != x` NaN check). Raises `RuntimeError` with the invalid value for debugging.

**Status:** FIXED (2026-02-28)

---

### 3.4 Decision Manager Division by Zero on entry_price
**File:** `llm_agent/decision_manager.py` — `resolve_pending()` | **Severity:** MEDIUM

**Problem:** PnL calculations like `(sl - entry_price) / entry_price * 100` assumed `entry_price > 0`. The guard at line 94 used `if not all([entry_price, tp, sl, ts])`, but Python's `not all(...)` treats `0` as falsy — so an `entry_price` of 0 would be caught. However, a negative `entry_price` (from corrupted data) would pass and cause a negative-denominator division, producing inverted PnL signs.

**Fix:** Changed guard to `if not ts or not entry_price or entry_price <= 0 or not tp or not sl:` — explicitly rejects zero and negative entry prices before reaching any division.

**Status:** FIXED (2026-02-28)

---

### 3.5 Config Read Race Condition Between Stages
**File:** `llm_agent/reasoning_agent.py` — `run()` | **Severity:** MEDIUM

**Problem:** `config/settings.yaml` was read twice: once at Stage 0 (line 89) and again at Stage 7 (line 251). If the config file was modified between these reads (e.g., switching from `mode: paper` to `mode: live` while the pipeline is running), Stage 7 could execute in a different mode than Stage 0, potentially placing live trades when orphan cleanup ran in paper mode, or vice versa.

**Fix:** Config is now loaded once at Stage 0 using a proper `with open(...)` context manager. The `exec_cfg` and `mode` variables are reused at Stage 7. Added a comment explaining the single-load pattern.

**Status:** FIXED (2026-02-28)

---

### 3.6 Unbounded Forward-Fill Propagates Stale Data
**File:** `features/alignment.py` — `align_ffill()`, `align_daily()` | **Severity:** MEDIUM

**Problem:** `ffill()` was called without a `limit` parameter. If a data source had a multi-hour gap (e.g., exchange API outage, delayed data delivery), the last known value would propagate forward indefinitely — filling hundreds of 5-minute candles with a single stale value. This could introduce look-ahead bias during training (stale features correlated with future price movements) and degrade live signal quality.

**Fix:**
- `align_ffill()`: Added `ffill_limit` parameter (default 288 = 24 hours of 5-min candles). Values beyond this gap become NaN.
- `align_daily()`: Added `limit=576` (2 days) to the ffill call. Daily data legitimately fills 288 candles per day, so 576 allows for weekends/holidays while still catching multi-day gaps.

**Status:** FIXED (2026-02-28)

---

### 3.7 Timezone-Aware ISO Strings Parsed Incorrectly
**File:** `downloaders/base.py` — `_iso_to_ms()` | **Severity:** LOW-MEDIUM

**Problem:** `_iso_to_ms()` used `dt.replace(tzinfo=timezone.utc)` for naive datetime objects. However, if the input string was already timezone-aware (e.g., `"2024-01-01T12:00:00+05:00"`), `pd.Timestamp().to_pydatetime()` preserves the timezone, and the existing code would skip the `replace()` — so the non-UTC timezone would remain, and `dt.timestamp()` would correctly convert it to UTC epoch seconds. **But**: the original code used `replace(tzinfo=timezone.utc)` which would _overwrite_ an existing timezone instead of _converting_ it, producing wrong timestamps if a future code path passed timezone-aware strings.

**Fix:** Changed to use `pd.Timestamp` with `tz_localize("UTC")` for naive strings and `tz_convert("UTC")` for timezone-aware strings. This correctly handles both cases without the risk of timezone overwrite.

**Status:** FIXED (2026-02-28)

---

## 4. RESOLVED ISSUES FROM ROUND 1 (Archive)

| # | Issue | Severity | Fixed Date | Files |
|---|-------|----------|------------|-------|
| R1.1 | No orphan order cleanup on startup | MEDIUM | 2026-02-26 | `dydx_executor.py`, `dydx_client.py` |
| R1.2 | No position close monitoring | MEDIUM | 2026-02-28 | `dydx_executor.py` |
| R1.3 | Backtest/live threshold mismatch | HIGH | 2026-02-26 | `train_model_v23.py` |
| R1.4 | DD breaker doesn't unwind positions | MEDIUM | 2026-02-28 | `train_model_v23.py` |
| R1.5 | Recent-weighted quality scoring bias | LOW | 2026-02-28 | `train_model_v23.py` |
| R1.6 | Sentiment data lag (look-ahead bias) | LOW | 2026-02-28 | `sentiment.py` |
| R1.7 | Signal generator warmup gap | MEDIUM | 2026-02-26 | `signal_generator.py` |
| R1.8 | Grok JSON fragile fallback parsing | LOW | 2026-02-26 | `grok_client.py` |
| R1.9 | No pipeline health check / heartbeat | LOW | 2026-02-28 | `reasoning_agent.py` |
| R1.10 | Subprocess timeout orphan processes | LOW | N/A | moot (no subprocess usage) |

---

## 5. SUMMARY TABLE (All Issues)

| # | Issue | Severity | Status | File(s) |
|---|-------|----------|--------|---------|
| 1.1 | Reported performance unrealistic (retrain needed) | HIGH | **OPEN** | `train_model_v23.py` |
| 1.2 | 618+ features overfitting risk | LOW | **OPEN** | design |
| 1.3 | Downloader pagination minor issues | LOW | **PARTIAL** | `base.py`, downloaders |
| 2.1 | Fill confirmation fallback matches wrong fill | CRITICAL | **FIXED** | `dydx_executor.py` |
| 2.2 | Daily loss circuit breaker fails open | MEDIUM | **FIXED** | `risk_manager.py` |
| 2.3 | Emergency close not verified | MEDIUM | **FIXED** | `dydx_executor.py` |
| 2.4 | No market data staleness check | MEDIUM | **FIXED** | `signal_generator.py` |
| 2.5 | Orphan cleanup misses duplicate TP/SL | LOW-MEDIUM | **FIXED** | `dydx_executor.py` |
| 2.6 | Target threshold parsing crash | LOW | **FIXED** | `signal_generator.py` |
| 3.1 | JSON extraction matches wrong brace | HIGH | **FIXED** | `grok_client.py` |
| 3.2 | Confidence/prices accept NaN/infinity | MEDIUM | **FIXED** | `grok_client.py` |
| 3.3 | Paper executor accepts invalid prices | MEDIUM | **FIXED** | `paper_executor.py` |
| 3.4 | Decision manager division by zero | MEDIUM | **FIXED** | `decision_manager.py` |
| 3.5 | Config read race condition | MEDIUM | **FIXED** | `reasoning_agent.py` |
| 3.6 | Unbounded ffill propagates stale data | MEDIUM | **FIXED** | `alignment.py` |
| 3.7 | ISO timestamp timezone handling | LOW-MEDIUM | **FIXED** | `base.py` |

---

## 6. PRIORITY — BEFORE LIVE TRADING

### Must Do
1. **Retrain all models** with corrected fees, slippage, purge, and Optuna fix — compare new metrics to originals (item 1.1)

### Should Do
2. Feature importance analysis and dimensionality reduction (item 1.2)
3. Test on historical bear market data when available (item 1.2)

### Nice to Have
4. Add incremental download logic to dYdX/OKX downloaders (item 1.3)
