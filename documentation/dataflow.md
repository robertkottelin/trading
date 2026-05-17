# BTC Perpetual Futures Trading System

> ML ensemble + 18 conventional strategies + LLM reasoning → automated execution on dYdX v4 mainnet.

---

## Data Flow

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 run_pipeline.py  (every 300s, tiered downloads)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1 — Market Context Download  →  market_context_data/
  ┌────────────────────────────────────────────────────────────────────┐
  │  FAST (every cycle):  binance 5m klines, dYdX candles, spot price  │
  │  MEDIUM (every 6th):  + Binance/Bybit funding, Coinalyze daily     │
  │  SLOW  (every 72nd):  + Deribit DVOL, premium index, taker vol,    │
  │                         macro (FRED+yfinance), F&G, on-chain,       │
  │                         blockchain, DeFi, CFTC COT                  │
  │  14 sources total │ 300s interval │ tiered to minimize API load     │
  └────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2 — reasoning_agent.py  (7 stages)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ STAGE 0: Position Monitor + Orphan Cleanup ───────────────────────┐
  │  Connect to dYdX mainnet (live mode only)                           │
  │  • cleanup_orphan_orders(): cancel open orders with no position     │
  │  • verify_position_orders(): if position missing TP/SL →           │
  │      reprotect at 1% SL / 1.5% TP from entry, GTT 24h             │
  │  • trail_stops(): advance SL through 3 tiers as PnL grows          │
  │      Tier 1: PnL ≥ 2.0% → SL to breakeven (0%)                    │
  │      Tier 2: PnL ≥ 3.5% → lock 1.5% profit                        │
  │      Tier 3: PnL ≥ 5.0% → lock 2.5% profit                        │
  │  ► EARLY EXIT: active position found → skip Stages 1–7             │
  │    (TP/SL + trailing stop already manage it; saves Grok API cost)   │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 1: ML Signal Generation ────────────────────────────────────┐
  │  signal_generator.py loads model configs from:                      │
  │    models/v23/production_config_v23.json   → bullish models (15+)  │
  │    models/bearish/production_config_bearish.json → bearish models  │
  │                                                                      │
  │  Feature pipeline (290+ features across 13 groups):                 │
  │    TA core (dYdX 5m)     → 280 features (RSI,MACD,BB,ATR,CCI…)    │
  │    Binance TA (5m)        → 86 features (prefixed bnc_)            │
  │    Cross-exchange         → 19 features                             │
  │    Funding rates          → 18 features (Binance + Bybit z-scores) │
  │    Open Interest          →  7 features                             │
  │    Positioning (L/S)      →  6 features                             │
  │    Implied Vol (DVOL)     →  7 features                             │
  │    Macro (SPX,NDX,DXY…)  → 39 features                             │
  │    Sentiment (F&G)        → 12 features                             │
  │    On-chain               → 19 features                             │
  │    DeFi (TVL, stables)    →  8 features                             │
  │    Coinalyze              → 22 features                             │
  │    Liquidations           →  4 features                             │
  │                                                                      │
  │  Each model → {prob, threshold, signal, strength, quality_weight}  │
  │  weighted_score = Σ(qw×prob_bull) − Σ(qw×prob_bear) / Σ(qw)       │
  │  Output: bullish_count / bearish_count / neutral_count / score     │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 1.5: Conventional Strategy Signals ─────────────────────────┐
  │  StrategyEngine runs all 18 strategies on market_context_data/      │
  │                                                                      │
  │  #   Strategy                Signal Logic                  Source   │
  │  ─────────────────────────────────────────────────────────────────  │
  │  1   Funding Rate Reversion  z-score < -2.0 (shorts)      Binance  │
  │                              z-score > +2.0 (longs)        Bybit    │
  │  2   Volatility Regime       IV/RV > 1.4 fear overblown   Deribit  │
  │                              IV/RV < 0.7 complacency       DVOL     │
  │  3   Liquidation & Position  liq cascade + OI divergence  Coinalyze│
  │  4   Sentiment & Capital     F&G < 20 or > 80 + confirm   F&G/DeFi │
  │  5   Trend Following         EMA144/288/864/2016 + ADX>25  Binance  │
  │  6   Technical Momentum      RSI+MACD+Stoch+BB+Fisher+CCI  Binance  │
  │  7   Macro Risk Regime       7-factor cross-asset score    FRED/yf  │
  │  8   Basis Reversion         premium z-score ±2.5σ/56d    Binance  │
  │  9   Taker Flow Imbalance    (buyvol−sellvol) z ±1.5σ/4h  Binance  │
  │  10  Commodity Risk          Cu/Au z-score ±0.75 (20d)    yfinance │
  │  11  Funding Carry Momentum  cum_funding > 0.0003 (3d)    Binance  │
  │  12  Supertrend OBV          ST(7,2) + OBV + TBR>0.505   Binance  │
  │  13  EMA Trend Regime        4h EMA5/13 × daily EMA50/200 Binance  │
  │  14  MACD Signal Cross       4h MACD(12,26,9) cross × GC  Binance  │
  │  15  BB Breakout OBV         4h BB(15,1.8)+OBV+ATR trend  Binance  │
  │  16  Stochastic EMA Cross    4h Stoch(8,3,3) oversold+EMA Binance  │
  │  17  RSI Trend Momentum      4h RSI(9) 50-cross from extr Binance  │
  │  18  VWAP RSI Reversion      4h VWAP(48)+ATR±1.5 breakout Binance  │
  │                                                                      │
  │  Output: long_count / short_count / inactive_count + text_summary  │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 2: Market Context ───────────────────────────────────────────┐
  │  context_builder assembles live data snapshot (~1800 chars):        │
  │    BTC price (dYdX), 24h change, funding rate                       │
  │    Open interest, L/S ratio, liquidation data                       │
  │    Deribit DVOL (implied vol), options OI                           │
  │    Fear & Greed index, on-chain activity                            │
  │    DXY, macro regime summary                                        │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 3: Portfolio State (live dYdX query) ───────────────────────┐
  │    Equity, free collateral, margin used %                           │
  │    Open positions: side, size, entry_price, unrealized_pnl         │
  │    Recent fills (last 5)                                            │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 4: Resolve Pending Decisions ───────────────────────────────┐
  │    Mark previous decisions as TP_HIT / SL_HIT / EXPIRED            │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 5: Decision & Trade History ────────────────────────────────┐
  │    Last 20 trades: action, direction, entry, fill, PnL, fees       │
  │    Aggregate stats: win rate, total PnL, avg trade size            │
  │    Last 10 equity snapshots (curve)                                 │
  │    RISK_LEVEL derived from recent streak (NORMAL / ELEVATED / HIGH) │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 6: Grok LLM Decision ───────────────────────────────────────┐
  │  Model:   grok-4-1-fast-reasoning                                   │
  │  API:     https://api.x.ai/v1/responses                             │
  │  Tools:   web_search + x_search (real-time BTC news + sentiment)   │
  │  Retries: 2 attempts × 5s delay; 3 consecutive failures → NO_TRADE │
  │                                                                      │
  │  Full prompt assembled in order:                                    │
  │    1. CURRENT TIME (UTC)                                            │
  │    2. ML signals text (bullish/bearish/neutral counts, score,       │
  │         per-model detail with prob/threshold/strength)              │
  │    3. Strategy signals text (18 strategies, direction, confidence,  │
  │         explanation, key metrics, consensus counts)                 │
  │    4. Market context (~1800 chars of live data)                     │
  │    5. Portfolio state (equity, positions, fills)                    │
  │    6. Trade history (last 20 trades + aggregate stats + equity curve│
  │    7. Decision history (recent decisions + RISK_LEVEL + streak)    │
  │    8. Instruction: search web + X, output JSON                     │
  │                                                                      │
  │  System prompt: see documentation/llm_context.md                   │
  │                                                                      │
  │  Output JSON:                                                        │
  │    { direction, confidence, entry_price, take_profit, stop_loss,   │
  │      duration_minutes, position_size_usd, rationale }               │
  └────────────────────────────────────────────────────────────────────┘
                               │
  ┌─ STAGE 7: Trade Execution ─────────────────────────────────────────┐
  │  RiskManager — 9 sequential checks (all must pass):                │
  │    1. direction ∈ {LONG, SHORT}                                     │
  │    2. equity ≥ $50                                                  │
  │    3. confidence ≥ 0.62                                             │
  │    4. price ordering valid (LONG: TP>entry>SL, SHORT: SL>entry>TP) │
  │    5. R:R ≥ 1.5:1                                                   │
  │    6. open positions < 1 (max_open_positions)                       │
  │    7. free_collateral / equity ≥ 20%                                │
  │    8. $20 ≤ position_size_usd ≤ $600, BTC size ≤ 0.05             │
  │    9. daily PnL loss < 10% (circuit breaker)                        │
  │                                                                      │
  │  DydxExecutor (if all checks pass):                                 │
  │    • Entry: MARKET order, IOC, ±0.5% slippage, GTB +10 blocks      │
  │    • Poll fill: 10 × 2s (20s max)                                   │
  │    • TP: TAKE_PROFIT_MARKET, reduce-only, GTT 24h                  │
  │    • SL: STOP_MARKET, reduce-only, limit ±1%, GTT 24h              │
  │                                                                      │
  │  All results written to:                                            │
  │    state_data/trades.jsonl   (every order action)                   │
  │    state_data/portfolio.jsonl (equity snapshots)                    │
  │    state_data/decisions.jsonl (every Grok decision)                 │
  │    state_data/heartbeat.json  (cycle health)                        │
  └────────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
trading/
├── run_pipeline.py                  # Orchestrator — single entry point
├── config/settings.yaml             # All configuration
├── downloaders/                     # 14 data source downloaders
│   ├── market_context.py            #   Live refresh (last 24h)
│   ├── binance_hist.py              #   Spot + futures klines, funding, OI, L/S, taker vol
│   ├── dydx_hist.py                 #   dYdX v4 candles, funding, trades
│   ├── deribit_hist.py              #   DVOL, options OI, IV surface
│   ├── bybit_hist.py / okx_hist.py  #   Funding, OI
│   ├── coinalyze_hist.py            #   OI, liquidations, L/S, aggregated funding
│   ├── macro_hist.py                #   FRED (rates, credit) + yfinance (equities, FX, commodities)
│   ├── sentiment_hist.py            #   Fear & Greed, CoinGecko, Google Trends
│   ├── btc_network_hist.py          #   mempool.space (mempool, mining, lightning)
│   ├── blockchain_hist.py           #   Blockchain.com (on-chain: addresses, tx vol, hash rate)
│   ├── defi_hist.py                 #   DefiLlama (TVL, stablecoin supply)
│   ├── cftc_hist.py                 #   CFTC COT reports (institutional positioning)
│   ├── coinbase_premium_hist.py     #   Coinbase premium spread
│   └── hyperliquid_hist.py          #   Hyperliquid candles, OI
├── features/                        # Feature engineering (14 modules, 290+ features)
├── model_training/                  # LightGBM/CatBoost training pipelines
│   ├── v2_all_pipeline.py           #   Bullish models (v23)
│   └── bearish_pipeline.py          #   Bearish models
├── models/
│   ├── v23/                         # 15+ bullish models (up_12_*, up_24_*, up_48_*)
│   └── bearish/                     # Bearish models (bear_12_*, bear_24_*)
├── strategies/                      # 18 conventional strategies
│   ├── base.py                      #   BaseStrategy + StrategySignal (loads YAML params)
│   ├── engine.py                    #   StrategyEngine (runs all 13, reload_params())
│   └── *.py                         #   One file per strategy
├── retraining/                      # Automated retraining + tuning
│   ├── retrain_manager.py           #   Orchestrator: should_retrain, run_retrain_chain, deploy
│   ├── run_retrain_chain.sh         #   Shell chain: dataset → bullish → bearish → tuning
│   ├── train_v2_staging.py          #   Bullish shim → models/v23_staging/
│   ├── train_bearish_staging.py     #   Bearish shim → models/bearish_staging/
│   └── strategy_tuner.py           #   Optuna tuner → config/strategy_params_staging.yaml
├── llm_agent/
│   ├── reasoning_agent.py           #   7-stage pipeline orchestrator
│   ├── grok_client.py               #   xAI Grok API wrapper + system prompt
│   ├── signal_generator.py          #   ML inference (loads models, builds features)
│   ├── context_builder.py           #   Market context text assembly
│   ├── portfolio_reader.py          #   dYdX live portfolio query
│   ├── decision_manager.py          #   Decision persistence + status tracking
│   └── trade_history.py             #   Trade history + equity curve for prompt
├── execution/
│   ├── dydx_client.py               #   dYdX v4 async REST + chain client
│   ├── dydx_executor.py             #   Order placement, TP/SL, trailing stops, orphan cleanup
│   ├── risk_manager.py              #   9-check pre-trade validation
│   └── paper_executor.py            #   Paper trading mode
└── state_data/
    ├── trades.jsonl                 # Every order action (entry, TP, SL, orphan, reject)
    ├── decisions.jsonl              # Every Grok decision with rationale
    ├── portfolio.jsonl              # Equity snapshots
    └── heartbeat.json               # Pipeline health + cycle info
```

---

## ML Models

**Bullish ensemble** (`models/v23/`): 15+ LightGBM/CatBoost models, 3 horizon families:
- `up_12_*`: 1-hour prediction (most reliable, highest quality_weight)
- `up_24_*`: 2-hour prediction (second most reliable)
- `up_48_*`: 4-hour prediction

**Bearish ensemble** (`models/bearish/`): Dedicated downside models trained on bearish regimes:
- `bear_12_*`: 1-hour horizon
- `bear_24_*`: 2-hour horizon
- Trained April 2026: AUC 0.71/0.69, Sharpe 4.5/6.8 on walk-forward splits

**Inference**: Minimum 350 candles (24h × 5m). Each model outputs `prob`, `threshold`, `signal` (BULLISH/BEARISH/NEUTRAL), `strength` (NOT_FIRING/WEAK/MODERATE/STRONG).

**Weighted score**: `Σ(quality_weight × prob_bullish) − Σ(quality_weight × prob_bearish) / Σ(quality_weight)` → range [−1, +1].

> **Retraining**: Automated every 48h via `retraining/run_retrain_chain.sh`.
> Runs in background while pipeline keeps trading. New models deploy atomically after full chain passes.
> Manual trigger: `python run_pipeline.py --full` (blocks until complete, then starts loop).

---

## Conventional Strategies

**Parameter tuning**: Automated every 48h (same cycle as ML retraining) via `retraining/strategy_tuner.py`.
Optuna sweeps 30 trials per strategy, validates robustness across 4 periods. Best params written to
`config/strategy_params.yaml` and loaded by `BaseStrategy.__init__()` at engine init time. Class-level
constants serve as defaults when no YAML override exists.

**Adding a new strategy**:
1. Create `strategies/your_strategy.py` inheriting `BaseStrategy`
2. Implement `compute_signal(data)` and `compute_signal_series(data)`
3. Add to `strategies/engine.py` → `get_selected_strategies()`
4. Add name to `config/settings.yaml` → `strategies.selected`
5. Run `python -m strategies.backtest --strategy your_strategy` to verify

---

## Automated Retraining Cycle

Triggered every 48h during the loop, or immediately on `python run_pipeline.py --full`.
Runs as a background subprocess — pipeline continues trading with current models during retraining.

```
run_pipeline.py  (every 48h or --full)
      │
      └──► retraining/run_retrain_chain.sh  (background subprocess)
                │
                ├── 1. build_dataset.py (root)   → processed_data/btc_training_dataset.parquet
                │      ~10-30 min | 290+ features, ~240K rows
                │
                ├── 2. train_v2_staging.py        → models/v23_staging/
                │      ~5-6 hours | LightGBM+CatBoost | 40 Optuna trials | 10 walk-forward splits
                │
                ├── 3. train_bearish_staging.py   → models/bearish_staging/
                │      ~30-60 min | 2 bearish targets (bear_12, bear_24)
                │
                ├── 4. strategy_tuner.py          → config/strategy_params_staging.yaml
                │      ~1-2 hours | 30 Optuna trials × 18 strategies
                │      Anti-overfitting: 4-period validation, no period Sharpe < -0.2,
                │      min 8 trades/period, Sharpe spread < 4.0, min 20 trades/yr
                │
                └── 5. State → 'ready_to_deploy' (retrain_state.json)
                         │
                         └── check_and_deploy() called each pipeline cycle
                               Atomic rename: staging/ → live/ (same filesystem)
                               models/v23_staging/      → models/v23/
                               models/bearish_staging/  → models/bearish/
                               strategy_params_staging.yaml → strategy_params.yaml
                               strategy_engine.reload_params() → picks up new YAML
```

**State file**: `state_data/retrain_state.json` — tracks `status`, `last_deployed`, `pid`

**Log file**: `logs/retrain.log`

---

## Execution Parameters

| Parameter | Value |
|---|---|
| Market | BTC-USD perpetual (dYdX v4 mainnet) |
| Max open positions | 1 |
| Target leverage | 1–5× (conviction-scaled) |
| Position size | $20–$600 USD notional |
| Max BTC size | 0.05 BTC |
| Min equity | $50 (below this: halt) |
| Min free collateral | 20% of equity |
| Daily loss limit | 10% (circuit breaker) |
| Confidence threshold | 0.62 (NORMAL) / 0.70 (ELEVATED) / 0.75 (HIGH) |
| R:R minimum | 1.5:1 (target 1.65:1+) |
| Min SL width ≥60min trade | 1.5% |
| Min SL width 30–59min trade | 1.0% |
| Entry slippage tolerance | ±0.5% |
| SL limit slippage | ±1.0% |
| Fill poll | 10 attempts × 2s |
| TP/SL order expiry | 24h (GTT) |
| Trailing stop tiers | 2% → breakeven, 3.5% → +1.5%, 5% → +2.5% |

---

## Dashboard (Streamlit, local-only)

A separate Streamlit process at `http://127.0.0.1:8501` provides full visibility
and control over the live bot. **It does not modify pipeline behavior** beyond
the pause flag.

**Start:** `bash dashboard/run_dashboard.sh` (or `PORT=8502 bash dashboard/run_dashboard.sh`).

### Pages

1. **Overview** — bot status (heartbeat), equity/margin, latest decision, retrain state
2. **Bot Control** — start/stop/pause/resume/restart with full CLI flag picker
3. **Logs** — live tail of `logs/pipeline_*.log` with level + regex filters
4. **Pipeline** — 7-stage DAG with per-stage timing and next-run countdown
5. **Data Explorer** — browse `market_context_data/`, `processed_data/`, `raw_data/` (CSV / Parquet / JSONL) with auto-detected timestamp plotting
6. **ML Models** — `production_config_*.json` weights bar chart, firing history, live inference
7. **Strategies** — 18 conventional strategies' live direction + confidence + explanation
8. **Positions** — equity curve + drawdown, win-rate, P&L by direction, trades ledger
9. **Backtests** — trigger `strategies.backtest_v2` with live log tail
10. **Retrain** — `retrain_state.json` viewer + `retraining.retrain_manager.run_retrain_chain` trigger
11. **Config** — YAML editor for `config/*.yaml` with diff preview and timestamped backups

### Artifacts read by the dashboard

| Artifact | Source |
|---|---|
| Heartbeat | `state_data/heartbeat.json` (written by `run_pipeline.py:177`) |
| Trades ledger | `state_data/trades.jsonl` |
| Portfolio snapshots | `state_data/portfolio.jsonl` |
| Decisions | `state_data/decisions.jsonl`, `llm_agent/decision.json`, `llm_agent/decision_history.json` |
| Retrain state | `state_data/retrain_state.json` |
| Pipeline logs | `logs/pipeline_*.log`, `logs/pipeline_live.log` |
| ML weights | `models/v23/production_config_v23.json`, `models/bearish/production_config_bearish.json` |
| Market data | `market_context_data/*.csv`, `processed_data/*.parquet` |
| Configs | `config/*.yaml` |

### Artifacts written by the dashboard

| Artifact | Purpose |
|---|---|
| `state_data/.bot_pid.json` | tracks the bot process spawned via the UI |
| `state_data/.pause_flag.json` | pause signal honoured by `reasoning_agent.run()` |
| `config/.backups/<name>.<UTC>.bak` | timestamped backup of any YAML save |
| `logs/dashboard_bot_*.log` | stdout/stderr of bot launched from the UI |
| `logs/dashboard_backtest_*.log` | stdout of backtest jobs |
| `logs/dashboard_retrain_*.log` | stdout of retrain jobs |

### Pause-flag protocol

The dashboard's **Pause** button writes `state_data/.pause_flag.json`. The
reasoning agent checks this file **at the top of every cycle** (right after
Stage 0). If present, it logs `PAUSE FLAG active — skipping Stages 1-7` and
returns. Open positions remain protected because Stage 0 (orphan cleanup,
trailing stops, position monitoring) has already run. **Resume** deletes the
flag, restoring full pipeline operation on the next cycle.
