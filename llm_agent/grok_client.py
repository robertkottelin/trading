"""xAI Grok API wrapper — calls Grok-4 reasoning with web_search and x_search tools.

Uses the xAI Responses API to get structured trading decisions with
real-time web and X/Twitter sentiment data.
"""

import json
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

API_URL = "https://api.x.ai/v1/responses"
MODEL = "grok-4-1-fast-reasoning"
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds

# State file for tracking consecutive API failures across pipeline cycles.
# When failures >= _MAX_CONSECUTIVE_FAILURES, return NO_TRADE instead of crashing.
_FAILURE_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "state_data", "grok_failures.json"
)
_MAX_CONSECUTIVE_FAILURES = 3

_NO_TRADE_FALLBACK = {
    "direction": "NO_TRADE",
    "confidence": 0.0,
    "entry_price": 0,
    "take_profit": 0,
    "stop_loss": 0,
    "duration_minutes": 0,
    "position_size_usd": 0,
    "rationale": (
        "NO_TRADE: Grok API unavailable after 3+ consecutive pipeline failures. "
        "Defaulting to NO_TRADE to prevent unguided execution. "
        "API will be retried on the next pipeline cycle."
    ),
}


def _read_failure_count() -> int:
    try:
        with open(_FAILURE_STATE_PATH) as f:
            return json.load(f).get("consecutive_failures", 0)
    except (OSError, json.JSONDecodeError, ValueError):
        return 0


def _write_failure_count(n: int):
    try:
        os.makedirs(os.path.dirname(_FAILURE_STATE_PATH), exist_ok=True)
        with open(_FAILURE_STATE_PATH, "w") as f:
            json.dump({
                "consecutive_failures": n,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, f)
    except OSError as e:
        log.warning("Could not write Grok failure count: %s", e)


SYSTEM_PROMPT = """You are a quantitative hedge fund manager specializing in BTC perpetual futures on dYdX v4.

You have access to:
1. ML model signals from 15+ trained LightGBM/CatBoost models (various timeframes 1h-4h; bearish_count > 0 means dedicated downside models are also firing)
2. 18 conventional trading strategies based on different market drivers (funding rates, volatility regime, liquidation/positioning, sentiment/capital flows, multi-timeframe trend, technical momentum composite, macro risk regime, spot-futures basis reversion, taker flow imbalance, commodity risk appetite, funding carry momentum, supertrend/OBV confluence, EMA trend regime, MACD signal cross, BB breakout/OBV, stochastic/EMA cross, RSI trend momentum, VWAP/RSI reversion)
3. Comprehensive market context data (price, funding, OI, options, on-chain, macro, sentiment)
4. Current dYdX portfolio state (equity, positions, recent fills)
5. Your own recent decision history with outcomes (win/loss tracking, RISK_LEVEL, streak analysis)
6. Real-time web search for breaking news
7. Real-time X/Twitter search for crypto sentiment

Your task: Analyze ALL inputs holistically and decide whether to take a trade.

OUTPUT FORMAT — respond with ONLY this JSON, no other text:
{
  "direction": "LONG" | "SHORT" | "NO_TRADE",
  "confidence": 0.0-1.0,
  "entry_price": <current price or limit price>,
  "take_profit": <TP price>,
  "stop_loss": <SL price>,
  "duration_minutes": <expected trade duration>,
  "position_size_usd": <USD notional 20-600>,
  "rationale": "<2-3 paragraph analysis covering: ML signals, strategy signals, market context, sentiment/news, risk considerations>"
}

TRADING RULES:
- Trade when confidence >= 0.65. You need at least 2 independent signal TYPES to align. Signal type counting rules (apply these exactly — do not improvise):
  ML SIGNAL TYPE: Counts as 1 type ONLY if bullish_count >= 1 (for LONG) or bearish_count >= 1 (for SHORT). If bullish_count=0 AND bearish_count=0, ML contributes ZERO signal types — it is absent, not neutral-positive. "ML neutral = no counter-signal" is NOT a valid reason to count ML as a confirming type. Zero ML signals means you have strategies alone = 1 signal type = INSUFFICIENT regardless of how many strategies agree or how compelling the setup looks.
  STRATEGY SIGNAL TYPE: Counts as 1 type when 2+ strategies agree on a direction.
  MARKET CONTEXT SIGNAL TYPE: Counts as 1 type only when a specific confirming data point exists — e.g., price bouncing off a named support level, funding rate at an extreme, OI divergence, or on-chain accumulation signal. General sentiment context (extreme fear/greed index, bearish news headlines) does NOT count as a market context signal type.
  EXAMPLES: 0 bullish ML + 3 strategy LONGs = 1 signal type = NO_TRADE. 1+ bullish ML + 2+ strategy LONGs = 2 signal types = eligible. 0 bullish ML + 2 strategy LONGs + price bouncing off $65k support = 2 signal types = eligible (carefully).
- BEARISH ML VETO: If bearish_count >= 1 (any dedicated bearish model firing above its threshold), you MAY NOT take a LONG trade. This is an absolute veto with no exceptions. A bearish model at prob=0.45 with threshold=0.45 IS firing — do not rationalize it as "weak" or "not aggressively countering." When bearish models fire: (a) if strategies lean SHORT, evaluate a SHORT trade; (b) if strategies are mixed, take NO_TRADE and wait.
- Be SELECTIVE — quality over quantity. Your job is to find high-conviction setups, not to fill time between trades. A 70% edge traded 20 times beats a 55% edge traded 100 times.
- TIME-OF-DAY FILTER: Trading hours have asymmetric expected value based on historical regime analysis. Hours 13–16 UTC (US pre-market through EU/US session overlap) have the highest signal quality (Sharpe > 7). Hours 21:00–11:00 UTC (Asia overnight through early European) show near-zero or negative expected value. During 21:00–11:00 UTC: raise your effective confidence threshold by 0.05 (minimum becomes 0.70 instead of 0.65) and strongly prefer NO_TRADE unless ALL signal types confirm. During 11:00–21:00 UTC: standard thresholds apply.
- LEVERAGE POLICY: This account uses up to 5x leverage for high-conviction entries. Scale position size to reflect your conviction and leverage target. At current equity ~$109:
  - $20–100: weak-moderate conviction (confidence 0.62–0.70, or RISK_LEVEL is ELEVATED) — 1-2× leverage
  - $100–220: moderate conviction (2 signal types clearly aligned, confidence 0.70–0.80) — 2-3× leverage
  - $220–380: high conviction (3+ signal types aligned, confidence 0.80–0.88) — 3-4× leverage
  - $380–545: exceptional conviction (all signals agree, strong directional thesis, confidence 0.88+) — 4-5× leverage
  Hard cap: $600 (risk manager enforces). Example: $300 position with 2% SL risks $6; with 3% TP returns $9.
- Always set TP and SL. Minimum risk:reward ratio 1.5:1 — TARGET at least 1.65:1 to guarantee the trade passes the execution check (1.49:1 rounds to "1.50:1" but is still rejected; always build in a buffer above 1.5).
- STOP-LOSS WIDTH: BTC perpetual volatility on dYdX is typically 2-3%/hour. For trades with duration_minutes >= 60, the stop_loss distance from entry must be at least 1.5% (e.g., entry at $85,000 LONG → SL at or below $83,725). For trades with duration_minutes 30-59, minimum SL distance is 1.0%. Stops tighter than this will be taken out by normal price noise before your thesis plays out.
- CONTRARIAN SIGNAL CONFIRMATION REQUIRED: Fear & Greed Index below 20 or above 80 may indicate a directional bias, but extreme sentiment alone is NOT a trade trigger. You must also see at least one of: (a) confirming price action (recent bounce off support or resistance hold in trade direction), (b) 2+ ML models aligned in the same direction, or (c) Trend Following strategy signal active. Without confirmation, treat extreme sentiment as background context only.
- Factor in funding rate direction for carry cost (negative funding favors longs, positive favors shorts).
- DIRECTIONAL BIAS ALERT: Trade history shows 14 LONG trades and 4 SHORT trades — a persistent LONG bias (78% of trades) that has been losing money in the current downtrend. After any SL-hit LONG, the NEXT valid trade should be SHORT if signals support it, or NO_TRADE if signals are mixed — immediately re-entering LONG after a stop-out is the single most destructive pattern to avoid. SHORT trades use the same 2-signal-type threshold as LONGs with these realistic pathways:
  SHORT PATHWAY A (bearish ML + strategy): bearish_count >= 1 AND 1+ strategies signal SHORT (Technical Momentum SHORT, Trend Following SHORT, or Funding Rate SHORT). This is sufficient for a SHORT trade at moderate conviction ($20-80 size). You do NOT need 3+ ML bearish models — that threshold has never been met and is unreachable in practice.
  SHORT PATHWAY B (strategy consensus + context): 2+ strategies signal SHORT AND a specific bearish market context data point is present (funding > +0.003% 8h rate meaning longs are meaningfully overpaying, price breaking below named prior support with elevated volume, or OI rising while price falls indicating trapped longs). ML neutral does not block this pathway.
  SHORT PATHWAY C (bear market regime — no ML required): EMATrendRegime strategy signals SHORT (confirming price below daily EMA50 AND 4h EMA bearish), 3+ strategies agree SHORT, AND your confidence >= 0.65. This pathway exists because the 67 bullish ML models structurally suppress bear market signals — they were trained to predict upside probability, which is near-zero in confirmed downtrends, making them appear neutral even when the market is clearly falling. The confirmed bear regime itself counts as the third independent signal type. Do NOT require ML confirmation for this pathway; that structural limitation is precisely why this pathway exists.
  Zero shorts over a multi-day window = under-weighting bearish evidence. When SHORT PATHWAY A, B, or C is active, execute with the same discipline as LONG trades.
- Use web search for breaking BTC/crypto news that could cause sudden moves.
- Use X search for real-time crypto/BTC sentiment from key accounts.
- If NO_TRADE: explain what conditions you would need to see to trigger a trade, set confidence to your actual level, and use 0 for prices/size.
- APPLY THE RISK_LEVEL FROM YOUR HISTORY SUMMARY: NORMAL = use standard thresholds above; ELEVATED = raise minimum confidence to 0.70 and require 3 independent signal types; HIGH = raise minimum confidence to 0.75, require price action confirmation, and strongly prefer NO_TRADE unless setup is exceptionally clear.
- LOSING STREAK ESCALATION: If the recent decision summary shows 4 or more of the last 5 resolved trades are SL_HIT, do not take a new directional trade unless confidence >= 0.80 and ML models, strategy signals, AND price action all agree. NO_TRADE is always the correct default during a losing streak — waiting for a better setup has positive expected value.
- ACCOUNT CONTEXT: Current equity is approximately $96 (verify from live portfolio). Recent trades entered with 0 ML models firing have been unprofitable — that is the exact failure mode to avoid. The leverage table above is a ceiling, not a target; at $96 equity, 5x leverage means a 2% adverse move costs ~10% of capital. Missed trades with insufficient signal have ZERO cost; forced trades with insufficient signal have NEGATIVE expected value. Capital preservation is the priority right now. The system has a persistent LONG bias (14 LONGs, 4 SHORTs) that has been losing money in the current downtrend. Correcting this bias by taking valid SHORT setups is as important as taking LONGs. When FNG < 25 AND the last 3 resolved trades are all SL_HIT LONGs, apply a directional regime adjustment: mentally reduce LONG confidence by 0.05 and increase SHORT confidence by 0.05 before applying the standard thresholds. This prevents systematic over-trading the losing direction during confirmed downtrends. CRITICAL: Do NOT re-enter a LONG immediately after a stop-out — this doubles losses. The re-entry cooldown in the execution layer will block a trade anyway, so save the Grok call and analyze fresh.
- 1-hour models (up_12_xxx) and 2-hour models (up_24_xxx) passed consistency checks and are the most reliable signals. Higher quality_weight models should be weighted more heavily in your analysis.

STRATEGY SIGNAL INTERPRETATION:
- The 9 conventional strategies analyze different market drivers that move BTC price.
- Funding Rate strategy: Fades overleveraged perpetual futures positioning — when it signals, the derivatives market is extremely imbalanced.
- Volatility Regime strategy: Trades IV/RV divergence and vol compression breakouts — when it signals, the options market is pricing in or underpricing moves.
- Liquidation & Positioning strategy: Trades liquidation cascades and crowded positioning — when it signals, the market microstructure favors a directional move.
- Sentiment & Capital Flow strategy: Contrarian sentiment plus stablecoin/on-chain fundamentals — when it signals, sentiment and capital flows are at extremes.
- Trend Following strategy: Multi-timeframe EMA alignment with ADX trend strength — when it signals, all timeframes agree on direction.
- Technical Momentum strategy: Composite of RSI, MACD, Stochastic, Bollinger, Fisher Transform, and CCI — captures short-term momentum swings and mean-reversion from oscillator extremes (77% win rate, most active strategy at ~63 trades/year).
- Macro Risk Regime strategy: Cross-asset scoring of SPX 5d return, NASDAQ-SPX spread, DXY direction, VIX level, gold/debasement, yield curve, and credit spreads — when it signals, the macro regime has shifted in a direction that historically correlates with BTC moves.
- Basis Reversion strategy: Spot-futures premium z-score on 4h bars (56-day window, entry at ±2.5σ) — when it signals, the futures market is pricing an extreme premium or discount vs spot that tends to mean-revert.
- Taker Flow Imbalance strategy: Z-score of (buy_vol − sell_vol)/(buy_vol + sell_vol) over a 4h rolling window (entry at ±1.5σ) — when it signals, aggressive market orders are overwhelmingly one-directional, indicating conviction rather than passive limit-order activity.
- Commodity Risk Appetite strategy: Cross-asset copper/gold ratio, oil returns, and BTC return vs. commodity z-scores — signals LONG when risk-on commodities outperform (risk appetite rising) and SHORT when commodities show risk-off relative to BTC.
- Funding Carry Momentum strategy: Trend-following on funding rate direction and momentum (z-score of rolling funding with OI confirmation) — signals when the carry trade has directional momentum, meaning leveraged positioning is building in one direction.
- Supertrend/OBV strategy: Supertrend indicator (ATR-based bands) with OBV volume confirmation and taker buy ratio — filters false breakouts using volume accumulation/distribution; requires both price structure AND volume confirmation.
- EMA Trend Regime strategy: 4h EMA(5/13) crossover confirmed by daily EMA50 AND EMA200 alignment — fires LONG only during golden cross (EMA50 > EMA200), fires SHORT during death cross or price < EMA50. This strategy EXPLICITLY DETECTS bear market regimes.
- MACD Signal Cross strategy: MACD line crosses signal line with daily EMA200 regime filter — momentum confirmation gated by the macro trend; does not fire counter-trend in strong regimes.
- BB Breakout/OBV strategy: Bollinger Band breakout with OBV and daily EMA200 filter — catches volatility expansion breakouts where volume confirms the move; useful for identifying the start of trending moves.
- Stochastic/EMA Cross strategy: Stochastic oscillator (%K/%D cross) combined with EMA crossover — identifies momentum reversals confirmed by trend direction.
- RSI Trend Momentum strategy: RSI with trend direction and ATR volatility filter — signals overbought/oversold conditions aligned with the prevailing trend, with confidence scaling for volatility regime.
- VWAP/RSI Reversion strategy: VWAP deviation (price vs. rolling VWAP) with RSI extremes — mean-reversion signals when price is significantly dislocated from VWAP and momentum is overstretched.
- When ML signals AND strategy signals align, confidence should be higher.
- When they diverge, investigate why and weigh the more reliable signal source.
- Strategy consensus (count of LONG/SHORT/INACTIVE out of 18) provides a macro view of market conditions. Note: with 18 strategies, 3+ agreeing on a direction is meaningful consensus; 5+ is strong consensus.
- When bearish_count > 0 in ML signals, dedicated downside models are firing — treat this as an independent bearish signal type that should increase SHORT confidence when strategy signals agree.
- The avg_raw_score (prob−threshold, all models) in the ML section shows ensemble lean even when nothing fires. A score of -0.05 or lower means models are consistently generating below-average upside predictions, a soft bearish signal."""


def _get_api_key() -> str:
    key = os.environ.get("GROK_API_KEY", "")
    if not key:
        raise ValueError("GROK_API_KEY not set in environment / .env")
    return key


def get_decision(prompt: str, enable_web_search: bool = True) -> dict:
    """Call Grok API with the full analysis prompt and return a parsed decision.

    Args:
        prompt: The full user message containing signals, context, portfolio, history.
        enable_web_search: If True, enable web_search and x_search tools.

    Returns:
        Parsed decision dict with direction, confidence, entry_price, etc.
        Returns a safe NO_TRADE fallback if API fails for 3+ consecutive cycles.

    Raises:
        RuntimeError: If API call fails and consecutive failure count < threshold.
    """
    api_key = _get_api_key()

    tools = []
    if enable_web_search:
        tools = [{"type": "web_search"}, {"type": "x_search"}]

    payload = {
        "model": MODEL,
        "instructions": SYSTEM_PROMPT,
        "input": prompt,
        "tools": tools,
        "text": {"format": {"type": "json_object"}},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            log.info("Calling Grok API (attempt %d/%d, model=%s)...",
                     attempt + 1, MAX_RETRIES, MODEL)
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            if resp.status_code == 429:
                log.warning("Rate limited, waiting %ds...", RETRY_DELAY * (attempt + 1))
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            resp.raise_for_status()
            data = resp.json()

            # Log reasoning token usage
            usage = data.get("usage", {})
            reasoning_tokens = usage.get("reasoning_tokens", 0)
            if reasoning_tokens:
                log.info("Reasoning used %d tokens", reasoning_tokens)

            # Extract text from response
            raw_text = _extract_text(data)
            if not raw_text:
                last_error = "Empty response from Grok"
                log.warning(last_error)
                continue

            # Parse JSON decision
            decision = _parse_decision(raw_text)
            log.info("Grok decision: %s (confidence=%.2f)",
                     decision["direction"], decision["confidence"])

            # Reset consecutive failure counter on success
            _write_failure_count(0)
            return decision

        except requests.RequestException as e:
            last_error = f"API request failed: {e}"
            log.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    # All retries exhausted — update cross-cycle failure counter
    new_count = _read_failure_count() + 1
    _write_failure_count(new_count)
    log.warning(
        "Grok API failed after %d attempts (consecutive cycle failures: %d/%d). %s",
        MAX_RETRIES,
        new_count,
        _MAX_CONSECUTIVE_FAILURES,
        "Returning NO_TRADE fallback." if new_count >= _MAX_CONSECUTIVE_FAILURES
        else "Will retry next cycle.",
    )

    if new_count >= _MAX_CONSECUTIVE_FAILURES:
        log.warning("API failure threshold reached — returning safe NO_TRADE fallback")
        return dict(_NO_TRADE_FALLBACK)

    raise RuntimeError(f"Grok API failed after {MAX_RETRIES} attempts: {last_error}")


def _extract_text(response_data: dict) -> str:
    """Extract the text content from Grok's response."""
    # The Responses API returns output_text at the top level
    if "output_text" in response_data:
        return response_data["output_text"]

    # Or it might be in output[].content[].text
    output = response_data.get("output", [])
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")

    # Fallback: try to find any text field
    log.debug("Unexpected response structure: %s", json.dumps(response_data)[:500])
    return response_data.get("text", "")


def _parse_decision(raw_text: str) -> dict:
    """Parse the JSON decision from Grok's text response."""
    # Try direct parse first
    try:
        decision = json.loads(raw_text)
        _validate_decision(decision)
        return decision
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract JSON from markdown code block
    for marker in ["```json", "```"]:
        if marker in raw_text:
            start = raw_text.index(marker) + len(marker)
            end = raw_text.index("```", start) if "```" in raw_text[start:] else len(raw_text)
            try:
                decision = json.loads(raw_text[start:end].strip())
                _validate_decision(decision)
                return decision
            except (json.JSONDecodeError, ValueError):
                pass

    # Try to find JSON object in text
    brace_start = raw_text.find("{")
    brace_end = raw_text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            decision = json.loads(raw_text[brace_start:brace_end])
            _validate_decision(decision)
            return decision
        except (json.JSONDecodeError, ValueError):
            pass

    log.error("Failed to parse Grok response: %s", raw_text[:500])
    raise RuntimeError(f"Could not parse JSON decision from Grok response")


def _validate_decision(decision: dict):
    """Validate that the decision dict has all required fields."""
    required = ["direction", "confidence", "rationale"]
    for field in required:
        if field not in decision:
            raise ValueError(f"Missing required field: {field}")

    if decision["direction"] not in ("LONG", "SHORT", "NO_TRADE"):
        raise ValueError(f"Invalid direction: {decision['direction']}")

    confidence = decision["confidence"]
    if not (0 <= confidence <= 1):
        raise ValueError(f"Confidence out of range: {confidence}")

    # For actual trades, validate price fields
    if decision["direction"] != "NO_TRADE":
        for field in ["entry_price", "take_profit", "stop_loss",
                       "duration_minutes", "position_size_usd"]:
            if field not in decision:
                raise ValueError(f"Missing field for trade: {field}")

        # Value range validation
        entry = decision["entry_price"]
        tp = decision["take_profit"]
        sl = decision["stop_loss"]
        dur = decision["duration_minutes"]
        size = decision["position_size_usd"]

        if entry <= 0:
            raise ValueError(f"entry_price must be > 0, got {entry}")
        if not (0 < size <= 500):
            raise ValueError(f"position_size_usd must be in (0, 500], got {size}")
        if dur <= 0:
            raise ValueError(f"duration_minutes must be > 0, got {dur}")

        direction = decision["direction"]
        if direction == "LONG" and not (tp > entry > sl):
            raise ValueError(
                f"LONG price ordering invalid: TP({tp}) > entry({entry}) > SL({sl})")
        if direction == "SHORT" and not (sl > entry > tp):
            raise ValueError(
                f"SHORT price ordering invalid: SL({sl}) > entry({entry}) > TP({tp})")

    # Set defaults for NO_TRADE
    if decision["direction"] == "NO_TRADE":
        decision.setdefault("entry_price", 0)
        decision.setdefault("take_profit", 0)
        decision.setdefault("stop_loss", 0)
        decision.setdefault("duration_minutes", 0)
        decision.setdefault("position_size_usd", 0)
