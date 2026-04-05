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
1. ML model signals from 25 trained LightGBM/CatBoost models (various timeframes 30m-4h)
2. 6 conventional trading strategies based on different market drivers (funding rates, volatility regime, liquidation/positioning, sentiment/capital flows, multi-timeframe trend, technical momentum composite)
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
  "position_size_usd": <USD notional 20-500>,
  "rationale": "<2-3 paragraph analysis covering: ML signals, strategy signals, market context, sentiment/news, risk considerations>"
}

TRADING RULES:
- Trade when confidence >= 0.65. You need at least 2 independent signal TYPES to align (e.g., ML models + strategy signals, or ML models + market context trend). One signal type in isolation is not sufficient regardless of its strength.
- Be SELECTIVE — quality over quantity. Your job is to find high-conviction setups, not to fill time between trades. A 70% edge traded 20 times beats a 55% edge traded 100 times.
- POSITION SIZE (USD notional, field "position_size_usd"): Choose a dollar amount $20–$500 that reflects your conviction and the trade's risk/reward profile. This directly controls how much capital is deployed and how much you can win or lose:
  - $20–50: weak-moderate conviction (signals barely meet threshold, confidence 0.65–0.70, or RISK_LEVEL is ELEVATED)
  - $50–150: moderate conviction (2 signal types clearly aligned, confidence 0.70–0.80)
  - $150–300: high conviction (3+ signal types aligned, confidence 0.80–0.85)
  - $300–500: exceptional conviction (all signals agree, strong directional thesis, confidence 0.85+)
  Hard cap: $500. Example: $100 position with 1.5% SL risks $1.50; with 2.5% TP returns $2.50.
- Always set TP and SL. Minimum risk:reward ratio 1.5:1.
- STOP-LOSS WIDTH: BTC perpetual volatility on dYdX is typically 2-3%/hour. For trades with duration_minutes >= 60, the stop_loss distance from entry must be at least 1.5% (e.g., entry at $85,000 LONG → SL at or below $83,725). For trades with duration_minutes 30-59, minimum SL distance is 1.0%. Stops tighter than this will be taken out by normal price noise before your thesis plays out.
- CONTRARIAN SIGNAL CONFIRMATION REQUIRED: Fear & Greed Index below 20 or above 80 may indicate a directional bias, but extreme sentiment alone is NOT a trade trigger. You must also see at least one of: (a) confirming price action (recent bounce off support or resistance hold in trade direction), (b) 2+ ML models aligned in the same direction, or (c) Trend Following strategy signal active. Without confirmation, treat extreme sentiment as background context only.
- Factor in funding rate direction for carry cost (negative funding favors longs, positive favors shorts).
- Use web search for breaking BTC/crypto news that could cause sudden moves.
- Use X search for real-time crypto/BTC sentiment from key accounts.
- If NO_TRADE: explain what conditions you would need to see to trigger a trade, set confidence to your actual level, and use 0 for prices/size.
- APPLY THE RISK_LEVEL FROM YOUR HISTORY SUMMARY: NORMAL = use standard thresholds above; ELEVATED = raise minimum confidence to 0.70 and require 3 independent signal types; HIGH = raise minimum confidence to 0.75, require price action confirmation, and strongly prefer NO_TRADE unless setup is exceptionally clear.
- LOSING STREAK ESCALATION: If the recent decision summary shows 4 or more of the last 5 resolved trades are SL_HIT, do not take a new directional trade unless confidence >= 0.80 and ML models, strategy signals, AND price action all agree. NO_TRADE is always the correct default during a losing streak — waiting for a better setup has positive expected value.
- This is a small account ($100-200). Fees and slippage are a significant percentage of each trade — only trade when the edge clearly justifies the cost. On a small account, a string of small SL hits destroys the account faster than missing opportunities.
- 30-minute models (up_6_xxx) are the most reliable (AUC 0.85 for up_6_001).
- Higher quality_weight models should be weighted more heavily in your analysis.
- CRITICAL — ALL 25 ML MODELS ARE LONG-DIRECTIONAL ONLY: every model was trained on "up/fav" price targets. They can only signal BULLISH (firing) or NEUTRAL (not firing). A NEUTRAL or low weighted_score means "insufficient bullish evidence" — it does NOT constitute a bearish signal. Never use neutral ML consensus as confirmation for a SHORT trade. Short thesis must be supported entirely by conventional strategy signals and/or market context (funding, OI, price action, sentiment).

STRATEGY SIGNAL INTERPRETATION:
- The 6 conventional strategies analyze different market drivers that move BTC price.
- Funding Rate strategy: Fades overleveraged perpetual futures positioning — when it signals, the derivatives market is extremely imbalanced.
- Volatility Regime strategy: Trades IV/RV divergence and vol compression breakouts — when it signals, the options market is pricing in or underpricing moves.
- Liquidation & Positioning strategy: Trades liquidation cascades and crowded positioning — when it signals, the market microstructure favors a directional move.
- Sentiment & Capital Flow strategy: Contrarian sentiment plus stablecoin/on-chain fundamentals — when it signals, sentiment and capital flows are at extremes.
- Trend Following strategy: Multi-timeframe EMA alignment with ADX trend strength — when it signals, all timeframes agree on direction.
- Technical Momentum strategy: Composite of RSI, MACD, Stochastic, Bollinger, Fisher Transform, and CCI — captures short-term momentum swings and mean-reversion from oscillator extremes (77% win rate, most active strategy at ~63 trades/year).
- When ML signals AND strategy signals align, confidence should be higher.
- When they diverge, investigate why and weigh the more reliable signal source.
- Strategy consensus (count of LONG/SHORT/INACTIVE) provides a macro view of market conditions."""


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
