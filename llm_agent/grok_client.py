"""xAI Grok API wrapper — calls Grok-3 with web_search and x_search tools.

Uses the xAI Responses API to get structured trading decisions with
real-time web and X/Twitter sentiment data.
"""

import json
import logging
import math
import os
import time

import requests

log = logging.getLogger(__name__)

API_URL = "https://api.x.ai/v1/responses"
MODEL = "grok-4-fast-non-reasoning"
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds

SYSTEM_PROMPT = """You are a quantitative hedge fund manager specializing in BTC perpetual futures on dYdX v4.

You have access to:
1. ML model signals from 25 trained LightGBM/CatBoost models (various timeframes 30m-4h)
2. Comprehensive market context data (price, funding, OI, options, on-chain, macro, sentiment)
3. Current dYdX portfolio state (equity, positions, recent fills)
4. Your own recent decision history with outcomes (win/loss tracking)
5. Real-time web search for breaking news
6. Real-time X/Twitter search for crypto sentiment

Your task: Analyze ALL inputs holistically and decide whether to take a trade.

OUTPUT FORMAT — respond with ONLY this JSON, no other text:
{
  "direction": "LONG" | "SHORT" | "NO_TRADE",
  "confidence": 0.0-1.0,
  "entry_price": <current price or limit price>,
  "take_profit": <TP price>,
  "stop_loss": <SL price>,
  "duration_minutes": <expected trade duration>,
  "position_size_pct": 0.0-1.0,
  "rationale": "<2-3 paragraph analysis covering: ML signals interpretation, market context assessment, sentiment/news factors, risk considerations>"
}

TRADING RULES:
- Only trade when confidence > 0.6 AND multiple signal types align (ML + context + sentiment)
- Position size scales with confidence: 0.05 at 0.6, 0.10 at 0.7, 0.15 at 0.8, 0.25 at 0.9+
- Always set TP and SL. Minimum risk:reward ratio 1.5:1
- Consider current portfolio exposure — don't add to existing positions in same direction unless very high conviction
- Factor in funding rate direction for carry cost (negative funding favors longs, positive favors shorts)
- Use web search for breaking BTC/crypto news that could cause sudden moves
- Use X search for real-time crypto/BTC sentiment from key accounts
- If NO_TRADE: still explain what conditions you'd need to see, set confidence to your actual confidence level, and use 0 for prices/size
- Learn from recent decision outcomes — if recent SLs hit frequently, be more conservative
- 30-minute models (up_6_xxx) are the most reliable (AUC 0.85 for up_6_001)
- Higher quality_weight models should be weighted more heavily in your analysis
- Extreme Fear & Greed readings are contrarian signals (extreme fear = potential long opportunity)"""


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

    Raises:
        RuntimeError: If API call fails after retries or returns unparseable response.
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
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if resp.status_code == 429:
                log.warning("Rate limited, waiting %ds...", RETRY_DELAY * (attempt + 1))
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            resp.raise_for_status()
            data = resp.json()

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
            return decision

        except requests.RequestException as e:
            last_error = f"API request failed: {e}"
            log.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

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

    # Try to find JSON object in text using balanced brace matching
    brace_start = raw_text.find("{")
    if brace_start >= 0:
        depth = 0
        brace_end = -1
        for i in range(brace_start, len(raw_text)):
            if raw_text[i] == "{":
                depth += 1
            elif raw_text[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break
        if brace_end > brace_start:
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
    if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
        raise ValueError(f"Confidence must be a finite number, got: {confidence}")
    if not (0 <= confidence <= 1):
        raise ValueError(f"Confidence out of range: {confidence}")

    # For actual trades, validate price fields
    if decision["direction"] != "NO_TRADE":
        for field in ["entry_price", "take_profit", "stop_loss",
                       "duration_minutes", "position_size_pct"]:
            if field not in decision:
                raise ValueError(f"Missing field for trade: {field}")

        # Value range validation
        entry = decision["entry_price"]
        tp = decision["take_profit"]
        sl = decision["stop_loss"]
        dur = decision["duration_minutes"]
        size = decision["position_size_pct"]

        if not isinstance(entry, (int, float)) or math.isnan(entry) or math.isinf(entry) or entry <= 0:
            raise ValueError(f"entry_price must be a positive finite number, got {entry}")
        if not isinstance(tp, (int, float)) or math.isnan(tp) or math.isinf(tp) or tp <= 0:
            raise ValueError(f"take_profit must be a positive finite number, got {tp}")
        if not isinstance(sl, (int, float)) or math.isnan(sl) or math.isinf(sl) or sl <= 0:
            raise ValueError(f"stop_loss must be a positive finite number, got {sl}")
        if not (0 < size <= 1):
            raise ValueError(f"position_size_pct must be in (0, 1], got {size}")
        if dur <= 0:
            raise ValueError(f"duration_minutes must be > 0, got {dur}")

        direction = decision["direction"]
        if direction == "LONG" and not (tp > entry > sl):
            raise ValueError(
                f"LONG price ordering invalid: need TP({tp}) > entry({entry}) > SL({sl})")
        if direction == "SHORT" and not (sl > entry > tp):
            raise ValueError(
                f"SHORT price ordering invalid: need SL({sl}) > entry({entry}) > TP({tp})")

    # Set defaults for NO_TRADE
    if decision["direction"] == "NO_TRADE":
        decision.setdefault("entry_price", 0)
        decision.setdefault("take_profit", 0)
        decision.setdefault("stop_loss", 0)
        decision.setdefault("duration_minutes", 0)
        decision.setdefault("position_size_pct", 0)
