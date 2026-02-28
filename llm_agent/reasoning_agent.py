"""LLM Reasoning Agent — main orchestrator for BTC trading decisions.

Combines ML model signals, market context, portfolio state, and decision
history into a prompt for Grok, then saves the structured decision.

Usage:
    python -m llm_agent.reasoning_agent                    # full run (with execution)
    python -m llm_agent.reasoning_agent --no-execute       # skip execution stage
    python -m llm_agent.reasoning_agent --dry-run          # show prompt, skip Grok
    python -m llm_agent.reasoning_agent --skip-signals     # skip ML inference
    python -m llm_agent.reasoning_agent --skip-web-search  # disable web/X search
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

from llm_agent import signal_generator, context_builder, portfolio_reader
from llm_agent import decision_manager, grok_client, trade_history

log = logging.getLogger("llm_agent")

HEARTBEAT_PATH = os.path.join("state_data", "heartbeat.json")


def _write_heartbeat(stage: str, status: str = "running"):
    """Write heartbeat to state_data/heartbeat.json for external monitoring."""
    os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
    heartbeat = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "component": "reasoning_agent",
        "stage": stage,
        "status": status,
    }
    tmp_path = HEARTBEAT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(heartbeat, f)
    os.replace(tmp_path, HEARTBEAT_PATH)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_prompt(signals_text: str, context_text: str,
                 portfolio_text: str, history_text: str,
                 trade_history_text: str) -> str:
    """Compose the full prompt from all sections."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"CURRENT TIME: {now}",
        "",
        signals_text,
        "",
        context_text,
        "",
        portfolio_text,
        "",
        trade_history_text,
        "",
        history_text,
        "",
        "Based on ALL the above data, make your trading decision. "
        "Search the web for any breaking BTC/crypto news. "
        "Search X/Twitter for real-time crypto sentiment. "
        "Then output your decision as JSON.",
    ]
    return "\n".join(sections)


def run(args):
    """Main execution flow."""
    load_dotenv()
    _write_heartbeat("start")

    # --- Stage 0: Orphan order cleanup (live mode only) ---
    # Load config ONCE — reused by Stage 0 and Stage 7 to prevent
    # race condition if config file is modified between stages.
    with open("config/settings.yaml") as _cfg_f:
        _full_cfg = yaml.safe_load(_cfg_f)
    exec_cfg = _full_cfg.get("execution", {})
    mode = exec_cfg.get("mode", "paper")
    if mode == "live" and not args.no_execute:
        log.info("Stage 0/7: Checking for orphan orders...")
        try:
            from execution.dydx_client import DydxClient
            from execution.dydx_executor import DydxExecutor

            async def _cleanup():
                client = DydxClient()
                await client.connect()
                try:
                    executor = DydxExecutor(client, config=_full_cfg)
                    return await executor.cleanup_orphan_orders()
                finally:
                    await client.disconnect()

            cleaned = asyncio.run(_cleanup())
            if cleaned > 0:
                log.warning("Cleaned up %d orphan order(s)", cleaned)
        except Exception as e:
            log.warning("Orphan cleanup failed (non-fatal): %s", e)

        # Verify open positions have TP/SL protection
        try:
            async def _verify():
                client = DydxClient()
                await client.connect()
                try:
                    executor = DydxExecutor(client, config=_full_cfg)
                    return await executor.verify_position_protection()
                finally:
                    await client.disconnect()

            unprotected = asyncio.run(_verify())
            if unprotected > 0:
                log.critical("Emergency-closed %d unprotected position(s)", unprotected)
        except Exception as e:
            log.warning("Position protection check failed (non-fatal): %s", e)

    _write_heartbeat("stage_0_complete")

    # --- Stage 1: ML signals ---
    signals_result = None
    if args.skip_signals:
        signals_text = "ML MODEL SIGNALS: Skipped (--skip-signals flag)."
        log.info("Skipping ML signal generation")
    else:
        log.info("Stage 1/7: Generating ML signals...")
        try:
            signals_result = signal_generator.generate_signals()
            signals_text = signals_result["text_summary"]
            log.info("ML signals: %d bullish, %d neutral",
                     signals_result["consensus"].get("bullish_count", 0),
                     signals_result["consensus"].get("neutral_count", 0))
        except Exception as e:
            log.warning("Signal generation failed: %s", e)
            signals_text = f"ML MODEL SIGNALS: Error — {e}"

    _write_heartbeat("stage_1_signals")

    # --- Stage 2: Market context ---
    log.info("Stage 2/7: Building market context...")
    try:
        context_text = context_builder.build_context()
        log.info("Market context built (%d chars)", len(context_text))
    except Exception as e:
        log.warning("Context building failed: %s", e)
        context_text = f"MARKET CONTEXT: Error — {e}"

    _write_heartbeat("stage_2_context")

    # --- Stage 3: Portfolio state ---
    log.info("Stage 3/7: Reading portfolio state...")
    try:
        portfolio_text = portfolio_reader.get_portfolio()
        log.info("Portfolio state retrieved")
    except Exception as e:
        log.warning("Portfolio reader failed: %s", e)
        portfolio_text = f"PORTFOLIO STATE: Unavailable — {e}"

    _write_heartbeat("stage_3_portfolio")

    # --- Stage 4: Resolve pending decisions ---
    log.info("Stage 4/7: Resolving pending decisions...")
    try:
        resolved = decision_manager.resolve_pending()
        if resolved > 0:
            log.info("Resolved %d pending decision(s)", resolved)
    except Exception as e:
        log.warning("Decision resolution failed: %s", e)

    _write_heartbeat("stage_4_resolve")

    # --- Stage 5a: Get decision history ---
    log.info("Stage 5/7: Loading decision history...")
    try:
        history_text = decision_manager.get_recent_summary()
    except Exception as e:
        log.warning("History loading failed: %s", e)
        history_text = "RECENT DECISIONS: Unavailable"

    # --- Stage 5b: Get trade execution history ---
    try:
        trade_history_text = trade_history.get_trade_history()
        log.info("Trade history loaded")
    except Exception as e:
        log.warning("Trade history loading failed: %s", e)
        trade_history_text = "TRADE HISTORY: Unavailable"

    _write_heartbeat("stage_5_history")

    # --- Compose prompt ---
    prompt = build_prompt(signals_text, context_text, portfolio_text,
                          history_text, trade_history_text)

    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN — Full prompt that would be sent to Grok:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print(f"\nPrompt length: {len(prompt)} chars, ~{len(prompt)//4} tokens")
        print("Exiting (--dry-run)")
        return

    # --- Stage 6: Call Grok ---
    log.info("Stage 6/7: Calling Grok for decision...")
    try:
        enable_web = not args.skip_web_search
        decision = grok_client.get_decision(prompt, enable_web_search=enable_web)
    except Exception as e:
        log.error("Grok API failed: %s", e)
        _write_heartbeat("stage_6_failed", status="error")
        print(f"\nERROR: Grok API call failed: {e}")
        sys.exit(1)

    # Add market metadata to decision
    decision["market_conditions"] = _extract_market_conditions(context_text)

    if signals_result is not None:
        decision["model_consensus"] = signals_result.get("consensus", {})

    # --- Save decision ---
    try:
        decision_manager.save_decision(decision)
    except Exception as e:
        log.error("Failed to save decision: %s", e)

    # --- Print summary ---
    _print_summary(decision)

    _write_heartbeat("stage_6_decision")

    # --- Stage 7: Execute trade ---
    if not args.no_execute:
        log.info("Stage 7/7: Executing trade...")
        try:
            # exec_cfg and mode already loaded once at Stage 0
            if mode == "paper":
                from execution.paper_executor import PaperExecutor
                trade = PaperExecutor().execute_decision(decision)
            else:
                from execution.dydx_client import DydxClient
                from execution.dydx_executor import DydxExecutor
                from execution.risk_manager import RiskManager

                async def _live_execute():
                    client = DydxClient()
                    await client.connect()
                    try:
                        executor = DydxExecutor(client, RiskManager(exec_cfg))
                        return await executor.execute_decision(decision)
                    finally:
                        await client.disconnect()

                trade = asyncio.run(_live_execute())

            status = trade.get("status", "?")
            action = trade.get("action", "?")
            log.info("Execution complete: %s — %s", action, status)
            if action == "REJECTED":
                print(f"  Trade rejected: {trade.get('rejection_reason', '?')}")
            elif action == "ENTRY":
                print(f"  {trade.get('direction')} {trade.get('size_btc')} BTC "
                      f"@ ${trade.get('fill_price', 0):,.2f} [{status}]")
        except Exception as e:
            log.error("Execution failed: %s", e)
            print(f"\nExecution error: {e}")
    else:
        log.info("Skipping execution (--no-execute flag)")

    _write_heartbeat("complete", status="success")


def _extract_market_conditions(context_text: str) -> dict:
    """Extract key market conditions from context text for decision metadata."""
    conditions = {}

    # Parse BTC price
    for line in context_text.split("\n"):
        if "BTC Price" in line and "$" in line:
            try:
                price_str = line.split("$")[1].split()[0].replace(",", "")
                conditions["btc_price"] = float(price_str)
            except (IndexError, ValueError):
                pass
        elif "Fear & Greed" in line:
            try:
                parts = line.split(":")
                if len(parts) > 1:
                    val = parts[1].strip().split()[0]
                    conditions["fng_value"] = int(val)
            except (ValueError, IndexError):
                pass
        elif "DXY:" in line:
            try:
                val = line.split("DXY:")[1].strip().split()[0]
                conditions["dxy"] = float(val)
            except (ValueError, IndexError):
                pass
        elif "Binance:" in line and "ann." in line and "funding_rate" not in str(conditions):
            try:
                rate_str = line.split(":")[1].strip().split()[0]
                conditions["funding_rate"] = float(rate_str)
            except (ValueError, IndexError):
                pass

    return conditions


def _print_summary(decision: dict):
    """Print a human-readable summary of the decision."""
    print("\n" + "=" * 60)
    print("TRADING DECISION")
    print("=" * 60)

    direction = decision.get("direction", "?")
    confidence = decision.get("confidence", 0)

    if direction == "NO_TRADE":
        print(f"  Direction:  NO TRADE")
        print(f"  Confidence: {confidence:.1%}")
    else:
        print(f"  Direction:  {direction}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Entry:      ${decision.get('entry_price', 0):,.2f}")
        print(f"  Take Profit:${decision.get('take_profit', 0):,.2f}")
        print(f"  Stop Loss:  ${decision.get('stop_loss', 0):,.2f}")
        print(f"  Duration:   {decision.get('duration_minutes', 0)} min")
        print(f"  Size:       {decision.get('position_size_pct', 0):.1%} of equity")

        # Risk:reward
        entry = decision.get("entry_price", 0)
        tp = decision.get("take_profit", 0)
        sl = decision.get("stop_loss", 0)
        if entry and sl and tp:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            print(f"  Risk:Reward: 1:{rr:.2f}")

    print(f"\n  Rationale:")
    rationale = decision.get("rationale", "N/A")
    # Word-wrap rationale at ~70 chars
    words = rationale.split()
    line = "    "
    for word in words:
        if len(line) + len(word) > 74:
            print(line)
            line = "    " + word
        else:
            line += (" " if len(line) > 4 else "") + word
    if line.strip():
        print(line)

    print("=" * 60)
    print(f"Decision saved to: llm_agent/decision.json")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Reasoning Agent — BTC trading decision engine"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show full prompt without calling Grok")
    parser.add_argument("--skip-signals", action="store_true",
                        help="Skip ML model inference (faster for debugging)")
    parser.add_argument("--skip-web-search", action="store_true",
                        help="Disable Grok web_search and x_search tools")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip trade execution (Stage 7)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
