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
import logging
import sys
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

from llm_agent import signal_generator, context_builder, portfolio_reader
from llm_agent import decision_manager, grok_client, trade_history
from strategies.engine import StrategyEngine

log = logging.getLogger("llm_agent")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy dydx4 SDK warnings (Python 3 .decode() compat bug)
    logging.getLogger("dydx4").setLevel(logging.ERROR)
    logging.getLogger("graphviz").setLevel(logging.WARNING)


def build_prompt(signals_text: str, strategies_text: str, context_text: str,
                 portfolio_text: str, history_text: str,
                 trade_history_text: str) -> str:
    """Compose the full prompt from all sections."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        f"CURRENT TIME: {now}",
        "",
        signals_text,
        "",
        strategies_text,
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

    # --- Stage 0: Orphan cleanup + position monitoring (live mode only) ---
    exec_cfg = yaml.safe_load(
        open("config/settings.yaml")
    ).get("execution", {})

    # CLI overrides for network and mode
    if hasattr(args, "testnet"):
        network = "testnet" if args.testnet else "mainnet"
        exec_cfg["network"] = network
    if hasattr(args, "live"):
        exec_cfg["mode"] = "live" if args.live else "paper"

    network = exec_cfg.get("network", "testnet")
    mode = exec_cfg.get("mode", "paper")
    log.info("Config override — network: %s, mode: %s", network, mode)
    if mode == "live" and not args.no_execute:
        log.info("Stage 0/7: Orphan cleanup + position monitoring + trailing stops...")
        try:
            from execution.dydx_client import DydxClient
            from execution.dydx_executor import DydxExecutor

            async def _startup_checks():
                client = DydxClient(config=exec_cfg)
                await client.connect()
                try:
                    executor  = DydxExecutor(client, config=exec_cfg)
                    cleaned   = await executor.cleanup_orphan_orders()
                    monitor   = await executor.verify_position_orders()
                    trail     = await executor.trail_stops()
                    portfolio = await client.get_portfolio_state()
                    return cleaned, monitor, trail, portfolio
                finally:
                    await client.disconnect()

            cleaned, monitor, trail, startup_portfolio = asyncio.run(_startup_checks())
            if cleaned > 0:
                log.warning("Cleaned up %d orphan order(s)", cleaned)
            if monitor["unprotected"]:
                for p in monitor["unprotected"]:
                    log.warning(
                        "UNPROTECTED position: %s %s %s — missing %s",
                        p["side"], p["size"], p["market"],
                        "+".join(p["missing"]),
                    )
            if trail["stops_updated"]:
                for s in trail["stops_updated"]:
                    log.info("Trail stop updated: %s SL → $%.0f (%s)",
                             s["market"], s["new_sl"], s["tier"])
            if trail["errors"]:
                log.warning("Trail stop errors: %s", trail["errors"])

            # Early exit if in an active trade — Grok call not needed since
            # max_open_positions=1 means any new signal would be rejected anyway.
            # TP/SL orders + trailing stop already manage the position.
            if startup_portfolio.get("positions"):
                pos = startup_portfolio["positions"][0]
                log.info(
                    "Active %s position in %s — skipping Grok API call (saves API cost). "
                    "Position protected by TP/SL orders and trailing stop.",
                    pos.get("side", "?"), pos.get("market", "?"),
                )
                return

        except Exception as e:
            log.warning("Startup checks failed (non-fatal): %s", e)

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

    # --- Stage 1.5: Conventional strategy signals ---
    log.info("Stage 1.5/7: Generating conventional strategy signals...")
    try:
        strategy_engine = StrategyEngine(data_dir="raw_data")
        strategy_result = strategy_engine.generate_signals()
        strategies_text = strategy_result["text_summary"]
        log.info("Strategy signals: %d long, %d short, %d inactive",
                 strategy_result["consensus"].get("long_count", 0),
                 strategy_result["consensus"].get("short_count", 0),
                 strategy_result["consensus"].get("inactive_count", 0))
    except Exception as e:
        log.warning("Strategy signal generation failed: %s", e)
        strategies_text = "CONVENTIONAL STRATEGY SIGNALS: Error — " + str(e)

    # --- Stage 2: Market context ---
    log.info("Stage 2/7: Building market context...")
    try:
        context_text = context_builder.build_context()
        log.info("Market context built (%d chars)", len(context_text))
    except Exception as e:
        log.warning("Context building failed: %s", e)
        context_text = f"MARKET CONTEXT: Error — {e}"

    # --- Stage 3: Portfolio state ---
    log.info("Stage 3/7: Reading portfolio state...")
    try:
        portfolio_text = portfolio_reader.get_portfolio(network=network)
        log.info("Portfolio state retrieved")
    except Exception as e:
        log.warning("Portfolio reader failed: %s", e)
        portfolio_text = f"PORTFOLIO STATE: Unavailable — {e}"

    # --- Stage 4: Resolve pending decisions ---
    log.info("Stage 4/7: Resolving pending decisions...")
    try:
        resolved = decision_manager.resolve_pending()
        if resolved > 0:
            log.info("Resolved %d pending decision(s)", resolved)
    except Exception as e:
        log.warning("Decision resolution failed: %s", e)

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

    # --- Compose prompt ---
    prompt = build_prompt(signals_text, strategies_text, context_text,
                          portfolio_text, history_text, trade_history_text)

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

    # --- Stage 7: Execute trade ---
    if not args.no_execute:
        log.info("Stage 7/7: Executing trade...")
        try:
            if mode == "paper":
                from execution.paper_executor import PaperExecutor
                trade = PaperExecutor(config={"execution": exec_cfg}).execute_decision(decision)
            else:
                from execution.dydx_client import DydxClient
                from execution.dydx_executor import DydxExecutor
                from execution.risk_manager import RiskManager

                async def _live_execute():
                    client = DydxClient(config=exec_cfg)
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
        size_usd = decision.get("position_size_usd")
        if size_usd is not None:
            print(f"  Size:       ${size_usd:.0f} USD notional")
        else:
            print(f"  Size:       {decision.get('position_size_pct', 0):.1%} of equity (legacy)")

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
    parser.add_argument("--testnet", action="store_true", default=True,
                        help="Use dYdX testnet (default)")
    parser.add_argument("--no-testnet", dest="testnet", action="store_false",
                        help="Use dYdX mainnet")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", default=True,
                            help="Execute trades on-chain (default)")
    mode_group.add_argument("--paper", dest="live", action="store_false",
                            help="Paper trading mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    run(args)


if __name__ == "__main__":
    main()
