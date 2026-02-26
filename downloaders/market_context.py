"""
Market Context Fetcher — downloads last 24h of all data sources
into market_context_data/ for LLM context.

Reuses the same downloaders as download_all.py but instantiates each with:
  - output_dir = market_context_data/
  - start_override_ms = now - hours * 3600 * 1000

Usage:
  python -m downloaders.market_context                    # all 14, last 24h
  python -m downloaders.market_context --hours 48         # last 48h
  python -m downloaders.market_context --sources dydx,binance  # specific sources
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from downloaders.download_all import DOWNLOADERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKET_CONTEXT_DIR = PROJECT_ROOT / "market_context_data"


def setup_logging():
    """Set up orchestrator-only logger (separate from downloader loggers)."""
    log = logging.getLogger("market_context")
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(ch)
    return log


def main():
    parser = argparse.ArgumentParser(
        description="Download recent market data for LLM context")
    parser.add_argument("--hours", type=int, default=24,
                        help="Hours of recent data to fetch (default: 24)")
    parser.add_argument("--sources", type=str, default="",
                        help="Comma-separated list of sources (default: all)")
    args = parser.parse_args()

    log = setup_logging()

    # Determine which sources to run
    if args.sources:
        requested = [s.strip() for s in args.sources.split(",")]
        unknown = [s for s in requested if s not in DOWNLOADERS]
        if unknown:
            log.error("Unknown sources: %s. Available: %s",
                      ", ".join(unknown), ", ".join(DOWNLOADERS.keys()))
            raise SystemExit(1)
        sources = {k: DOWNLOADERS[k] for k in requested}
    else:
        sources = DOWNLOADERS

    # Compute start_override_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - args.hours * 3600 * 1000

    log.info("=" * 70)
    log.info("MARKET CONTEXT — last %dh — %d sources", args.hours, len(sources))
    log.info("  Output: %s", MARKET_CONTEXT_DIR)
    log.info("  Start:  %s UTC", datetime.fromtimestamp(
        start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 70)

    results = {}
    total_t0 = time.time()

    for name, cls in sources.items():
        t0 = time.time()
        try:
            dl = cls(
                full=False,
                output_dir=str(MARKET_CONTEXT_DIR),
                start_override_ms=start_ms,
            )
            dl.download_recent(hours=args.hours)
            elapsed = time.time() - t0
            results[name] = ("OK", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = (f"ERROR: {e}", elapsed)
            log.error("  %s crashed: %s", name, e)

    total_elapsed = time.time() - total_t0

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("MARKET CONTEXT SUMMARY — %.1fs total", total_elapsed)
    log.info("=" * 70)
    ok_count = 0
    for name, (status, elapsed) in results.items():
        icon = "OK" if status == "OK" else "FAIL"
        if status == "OK":
            ok_count += 1
        log.info("  [%4s] %-20s  (%.1fs)  %s", icon, name, elapsed,
                 "" if status == "OK" else status)
    log.info("=" * 70)
    log.info("  %d/%d sources OK", ok_count, len(results))
    log.info("  Output: %s", MARKET_CONTEXT_DIR)

    # Exit with error if any failed
    failed = [n for n, (s, _) in results.items() if s != "OK"]
    if failed:
        log.warning("Failed sources: %s", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
