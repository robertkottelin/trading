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
  python -m downloaders.market_context --tier fast        # fast-tier only (4 exchange sources)
  python -m downloaders.market_context --tier medium      # fast + medium tiers
  python -m downloaders.market_context --tier slow        # fast + medium + slow (= all)
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from downloaders.download_all import DOWNLOADERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKET_CONTEXT_DIR = PROJECT_ROOT / "market_context_data"

# Tier definitions — loaded from config, with hardcoded fallbacks
_DEFAULT_TIERS = {
    "fast": ["dydx", "binance", "bybit", "okx"],
    "medium": ["deribit", "hyperliquid", "coinbase_premium", "coinalyze"],
    "slow": ["macro", "sentiment", "btc_network", "blockchain", "defi", "cftc"],
}


def _load_tier_sources() -> dict[str, list[str]]:
    """Load tier source lists from config/settings.yaml, falling back to defaults."""
    cfg_path = PROJECT_ROOT / "config" / "settings.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("pipeline", {}).get("tier_sources", _DEFAULT_TIERS)
    except Exception:
        return _DEFAULT_TIERS


def resolve_tier_sources(tier: str) -> list[str]:
    """Return the list of source names for a given tier level.

    Tiers are cumulative: 'fast' → fast only, 'medium' → fast+medium,
    'slow' or 'all' → all sources.
    """
    tiers = _load_tier_sources()
    if tier == "fast":
        return list(tiers.get("fast", []))
    elif tier == "medium":
        return list(tiers.get("fast", [])) + list(tiers.get("medium", []))
    else:  # "slow" or "all"
        return (list(tiers.get("fast", []))
                + list(tiers.get("medium", []))
                + list(tiers.get("slow", [])))


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
    parser.add_argument("--tier", type=str, default="",
                        choices=["", "fast", "medium", "slow", "all"],
                        help="Download tier: fast (4 exchanges), medium (+4 hourly), "
                             "slow/all (+6 daily). Cumulative. Overridden by --sources.")
    args = parser.parse_args()

    log = setup_logging()

    # Determine which sources to run (--sources takes precedence over --tier)
    if args.sources:
        requested = [s.strip() for s in args.sources.split(",")]
        unknown = [s for s in requested if s not in DOWNLOADERS]
        if unknown:
            log.error("Unknown sources: %s. Available: %s",
                      ", ".join(unknown), ", ".join(DOWNLOADERS.keys()))
            raise SystemExit(1)
        sources = {k: DOWNLOADERS[k] for k in requested}
    elif args.tier:
        requested = resolve_tier_sources(args.tier)
        unknown = [s for s in requested if s not in DOWNLOADERS]
        if unknown:
            log.warning("Tier references unknown sources (skipping): %s",
                        ", ".join(unknown))
        sources = {k: DOWNLOADERS[k] for k in requested if k in DOWNLOADERS}
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
