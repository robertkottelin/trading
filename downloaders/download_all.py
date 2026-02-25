"""
Orchestrator — runs all data source downloaders with error isolation.

Usage:
  python -m downloaders.download_all              # incremental (default)
  python -m downloaders.download_all --full       # full history
  python -m downloaders.download_all --sources dydx,binance   # specific sources only
"""

import argparse
import time
import logging

from downloaders.dydx_hist import DydxDownloader
from downloaders.binance_hist import BinanceDownloader
from downloaders.deribit_hist import DeribitDownloader
from downloaders.bybit_hist import BybitDownloader
from downloaders.okx_hist import OkxDownloader
from downloaders.coinbase_premium_hist import CoinbasePremiumDownloader
from downloaders.macro_hist import MacroDownloader
from downloaders.sentiment_hist import SentimentDownloader
from downloaders.btc_network_hist import BtcNetworkDownloader
from downloaders.blockchain_hist import BlockchainDownloader
from downloaders.defi_hist import DefiDownloader
from downloaders.coinalyze_hist import CoinalyzeDownloader
from downloaders.hyperliquid_hist import HyperliquidDownloader
from downloaders.cftc_hist import CftcDownloader

# Registry: order matters (exchanges first, then macro/sentiment/on-chain)
DOWNLOADERS = {
    "dydx": DydxDownloader,
    "binance": BinanceDownloader,
    "deribit": DeribitDownloader,
    "bybit": BybitDownloader,
    "okx": OkxDownloader,
    "coinbase_premium": CoinbasePremiumDownloader,
    "hyperliquid": HyperliquidDownloader,
    "macro": MacroDownloader,
    "sentiment": SentimentDownloader,
    "btc_network": BtcNetworkDownloader,
    "blockchain": BlockchainDownloader,
    "defi": DefiDownloader,
    "cftc": CftcDownloader,
    "coinalyze": CoinalyzeDownloader,
}


def setup_root_logging():
    log = logging.getLogger("data")
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(ch)
    return log


def main():
    parser = argparse.ArgumentParser(description="Download all data sources")
    parser.add_argument("--full", action="store_true", help="Full history (not incremental)")
    parser.add_argument("--sources", type=str, default="",
                        help="Comma-separated list of sources (default: all)")
    args = parser.parse_args()

    log = setup_root_logging()

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

    mode = "FULL" if args.full else "INCREMENTAL"
    log.info("=" * 70)
    log.info("DATA DOWNLOAD — %s — %d sources", mode, len(sources))
    log.info("=" * 70)

    results = {}
    total_t0 = time.time()

    for name, cls in sources.items():
        t0 = time.time()
        try:
            dl = cls(full=args.full)
            success = dl.run()
            elapsed = time.time() - t0
            results[name] = ("OK" if success else "FAILED", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            results[name] = (f"ERROR: {e}", elapsed)
            log.error("  %s crashed: %s", name, e)

    total_elapsed = time.time() - total_t0

    # Summary
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY — %.1fs total", total_elapsed)
    log.info("=" * 70)
    for name, (status, elapsed) in results.items():
        icon = "OK" if status == "OK" else "FAIL"
        log.info("  [%4s] %-15s  (%.1fs)  %s", icon, name, elapsed,
                 "" if status == "OK" else status)
    log.info("=" * 70)

    # Exit with error if any failed
    failed = [n for n, (s, _) in results.items() if s != "OK"]
    if failed:
        log.warning("Failed sources: %s", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
