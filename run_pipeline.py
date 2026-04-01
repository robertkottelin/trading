"""Trading pipeline orchestrator — single entry point for the full loop.

Runs the complete pipeline:
  1. Refresh market context data (14 sources, last 24h)
  2. Run reasoning agent (ML signals → market context → portfolio →
     decision history → trade history → Grok → execution)

Usage:
    python run_pipeline.py                        # full pipeline, testnet live
    python run_pipeline.py --paper                # paper trading mode
    python run_pipeline.py --no-testnet           # mainnet (CAUTION: real funds)
    python run_pipeline.py --skip-download        # skip data refresh
    python run_pipeline.py --skip-signals         # skip ML inference
    python run_pipeline.py --skip-web-search      # disable Grok web/X search
    python run_pipeline.py --no-execute           # stop after Grok decision
    python run_pipeline.py --dry-run              # show prompt, skip Grok
    python run_pipeline.py --loop --interval 300  # repeat every 5 minutes
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

log = logging.getLogger("pipeline")

# Toggle default network — set to True for testnet, False for mainnet
TESTNET = True


def setup_logging(verbose: bool = False):
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — INFO by default, DEBUG with -v
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler — always DEBUG, timestamped log file
    log_dir = os.path.join(os.path.dirname(__file__) or ".", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    log.info("Logging to file: %s", log_file)

    # Suppress noisy dydx4 SDK warnings — the SDK has a Python 3 compat bug
    # where it calls .decode() on str objects when parsing chain event attributes,
    # and its log format strings cause "--- Logging error ---" tracebacks.
    logging.getLogger("dydx4").setLevel(logging.ERROR)
    logging.getLogger("graphviz").setLevel(logging.WARNING)


def _run_with_pg(cmd, timeout, capture=False):
    """Run a command in its own process group; kill the group on timeout.

    Returns (returncode, stdout, stderr).  stdout/stderr are empty strings
    unless *capture* is True.
    """
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
        raise


def run_market_context(hours: int = 24, tier: str = "") -> bool:
    """Refresh market context data, optionally filtered by tier."""
    tier_label = f" [{tier} tier]" if tier else " [all tiers]"
    log.info("Step 1/2: Downloading market context data (last %dh)%s...", hours, tier_label)
    cmd = [sys.executable, "-m", "downloaders.market_context",
           "--hours", str(hours)]
    if tier:
        cmd.extend(["--tier", tier])
    try:
        rc, _, stderr = _run_with_pg(cmd, timeout=600, capture=True)
        if rc != 0:
            log.warning("Market context download had errors:\n%s",
                        stderr[-500:] if stderr else "no stderr")
            # Non-fatal — stale data is better than no data
        else:
            log.info("Market context data refreshed%s", tier_label)
        return True
    except subprocess.TimeoutExpired:
        log.error("Market context download timed out (10 min)")
        return False
    except Exception as e:
        log.error("Market context download failed: %s", e)
        return False


def run_reasoning_agent(args) -> bool:
    """Run the reasoning agent (stages 1-7)."""
    log.info("Step 2/2: Running reasoning agent...")
    cmd = [sys.executable, "-m", "llm_agent.reasoning_agent"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_signals:
        cmd.append("--skip-signals")
    if args.skip_web_search:
        cmd.append("--skip-web-search")
    if args.no_execute:
        cmd.append("--no-execute")
    if args.verbose:
        cmd.append("--verbose")
    # Pass network/mode flags through
    if args.testnet:
        cmd.append("--testnet")
    else:
        cmd.append("--no-testnet")
    if args.live:
        cmd.append("--live")
    else:
        cmd.append("--paper")

    try:
        rc, _, _ = _run_with_pg(cmd, timeout=300)
        if rc != 0:
            log.error("Reasoning agent exited with code %d", rc)
            return False
        log.info("Reasoning agent completed successfully")
        return True
    except subprocess.TimeoutExpired:
        log.error("Reasoning agent timed out (5 min)")
        return False
    except Exception as e:
        log.error("Reasoning agent failed: %s", e)
        return False


HEARTBEAT_PATH = os.path.join("state_data", "heartbeat.json")


def write_heartbeat(run_number: int, status: str, success: bool | None = None,
                    elapsed_s: float | None = None, next_run_at: str | None = None):
    """Write a heartbeat file with current pipeline state."""
    os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
    heartbeat = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_number": run_number,
        "status": status,
        "pid": os.getpid(),
    }
    if success is not None:
        heartbeat["last_success"] = success
    if elapsed_s is not None:
        heartbeat["elapsed_s"] = round(elapsed_s, 1)
    if next_run_at is not None:
        heartbeat["next_run_at"] = next_run_at
    try:
        with open(HEARTBEAT_PATH, "w") as f:
            json.dump(heartbeat, f)
    except Exception as e:
        log.warning("Failed to write heartbeat: %s", e)


def run_once(args, run_number: int = 1, tier: str = "") -> bool:
    """Run the full pipeline once.

    Args:
        tier: Download tier filter ('fast', 'medium', 'slow', or '' for all).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info("=" * 60)
    log.info("Pipeline run started at %s", now)
    log.info("=" * 60)

    write_heartbeat(run_number, "running")

    # Step 1: Data refresh
    if not args.skip_download:
        run_market_context(hours=args.context_hours, tier=tier)
    else:
        log.info("Skipping market context download (--skip-download)")

    # Step 2: Reasoning + execution
    success = run_reasoning_agent(args)

    log.info("Pipeline run %s", "completed" if success else "FAILED")
    return success


def _load_tier_intervals() -> tuple[int, int]:
    """Load tier interval settings from config/settings.yaml."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        pipeline = cfg.get("pipeline", {})
        return (pipeline.get("tier_medium_interval", 6),
                pipeline.get("tier_slow_interval", 72))
    except Exception:
        return 6, 72


def _select_tier(cycle: int, medium_every: int, slow_every: int) -> str:
    """Determine which download tier to use for a given cycle number.

    Tiers are cumulative:
      - Every cycle: fast (4 exchange sources)
      - Every medium_every cycles: fast + medium (8 sources)
      - Every slow_every cycles: all sources (14)
    Cycle 1 always runs all tiers.
    """
    if cycle == 1 or cycle % slow_every == 0:
        return ""       # empty string → all sources
    if cycle % medium_every == 0:
        return "medium"  # fast + medium
    return "fast"


def main():
    parser = argparse.ArgumentParser(
        description="Trading pipeline orchestrator — data refresh + reasoning + execution"
    )

    # Network and mode
    parser.add_argument("--testnet", action="store_true",
                        default=TESTNET,
                        help="Use dYdX testnet (default: %(default)s)")
    parser.add_argument("--no-testnet", dest="testnet", action="store_false",
                        help="Use dYdX mainnet (CAUTION: real funds)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--live", action="store_true", default=True,
                            help="Execute trades on-chain (default)")
    mode_group.add_argument("--paper", dest="live", action="store_false",
                            help="Paper trading mode (no on-chain execution)")

    # Pipeline control
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip market context data refresh")
    parser.add_argument("--context-hours", type=int, default=24,
                        help="Hours of context data to fetch (default: 24)")

    # Reasoning agent passthrough flags
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompt without calling Grok")
    parser.add_argument("--skip-signals", action="store_true",
                        help="Skip ML model inference")
    parser.add_argument("--skip-web-search", action="store_true",
                        help="Disable Grok web/X search tools")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip trade execution (Stage 7)")

    # Loop mode
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously on a schedule")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between cycle starts in loop mode (default: 300)")
    parser.add_argument("--no-tiers", action="store_true",
                        help="Disable tiered downloads — fetch all sources every cycle")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    network = "testnet" if args.testnet else "mainnet"
    mode = "live" if args.live else "paper"
    log.info("Network: %s | Mode: %s", network, mode)

    if args.loop:
        medium_every, slow_every = _load_tier_intervals()
        log.info("Starting pipeline loop (interval=%ds, tiers=%s). Ctrl+C to stop.",
                 args.interval, "off" if args.no_tiers else f"med/{medium_every} slow/{slow_every}")
        run_count = 0
        while True:
            run_count += 1
            cycle_start = time.time()

            # Determine download tier for this cycle
            tier = "" if args.no_tiers else _select_tier(run_count, medium_every, slow_every)
            tier_label = "all" if not tier else tier
            log.info("--- Run #%d [%s tier] ---", run_count, tier_label)

            try:
                success = run_once(args, run_number=run_count, tier=tier)
            except Exception as e:
                log.error("Pipeline run #%d failed: %s", run_count, e)
                success = False

            # Dynamic sleep: measure elapsed, sleep the remainder
            elapsed = time.time() - cycle_start
            sleep_time = max(0, args.interval - elapsed)
            next_run_at = datetime.fromtimestamp(
                cycle_start + args.interval, tz=timezone.utc
            ).isoformat(timespec="seconds")

            write_heartbeat(run_count,
                            "completed" if success else "failed",
                            success,
                            elapsed_s=elapsed,
                            next_run_at=next_run_at)

            if sleep_time > 0:
                log.info("Cycle took %.1fs, sleeping %.1fs (next run at %s)",
                         elapsed, sleep_time, next_run_at)
            else:
                log.warning("Cycle took %.1fs (> %ds interval), starting next immediately",
                            elapsed, args.interval)
            try:
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                log.info("Pipeline stopped by user after %d runs", run_count)
                break
    else:
        success = run_once(args, run_number=1)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
