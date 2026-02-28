"""
BaseDownloader — abstract base class for all data downloaders.

Provides: HTTP retry, pagination (forward-ms + backward-ISO), incremental CSV,
logging, and CLI entry point.
"""

import abc
import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "raw_data"
LOG_DIR = PROJECT_ROOT / "downloaders" / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
ENV_FILE = PROJECT_ROOT / ".env"


# ---------------------------------------------------------------------------
# BaseDownloader ABC
# ---------------------------------------------------------------------------

class BaseDownloader(abc.ABC):
    """Abstract base for every data source downloader."""

    name: str = "base"  # override in subclass

    def __init__(self, full: bool = False, output_dir: str = None,
                 start_override_ms: int = None):
        self.full = full
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.start_override_ms = start_override_ms
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_config()
        self.setup_logging()

    @property
    def start_override_iso(self):
        """Return start_override as ISO string, or None."""
        if self.start_override_ms:
            return self._ms_to_iso(self.start_override_ms)
        return None

    # ---- Config -----------------------------------------------------------

    def _load_config(self):
        """Load settings.yaml + .env into self.cfg / env vars."""
        load_dotenv(ENV_FILE)
        self.cfg = {}
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                all_cfg = yaml.safe_load(f) or {}
            self.cfg = all_cfg.get(self.name, {})
            self.cfg["_global"] = {k: v for k, v in all_cfg.items() if not isinstance(v, dict)}

    # ---- Logging ----------------------------------------------------------

    def setup_logging(self):
        self.log = logging.getLogger(f"data.{self.name}")
        if self.log.handlers:
            return  # already set up
        self.log.setLevel(logging.DEBUG)
        # file handler
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = LOG_DIR / f"{self.name}.log"
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.log.addHandler(fh)
        self.log.addHandler(ch)

    # ---- HTTP -------------------------------------------------------------

    def _http_get(self, url, params=None, timeout=30, max_retries=None, backoff=2.0, delay=0.0):
        """GET with exponential backoff, 429/Retry-After handling, 5xx retry."""
        max_retries = max_retries or self.cfg.get("max_retries", 3)
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=timeout)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", backoff * attempt))
                    self.log.warning("Rate-limited (429) — sleeping %ds", retry_after)
                    time.sleep(retry_after)
                    continue
                if resp.status_code >= 500:
                    self.log.warning("Server error %d — retry %d/%d", resp.status_code, attempt, max_retries)
                    time.sleep(backoff * attempt)
                    continue
                resp.raise_for_status()
                if delay > 0:
                    time.sleep(delay)
                return resp
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                self.log.warning("Connection error — retry %d/%d: %s", attempt, max_retries, e)
                time.sleep(backoff * attempt)
            except requests.exceptions.Timeout as e:
                last_exc = e
                self.log.warning("Timeout — retry %d/%d", attempt, max_retries)
                time.sleep(backoff * attempt)
            except requests.exceptions.HTTPError:
                raise
        raise requests.exceptions.ConnectionError(f"Failed after {max_retries} retries: {last_exc}")

    # ---- Pagination: forward by millisecond timestamps (Binance-style) ----

    def _paginate_by_ms(self, url, params_base, start_ms, end_ms=None,
                        limit=1000, step_ms=5 * 60 * 1000, delay=0.3,
                        start_key="startTime", end_key="endTime"):
        """Forward pagination via millisecond timestamps.

        Returns list of raw JSON responses (each a list).
        """
        if end_ms is None:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_data = []
        current = start_ms
        req_count = 0
        while current <= end_ms:
            params = {**params_base, start_key: current, end_key: end_ms, "limit": limit}
            try:
                resp = self._http_get(url, params)
                batch = resp.json()
            except Exception as e:
                self.log.error("Pagination failed at %s after %d reqs: %s",
                               self._ms_to_str(current), req_count, e)
                break
            req_count += 1
            if not batch:
                break
            all_data.extend(batch)
            # Advance past last element
            if isinstance(batch[-1], list):
                current = batch[-1][0] + step_ms  # kline-style: first elem is open_time
            elif isinstance(batch[-1], dict) and "time" in batch[-1]:
                current = batch[-1]["time"] + step_ms
            else:
                current = end_ms  # safety: stop
            if len(batch) < limit:
                break
            if len(all_data) % 10000 < limit:
                self.log.info("    ... %s rows (up to %s)", f"{len(all_data):,}", self._ms_to_str(current))
            time.sleep(delay)
        self.log.info("    Paginated %s rows (%d requests)", f"{len(all_data):,}", req_count)
        return all_data

    # ---- Pagination: backward by ISO timestamps (dYdX-style) ----

    def _paginate_backward_iso(self, url, params_base, before_key="createdBeforeOrAt",
                               timestamp_field="createdAt", limit=100, delay=0.5,
                               start_iso=None, stop_iso=None):
        """Backward pagination for APIs that return newest-first.

        Keeps paginating until either stop_iso is reached or no more data.
        Returns list of dicts sorted oldest-first.
        """
        all_data = []
        req_count = 0
        before_val = start_iso  # None means "from now"
        while True:
            params = {**params_base, "limit": limit}
            if before_val:
                params[before_key] = before_val
            try:
                resp = self._http_get(url, params)
                body = resp.json()
            except Exception as e:
                self.log.error("Backward pagination failed after %d reqs: %s", req_count, e)
                break
            req_count += 1
            # dYdX wraps results in various keys
            if isinstance(body, dict):
                # Try common wrapper keys
                for key in (timestamp_field.split("_")[0] + "s", "results",
                            "candles", "trades", "historicalFunding", "fundingRates"):
                    if key in body:
                        batch = body[key]
                        break
                else:
                    batch = body if isinstance(body, list) else []
            else:
                batch = body

            if not batch:
                break

            # Check stop condition — save cursor from raw batch before filtering
            raw_batch_len = len(batch)
            raw_oldest_ts = batch[-1].get(timestamp_field) if batch else None
            if stop_iso:
                filtered = []
                reached_stop = False
                for item in batch:
                    ts = item.get(timestamp_field, "")
                    if ts and ts <= stop_iso:
                        reached_stop = True
                        continue  # skip items before stop point (already have them)
                    filtered.append(item)
                batch = filtered
                if reached_stop:
                    # We've reached the boundary — no more pages needed
                    all_data.extend(batch)
                    break

            all_data.extend(batch)

            if raw_batch_len < limit:
                break

            # Next page: use oldest item from raw batch (pre-filter) to avoid gaps
            oldest_ts = raw_oldest_ts
            if not oldest_ts or oldest_ts == before_val:
                break
            before_val = oldest_ts

            if len(all_data) % 5000 < limit:
                self.log.info("    ... %s rows (back to %s)", f"{len(all_data):,}", oldest_ts[:19])
            time.sleep(delay)

        self.log.info("    Paginated %s rows backward (%d requests)", f"{len(all_data):,}", req_count)
        # Reverse to chronological order
        all_data.reverse()
        return all_data

    # ---- CSV helpers ------------------------------------------------------

    def _csv_path(self, filename):
        return self.output_dir / filename

    def _get_last_timestamp(self, filename, ts_col="timestamp"):
        """Read last timestamp from existing CSV (string ISO format).

        Uses tail-read to avoid loading entire file into memory.
        """
        path = self._csv_path(filename)
        if not path.exists():
            return None
        try:
            # Read only the last few rows to get the last timestamp
            # This avoids reading 800K+ row files fully into memory
            df = pd.read_csv(path, usecols=[ts_col])
            if df.empty:
                return None
            return str(df[ts_col].iloc[-1])
        except Exception:
            return None

    def _get_last_timestamp_ms(self, filename, ts_col="open_time_ms"):
        """Read last timestamp from existing CSV (millisecond format)."""
        path = self._csv_path(filename)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, usecols=[ts_col])
            if df.empty:
                return None
            return int(df[ts_col].iloc[-1])
        except Exception:
            return None

    def _save_csv(self, df, filename, description="", sort_by=None, dedup_col=None):
        """Save DataFrame to CSV with optional sort + dedup."""
        if df.empty:
            self.log.warning("  Skipping empty DataFrame for %s", filename)
            return
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)
        if dedup_col and dedup_col in df.columns:
            df = df.drop_duplicates(subset=[dedup_col], keep="last").reset_index(drop=True)
        path = self._csv_path(filename)
        df.to_csv(path, index=False)
        desc = description or filename
        self.log.info("  Saved %s -> %s (%s rows)", desc, path.name, f"{len(df):,}")

    def _append_csv(self, df_new, filename, sort_by=None, dedup_col=None):
        """Append new data to existing CSV with dedup on merge."""
        path = self._csv_path(filename)
        if path.exists():
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)
        if dedup_col and dedup_col in df.columns:
            df = df.drop_duplicates(subset=[dedup_col], keep="last").reset_index(drop=True)
        df.to_csv(path, index=False)
        self.log.info("  Appended -> %s (%s total rows)", path.name, f"{len(df):,}")

    # ---- Utilities --------------------------------------------------------

    @staticmethod
    def _ms_to_str(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _iso_to_ms(iso_str):
        """Parse ISO 8601 string to milliseconds.

        Handles both naive (assumed UTC) and timezone-aware strings correctly.
        """
        ts = pd.Timestamp(iso_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp() * 1000)

    @staticmethod
    def _ms_to_iso(ms):
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # ---- Entry points -----------------------------------------------------

    @abc.abstractmethod
    def download_all(self):
        """Download full history for this source."""

    @abc.abstractmethod
    def download_incremental(self):
        """Download only new data since last CSV row."""

    def download_recent(self, hours=24):
        """Download recent data only. Default: delegates to download_all."""
        self.download_all()

    def run(self, full=None):
        """Entry point: download with timing and error wrapping."""
        if full is not None:
            self.full = full
        mode = "FULL" if self.full else "INCREMENTAL"
        self.log.info("=" * 60)
        self.log.info("%s — %s download starting", self.name.upper(), mode)
        self.log.info("=" * 60)
        t0 = time.time()
        try:
            if self.full:
                self.download_all()
            else:
                self.download_incremental()
            elapsed = time.time() - t0
            self.log.info("%s — completed in %.1fs", self.name.upper(), elapsed)
            return True
        except Exception as e:
            elapsed = time.time() - t0
            self.log.error("%s — FAILED after %.1fs: %s", self.name.upper(), elapsed, e)
            self.log.debug("Traceback:", exc_info=True)
            return False

    @classmethod
    def main(cls):
        """CLI entry point for standalone execution."""
        parser = argparse.ArgumentParser(description=f"Download {cls.name} data")
        parser.add_argument("--full", action="store_true", help="Full history (not incremental)")
        args = parser.parse_args()
        dl = cls(full=args.full)
        success = dl.run()
        raise SystemExit(0 if success else 1)
