"""
Binance downloader — spot + futures OHLCV, mark/index/premium klines,
funding rates, open interest, L/S ratios, taker volume, futures basis.

Writes to NEW filenames (binance_*) — existing raw_spot_klines.csv etc. untouched.
"""

import time
import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader


class BinanceDownloader(BaseDownloader):
    name = "binance"

    def __init__(self, full=False):
        super().__init__(full=full)
        self.spot_base = self.cfg.get("spot_base_url", "https://api.binance.com")
        self.futures_base = self.cfg.get("futures_base_url", "https://fapi.binance.com")
        self.spot_ticker = self.cfg.get("spot_ticker", "BTCUSDT")
        self.futures_ticker = self.cfg.get("futures_ticker", "BTCUSDT")
        self.timeframe = self.cfg.get("timeframe", "5m")
        self.timeframe_ms = self.cfg.get("timeframe_ms", 300000)
        self.kline_limit = self.cfg.get("kline_limit", 1000)
        self.delay = self.cfg.get("rate_limit_delay", 0.3)
        self.spot_start_ms = int(datetime.strptime(
            self.cfg.get("spot_start_date", "2017-08-17"), "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc).timestamp() * 1000)
        self.futures_start_ms = int(datetime.strptime(
            self.cfg.get("futures_start_date", "2019-09-08"), "%Y-%m-%d"
        ).replace(tzinfo=timezone.utc).timestamp() * 1000)

    # ---- Klines (OHLCV) --------------------------------------------------

    def _download_klines(self, url, symbol, start_ms, filename, description):
        """Download klines using forward ms pagination."""
        self.log.info("  Downloading %s...", description)
        last_ms = self._get_last_timestamp_ms(filename, "open_time_ms")
        if last_ms and not self.full:
            start_ms = last_ms + self.timeframe_ms
            self.log.info("    Incremental from %s", self._ms_to_str(start_ms))

        raw = self._paginate_by_ms(
            url, {"symbol": symbol, "interval": self.timeframe},
            start_ms=start_ms, limit=self.kline_limit,
            step_ms=self.timeframe_ms, delay=self.delay,
        )

        if not raw:
            self.log.warning("  No data returned for %s", description)
            return

        rows = []
        for k in raw:
            rows.append({
                "open_time_ms": k[0],
                "timestamp": self._ms_to_str(k[0]) + " UTC",
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]), "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_volume": float(k[9]),
                "taker_buy_quote_volume": float(k[10]),
            })
        df = pd.DataFrame(rows)

        if self.full or not last_ms:
            self._save_csv(df, filename, description,
                           sort_by="open_time_ms", dedup_col="open_time_ms")
        else:
            self._append_csv(df, filename,
                             sort_by="open_time_ms", dedup_col="open_time_ms")

    def _download_price_klines(self, endpoint, filename, description,
                               ticker_param="symbol"):
        """Download mark/index/premium price klines (futures only).

        indexPriceKlines requires ``pair`` instead of ``symbol``.
        """
        self.log.info("  Downloading %s...", description)
        url = f"{self.futures_base}{endpoint}"
        start_ms = self.futures_start_ms
        last_ms = self._get_last_timestamp_ms(filename, "open_time_ms")
        if last_ms and not self.full:
            start_ms = last_ms + self.timeframe_ms

        raw = self._paginate_by_ms(
            url, {ticker_param: self.futures_ticker, "interval": self.timeframe},
            start_ms=start_ms, limit=self.kline_limit,
            step_ms=self.timeframe_ms, delay=self.delay,
        )

        if not raw:
            self.log.warning("  No data for %s", description)
            return

        rows = []
        for k in raw:
            rows.append({
                "open_time_ms": k[0],
                "timestamp": self._ms_to_str(k[0]) + " UTC",
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
            })
        df = pd.DataFrame(rows)

        if self.full or not last_ms:
            self._save_csv(df, filename, description,
                           sort_by="open_time_ms", dedup_col="open_time_ms")
        else:
            self._append_csv(df, filename,
                             sort_by="open_time_ms", dedup_col="open_time_ms")

    # ---- Funding Rates ----------------------------------------------------

    def _download_funding_rates(self):
        """Download historical funding rates (8h intervals)."""
        self.log.info("  Downloading Binance funding rates...")
        url = f"{self.futures_base}/fapi/v1/fundingRate"
        start_ms = self.futures_start_ms
        last_ms = self._get_last_timestamp_ms("binance_funding_rates.csv", "funding_time_ms")
        if last_ms and not self.full:
            start_ms = last_ms + 1

        all_data = []
        req_count = 0
        current = start_ms
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        limit = 1000

        while current < end_ms:
            params = {"symbol": self.futures_ticker, "startTime": current, "limit": limit}
            try:
                resp = self._http_get(url, params, delay=self.delay)
                batch = resp.json()
            except Exception as e:
                self.log.error("  Funding rates failed at %s: %s", self._ms_to_str(current), e)
                break
            req_count += 1
            if not batch:
                break
            all_data.extend(batch)
            current = batch[-1]["fundingTime"] + 1
            if len(batch) < limit:
                break
            time.sleep(self.delay)

        self.log.info("    Total: %s funding rates (%d requests)", f"{len(all_data):,}", req_count)

        if not all_data:
            return

        rows = []
        for f in all_data:
            rows.append({
                "funding_time_ms": f["fundingTime"],
                "timestamp": self._ms_to_str(f["fundingTime"]) + " UTC",
                "symbol": f.get("symbol", self.futures_ticker),
                "funding_rate": float(f["fundingRate"]),
                "mark_price": float(f.get("markPrice", 0) or 0),
            })
        df = pd.DataFrame(rows)

        if self.full or not last_ms:
            self._save_csv(df, "binance_funding_rates.csv", "Binance funding rates",
                           sort_by="funding_time_ms", dedup_col="funding_time_ms")
        else:
            self._append_csv(df, "binance_funding_rates.csv",
                             sort_by="funding_time_ms", dedup_col="funding_time_ms")

    # ---- 30-day Analytics Endpoints (OI, L/S, Taker, Basis) ---------------

    def _download_analytics(self, endpoint, filename, description, period="5m",
                            symbol_key="symbol", columns_map=None,
                            extra_params=None):
        """Download analytics from /futures/data/* endpoints.

        These retain ~30 days of history. We paginate using startTime/endTime
        within that window to get all available data (limit 500 per request).
        Older data is preserved via append + dedup.
        """
        self.log.info("  Downloading %s (30-day window, paginated)...", description)
        url = f"{self.futures_base}{endpoint}"

        # Start from the last known timestamp, or 30 days ago
        last_ms = self._get_last_timestamp_ms(filename, "timestamp_ms")
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = last_ms + 1 if last_ms else end_ms - (30 * 86_400_000)

        all_rows = []
        current = start_ms

        while current < end_ms:
            params = {symbol_key: self.futures_ticker, "period": period,
                      "limit": 500, "startTime": current}
            if extra_params:
                params.update(extra_params)
            try:
                resp = self._http_get(url, params, delay=self.delay)
                data = resp.json()
            except Exception as e:
                self.log.warning("  %s request failed at %s: %s", description,
                                 self._ms_to_str(current), e)
                break

            if isinstance(data, dict):
                code = data.get("code", 0)
                if code in (-1130, -1121, -1100):
                    # startTime out of range or invalid param — done
                    break
                if data.get("status") == "ERROR" or "error" in data:
                    self.log.warning("  %s API error: %s", description,
                                     data.get("errorData", data))
                    break
                data = data.get("data", [])

            if not data:
                break

            for d in data:
                if not isinstance(d, dict):
                    continue
                row = {"timestamp_ms": d.get("timestamp", 0)}
                row["timestamp"] = self._ms_to_str(row["timestamp_ms"]) + " UTC"
                if columns_map:
                    for csv_col, api_col in columns_map.items():
                        val = d.get(api_col, 0)
                        try:
                            row[csv_col] = float(val) if val != "" else 0
                        except (ValueError, TypeError):
                            row[csv_col] = val
                else:
                    for k, v in d.items():
                        if k not in ("timestamp", "symbol", "pair"):
                            try:
                                row[k] = float(v)
                            except (ValueError, TypeError):
                                row[k] = v
                all_rows.append(row)

            # Advance past the last timestamp in this batch
            last_batch_ts = max(d.get("timestamp", 0) for d in data if isinstance(d, dict))
            if last_batch_ts <= current:
                break  # no progress
            current = last_batch_ts + 1

            if len(data) < 500:
                break  # last page

            time.sleep(self.delay)

        self.log.info("    Total: %s %s records", f"{len(all_rows):,}", description)

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, filename, description,
                               sort_by="timestamp_ms", dedup_col="timestamp_ms")
            else:
                self._append_csv(df, filename, sort_by="timestamp_ms",
                                 dedup_col="timestamp_ms")

    # ---- Download orchestration -------------------------------------------

    def _download_all_klines(self):
        """Download all kline types."""
        # Spot
        self._download_klines(
            f"{self.spot_base}/api/v3/klines", self.spot_ticker,
            self.spot_start_ms, "binance_spot_klines_5m.csv", "Binance spot klines")
        # Futures
        self._download_klines(
            f"{self.futures_base}/fapi/v1/klines", self.futures_ticker,
            self.futures_start_ms, "binance_futures_klines_5m.csv", "Binance futures klines")
        # Mark price
        self._download_price_klines(
            "/fapi/v1/markPriceKlines", "binance_mark_price_klines.csv", "Binance mark price klines")
        # Index price (requires 'pair' not 'symbol')
        self._download_price_klines(
            "/fapi/v1/indexPriceKlines", "binance_index_price_klines.csv", "Binance index price klines",
            ticker_param="pair")
        # Premium index
        self._download_price_klines(
            "/fapi/v1/premiumIndexKlines", "binance_premium_index_klines.csv", "Binance premium index klines")

    def _download_all_analytics(self):
        """Download all 30-day analytics endpoints."""
        self._download_analytics(
            "/futures/data/openInterestHist",
            "binance_open_interest.csv", "Binance open interest",
            columns_map={
                "sum_open_interest": "sumOpenInterest",
                "sum_open_interest_value": "sumOpenInterestValue",
            })

        self._download_analytics(
            "/futures/data/topLongShortAccountRatio",
            "binance_top_ls_accounts.csv", "Binance top L/S accounts",
            columns_map={
                "long_short_ratio": "longShortRatio",
                "long_account": "longAccount",
                "short_account": "shortAccount",
            })

        self._download_analytics(
            "/futures/data/topLongShortPositionRatio",
            "binance_top_ls_positions.csv", "Binance top L/S positions",
            columns_map={
                "long_short_ratio": "longShortRatio",
                "long_account": "longAccount",
                "short_account": "shortAccount",
            })

        self._download_analytics(
            "/futures/data/globalLongShortAccountRatio",
            "binance_global_ls_ratio.csv", "Binance global L/S ratio",
            columns_map={
                "long_short_ratio": "longShortRatio",
                "long_account": "longAccount",
                "short_account": "shortAccount",
            })

        self._download_analytics(
            "/futures/data/takerlongshortRatio",
            "binance_taker_buy_sell.csv", "Binance taker buy/sell",
            columns_map={
                "buy_sell_ratio": "buySellRatio",
                "buy_vol": "buyVol",
                "sell_vol": "sellVol",
            })

        self._download_analytics(
            "/futures/data/basis",
            "binance_futures_basis.csv", "Binance futures basis",
            symbol_key="pair",
            extra_params={"contractType": "PERPETUAL"},
            columns_map={
                "index_price": "indexPrice",
                "contract_type": "contractType",
                "basis_rate": "basisRate",
                "futures_price": "futuresPrice",
                "annual_basis_rate": "annualizedBasisRate",
            })

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_all_klines()
        self._download_funding_rates()
        self._download_all_analytics()

    def download_incremental(self):
        self._download_all_klines()
        self._download_funding_rates()
        self._download_all_analytics()  # always re-fetch 30-day window


if __name__ == "__main__":
    BinanceDownloader.main()
