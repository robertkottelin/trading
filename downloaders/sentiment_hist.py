"""
Sentiment data — Alternative.me Fear & Greed, CoinGecko market data, Google Trends.
"""

import time
import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None


class SentimentDownloader(BaseDownloader):
    name = "sentiment"

    def __init__(self, full=False):
        super().__init__(full=full)
        self.fng_url = self.cfg.get("fng_url", "https://api.alternative.me/fng/")
        self.cg_base = self.cfg.get("coingecko_base_url", "https://api.coingecko.com/api/v3")
        self.cg_demo_key = self.cfg.get("coingecko_demo_key", "")
        self.delay = self.cfg.get("rate_limit_delay", 1.0)

    # ---- Fear & Greed Index -----------------------------------------------

    def _download_fear_greed(self):
        """Download full Fear & Greed history from Alternative.me."""
        self.log.info("  Downloading Fear & Greed Index...")
        params = {"limit": 0, "format": "json"}  # limit=0 = all history
        try:
            resp = self._http_get(self.fng_url, params)
            body = resp.json()
        except Exception as e:
            self.log.error("  Fear & Greed download failed: %s", e)
            return

        data = body.get("data", [])
        if not data:
            self.log.warning("  No Fear & Greed data returned")
            return

        rows = []
        for d in data:
            ts = int(d.get("timestamp", 0))
            rows.append({
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "fng_value": int(d.get("value", 0)),
                "fng_classification": d.get("value_classification", ""),
            })

        df = pd.DataFrame(rows)
        self._save_csv(df, "sentiment_fear_greed.csv", "Fear & Greed Index",
                       sort_by="date", dedup_col="date")

    # ---- CoinGecko: BTC dominance + total market cap ----------------------

    def _download_coingecko_market(self):
        """Download BTC market chart (price, mcap, volume) for dominance calc."""
        self.log.info("  Downloading CoinGecko BTC market data...")

        # CoinGecko free API: max 365 days per request
        cg_params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
        if self.cg_demo_key:
            cg_params["x_cg_demo_api_key"] = self.cg_demo_key

        # BTC market chart
        url = f"{self.cg_base}/coins/bitcoin/market_chart"
        try:
            resp = self._http_get(url, cg_params)
            btc_data = resp.json()
        except Exception as e:
            self.log.error("  CoinGecko BTC market chart failed: %s", e)
            return

        time.sleep(self.delay)

        # Total market cap
        url_global = f"{self.cg_base}/global"
        global_params = {}
        if self.cg_demo_key:
            global_params["x_cg_demo_api_key"] = self.cg_demo_key
        try:
            resp = self._http_get(url_global, global_params)
            global_data = resp.json().get("data", {})
        except Exception as e:
            self.log.warning("  CoinGecko global data failed: %s", e)
            global_data = {}

        # Parse BTC market chart
        prices = btc_data.get("prices", [])
        mcaps = btc_data.get("market_caps", [])
        volumes = btc_data.get("total_volumes", [])

        rows = []
        for i, p in enumerate(prices):
            ts_ms = p[0]
            date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            row = {
                "date": date,
                "btc_price": p[1],
                "btc_market_cap": mcaps[i][1] if i < len(mcaps) else None,
                "btc_volume_usd": volumes[i][1] if i < len(volumes) else None,
            }
            rows.append(row)

        if not rows:
            self.log.warning("  No CoinGecko market data")
            return

        df = pd.DataFrame(rows)

        # Add current global stats — point-in-time snapshot, only for today
        if global_data:
            btc_dom = global_data.get("market_cap_percentage", {}).get("btc")
            total_mcap = global_data.get("total_market_cap", {}).get("usd")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            df["btc_dominance"] = None
            df["total_market_cap"] = None
            if btc_dom is not None:
                df.loc[df["date"] == today, "btc_dominance"] = btc_dom
            if total_mcap is not None:
                df.loc[df["date"] == today, "total_market_cap"] = total_mcap

        self._save_csv(df, "sentiment_market.csv", "CoinGecko market data",
                       sort_by="date", dedup_col="date")

    # ---- Google Trends ----------------------------------------------------

    def _download_google_trends(self):
        """Download Google Trends for bitcoin-related keywords."""
        if TrendReq is None:
            self.log.warning("  pytrends not installed — skipping Google Trends")
            return

        self.log.info("  Downloading Google Trends...")
        try:
            pytrends = TrendReq(hl="en-US", tz=0)
            kw_list = ["bitcoin", "buy bitcoin"]
            pytrends.build_payload(kw_list, timeframe="all", geo="")
            df = pytrends.interest_over_time()
        except Exception as e:
            self.log.error("  Google Trends failed: %s", e)
            return

        if df.empty:
            self.log.warning("  No Google Trends data returned")
            return

        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        self._save_csv(df, "sentiment_google_trends.csv", "Google Trends",
                       sort_by="date", dedup_col="date")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_fear_greed()
        self._download_coingecko_market()
        self._download_google_trends()

    def download_incremental(self):
        # FNG + CoinGecko: re-fetch all (cheap, single request)
        # Google Trends: re-fetch all (no incremental support)
        self.download_all()


if __name__ == "__main__":
    SentimentDownloader.main()
