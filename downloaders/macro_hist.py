"""
Macro data downloader — yfinance (equities, VIX, FX, commodities, crypto-adjacent)
+ FRED (rates, liquidity, credit spreads).

All daily granularity — forward-filled to 5m grid by feature engine later.
"""

import os
import pandas as pd
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader

# Try importing yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None


class MacroDownloader(BaseDownloader):
    name = "macro"

    # yfinance ticker groups
    YF_GROUPS = {
        "macro_equities.csv": {
            "description": "equity indices",
            "tickers": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^N225", "^GDAXI", "^HSI", "^FTSE"],
        },
        # VIX moved to FRED (VIXCLS) — yfinance ^VIX is broken

        "macro_fx.csv": {
            "description": "FX (DXY, EUR, JPY, CNY)",
            "tickers": ["DX-Y.NYB", "EURUSD=X", "USDJPY=X", "USDCNY=X"],
        },
        "macro_commodities.csv": {
            "description": "commodities (gold, silver, oil, copper)",
            "tickers": ["GC=F", "SI=F", "CL=F", "HG=F"],
        },
        "macro_crypto_adjacent.csv": {
            "description": "crypto-adjacent (ETH, BTC, ETFs)",
            "tickers": ["ETH-USD", "BTC-USD", "IBIT", "FBTC", "GBTC", "BITO"],
        },
    }

    # FRED series groups
    FRED_GROUPS = {
        "macro_rates.csv": {
            "description": "rates & yields",
            "series": ["DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M",
                        "DFF", "DFII10", "T5YIE", "T10YIE", "VIXCLS"],
        },
        "macro_liquidity.csv": {
            "description": "liquidity (M2, Fed balance sheet, RRP)",
            "series": ["M2SL", "WM2NS", "WALCL", "RRPONTSYD"],
        },
        "macro_credit.csv": {
            "description": "credit spreads & financial stress",
            "series": ["BAMLH0A0HYM2", "BAMLC0A4CBBB", "BAA10Y", "STLFSI4", "NFCI"],
        },
    }

    def __init__(self, full=False):
        super().__init__(full=full)
        self.fred_url = self.cfg.get("fred_base_url",
                                     "https://api.stlouisfed.org/fred/series/observations")
        self.fred_key = os.environ.get("FRED_API_KEY", "")
        self.start_date = self.cfg.get("start_date", "2015-01-01")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)

    # ---- yfinance ---------------------------------------------------------

    def _download_yf_group(self, filename, group_cfg):
        """Download a group of yfinance tickers into one CSV."""
        if yf is None:
            self.log.error("  yfinance not installed — skipping %s", filename)
            return

        tickers = group_cfg["tickers"]
        desc = group_cfg["description"]
        self.log.info("  Downloading %s via yfinance (%d tickers)...", desc, len(tickers))

        # Determine start date
        if not self.full:
            last_ts = self._get_last_timestamp(filename, "date")
            if last_ts:
                # Re-fetch last 30 days for revisions
                start = (pd.Timestamp(last_ts) - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                start = self.start_date
        else:
            start = self.start_date

        try:
            # Bulk download
            data = yf.download(tickers, start=start, group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            self.log.error("  Bulk yfinance download failed: %s — trying per-ticker", e)
            data = pd.DataFrame()

        # Build wide DataFrame with one column per ticker per field
        rows_dict = {}
        if len(tickers) == 1:
            # Single ticker: yfinance doesn't nest by ticker
            ticker = tickers[0]
            safe_name = ticker.replace("^", "").replace("=", "").replace("-", "").replace(".", "")
            if not data.empty:
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in data.columns:
                        rows_dict[f"{safe_name}_{col.lower()}"] = data[col]
        else:
            for ticker in tickers:
                safe_name = ticker.replace("^", "").replace("=", "").replace("-", "").replace(".", "")
                try:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker]
                        for col in ["Open", "High", "Low", "Close", "Volume"]:
                            if col in ticker_data.columns:
                                rows_dict[f"{safe_name}_{col.lower()}"] = ticker_data[col]
                except Exception:
                    # Fallback: download individually
                    try:
                        self.log.info("    Fallback: downloading %s individually", ticker)
                        single = yf.download(ticker, start=start, auto_adjust=True, progress=False)
                        for col in ["Open", "High", "Low", "Close", "Volume"]:
                            if col in single.columns:
                                rows_dict[f"{safe_name}_{col.lower()}"] = single[col]
                    except Exception as e2:
                        self.log.warning("    Failed to download %s: %s", ticker, e2)

        if not rows_dict:
            self.log.warning("  No data for %s", desc)
            return

        df = pd.DataFrame(rows_dict)
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        if self.full:
            self._save_csv(df, filename, desc, sort_by="date", dedup_col="date")
        else:
            self._append_csv(df, filename, sort_by="date", dedup_col="date")

    # ---- FRED -------------------------------------------------------------

    def _download_fred_series(self, series_id, start_date):
        """Download a single FRED series via REST API."""
        if not self.fred_key:
            return pd.Series(dtype=float, name=series_id)

        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
        try:
            resp = self._http_get(self.fred_url, params, delay=self.delay)
            data = resp.json()
        except Exception as e:
            self.log.warning("    FRED %s failed: %s", series_id, e)
            return pd.Series(dtype=float, name=series_id)

        obs = data.get("observations", [])
        if not obs:
            return pd.Series(dtype=float, name=series_id)

        dates = []
        values = []
        for o in obs:
            dates.append(o["date"])
            val = o["value"]
            values.append(float(val) if val != "." else None)

        return pd.Series(values, index=pd.Index(dates, name="date"), name=series_id)

    def _download_fred_group(self, filename, group_cfg):
        """Download a group of FRED series into one CSV."""
        if not self.fred_key:
            self.log.warning("  FRED_API_KEY not set — skipping %s", filename)
            return

        series_list = group_cfg["series"]
        desc = group_cfg["description"]
        self.log.info("  Downloading FRED %s (%d series)...", desc, len(series_list))

        start = self.start_date
        if not self.full:
            last_ts = self._get_last_timestamp(filename, "date")
            if last_ts:
                start = (pd.Timestamp(last_ts) - timedelta(days=30)).strftime("%Y-%m-%d")

        all_series = []
        for sid in series_list:
            s = self._download_fred_series(sid, start)
            if not s.empty:
                all_series.append(s)
            self.log.info("    %s: %d observations", sid, len(s))

        if not all_series:
            self.log.warning("  No FRED data for %s", desc)
            return

        df = pd.concat(all_series, axis=1)
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        if self.full:
            self._save_csv(df, filename, f"FRED {desc}", sort_by="date", dedup_col="date")
        else:
            self._append_csv(df, filename, sort_by="date", dedup_col="date")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        for filename, cfg in self.YF_GROUPS.items():
            self._download_yf_group(filename, cfg)
        for filename, cfg in self.FRED_GROUPS.items():
            self._download_fred_group(filename, cfg)

    def download_incremental(self):
        # Same as full but start dates adjusted internally
        self.download_all()


if __name__ == "__main__":
    MacroDownloader.main()
