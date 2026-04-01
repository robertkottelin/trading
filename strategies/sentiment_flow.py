"""Strategy 5: Sentiment & Capital Flow Regime.

Thesis: Fear & Greed extremes are strong contrarian signals. Stablecoin supply
growth signals capital entering crypto. On-chain network health indicates
fundamental demand.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class SentimentFlow(BaseStrategy):

    name = "Sentiment & Capital Flow"
    description = "Contrarian sentiment + stablecoin inflows + on-chain health"
    data_files = [
        "sentiment_fear_greed.csv",
        "defi_stablecoin_history.csv",
        "defi_tvl.csv",
        "blockchain_onchain.csv",
    ]

    # Tunable parameters
    FNG_EXTREME_FEAR = 20
    FNG_FEAR = 30
    FNG_GREED = 70
    FNG_EXTREME_GREED = 80
    FNG_MOMENTUM_WINDOW = 14  # days
    STABLE_GROWTH_WINDOW = 7  # days
    STABLE_GROWTH_THRESHOLD = 0.005  # 0.5%
    STABLE_DECLINE_THRESHOLD = -0.003  # -0.3%
    ADDR_GROWTH_THRESHOLD = 0.03   # 3%
    TXVOL_GROWTH_THRESHOLD = 0.05  # 5%
    HASHRATE_GROWTH_WINDOW = 14
    HASHRATE_GROWTH_THRESHOLD = 0.02  # 2%
    TVL_GROWTH_WINDOW = 7
    TVL_GROWTH_THRESHOLD = 0.03  # 3%
    LONG_THRESHOLD = 1.5
    SHORT_THRESHOLD = -1.5

    def _build_daily_data(self, data: dict) -> pd.DataFrame:
        """Merge all sentiment and fundamental data into daily DataFrame."""
        # Fear & Greed
        fng = data.get("sentiment_fear_greed.csv", pd.DataFrame())
        if fng.empty or "fng_value" not in fng.columns:
            return pd.DataFrame()

        fng["date"] = pd.to_datetime(fng["date"]).dt.date
        fng["fng_value"] = pd.to_numeric(fng["fng_value"], errors="coerce")
        fng = fng[["date", "fng_value"]].drop_duplicates("date", keep="last")
        result = fng.copy()

        # Stablecoin supply
        stable = data.get("defi_stablecoin_history.csv", pd.DataFrame())
        if not stable.empty and "total_circulating_usd" in stable.columns:
            stable["date"] = pd.to_datetime(stable["date"]).dt.date
            stable["total_circulating_usd"] = pd.to_numeric(
                stable["total_circulating_usd"], errors="coerce")
            stable = stable[["date", "total_circulating_usd"]].drop_duplicates("date", keep="last")
            result = result.merge(stable, on="date", how="left")

        # DeFi TVL
        tvl = data.get("defi_tvl.csv", pd.DataFrame())
        if not tvl.empty and "tvl" in tvl.columns:
            tvl["date"] = pd.to_datetime(tvl["date"]).dt.date
            tvl["tvl"] = pd.to_numeric(tvl["tvl"], errors="coerce")
            tvl = tvl[["date", "tvl"]].drop_duplicates("date", keep="last")
            result = result.merge(tvl, on="date", how="left")

        # On-chain metrics
        bc = data.get("blockchain_onchain.csv", pd.DataFrame())
        if not bc.empty:
            bc["date"] = pd.to_datetime(bc["date"]).dt.date
            for col in ["n_unique_addresses", "estimated_transaction_volume_usd",
                         "hash_rate"]:
                if col in bc.columns:
                    bc[col] = pd.to_numeric(bc[col], errors="coerce")
            cols = ["date"] + [c for c in ["n_unique_addresses",
                     "estimated_transaction_volume_usd", "hash_rate"] if c in bc.columns]
            bc_out = bc[cols].drop_duplicates("date", keep="last")
            result = result.merge(bc_out, on="date", how="left")

        result = result.sort_values("date").reset_index(drop=True)
        result = result.ffill()
        return result

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        df = self._build_daily_data(data)
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        scores = pd.DataFrame(index=df.index)

        # Sub-signal 1: FNG contrarian
        fng = df["fng_value"]
        fng_score = np.zeros(len(df))
        for i in range(len(df)):
            v = fng.iloc[i]
            if pd.isna(v):
                continue
            if v < self.FNG_EXTREME_FEAR:
                fng_score[i] = 2.5
            elif v < self.FNG_FEAR:
                fng_score[i] = 1.0
            elif v > self.FNG_EXTREME_GREED:
                fng_score[i] = -2.5
            elif v > self.FNG_GREED:
                fng_score[i] = -1.0
        scores["fng_score"] = fng_score

        # FNG momentum
        fng_mom = fng.diff(self.FNG_MOMENTUM_WINDOW)
        fng_mom_score = np.zeros(len(df))
        for i in range(len(df)):
            v = fng.iloc[i]
            m = fng_mom.iloc[i]
            if pd.isna(v) or pd.isna(m):
                continue
            if v < 40 and m > 5:  # Rising from lows
                fng_mom_score[i] = 0.5
            elif v > 60 and m < -5:  # Falling from highs
                fng_mom_score[i] = -0.5
        scores["fng_momentum"] = fng_mom_score

        # Sub-signal 2: Stablecoin flow
        if "total_circulating_usd" in df.columns:
            stable_chg = df["total_circulating_usd"].pct_change(self.STABLE_GROWTH_WINDOW)
            scores["stable_score"] = np.where(
                stable_chg > self.STABLE_GROWTH_THRESHOLD, 1.0,
                np.where(stable_chg < self.STABLE_DECLINE_THRESHOLD, -1.0, 0)
            )
        else:
            scores["stable_score"] = 0

        # Sub-signal 3: On-chain health
        onchain_score = np.zeros(len(df))
        if "n_unique_addresses" in df.columns:
            addr_chg = df["n_unique_addresses"].pct_change(7)
            onchain_score += np.where(addr_chg > self.ADDR_GROWTH_THRESHOLD, 0.5, 0)
            onchain_score += np.where(addr_chg < -self.ADDR_GROWTH_THRESHOLD, -0.3, 0)

        if "estimated_transaction_volume_usd" in df.columns:
            txvol_chg = df["estimated_transaction_volume_usd"].pct_change(7)
            onchain_score += np.where(txvol_chg > self.TXVOL_GROWTH_THRESHOLD, 0.3, 0)

        if "hash_rate" in df.columns:
            hr_chg = df["hash_rate"].pct_change(self.HASHRATE_GROWTH_WINDOW)
            onchain_score += np.where(hr_chg > self.HASHRATE_GROWTH_THRESHOLD, 0.2, 0)

        scores["onchain_score"] = onchain_score

        # Sub-signal 4: DeFi TVL
        if "tvl" in df.columns:
            tvl_chg = df["tvl"].pct_change(self.TVL_GROWTH_WINDOW)
            scores["tvl_score"] = np.where(
                tvl_chg > self.TVL_GROWTH_THRESHOLD, 0.3,
                np.where(tvl_chg < -self.TVL_GROWTH_THRESHOLD, -0.3, 0)
            )
        else:
            scores["tvl_score"] = 0

        # Total score
        score_cols = [c for c in scores.columns]
        df["total_score"] = scores[score_cols].sum(axis=1)

        # Lag by 1 day
        df["score_lagged"] = df["total_score"].shift(1)

        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))

        for i in range(1, len(df)):
            s = df["score_lagged"].iloc[i]
            if pd.isna(s):
                continue
            if s > self.LONG_THRESHOLD:
                signal[i] = 1
                confidence[i] = min(0.55 + (s - self.LONG_THRESHOLD) * 0.06, 0.85)
            elif s < self.SHORT_THRESHOLD:
                signal[i] = -1
                confidence[i] = min(0.55 + abs(s - self.SHORT_THRESHOLD) * 0.06, 0.85)

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence

        return df[["date", "signal", "confidence", "total_score", "fng_value"]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient sentiment data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "fng": f"{last['fng_value']:.0f}",
            "score": f"{last['total_score']:+.2f}",
        }

        if direction == "LONG":
            expl = f"Bullish sentiment regime (FNG={last['fng_value']:.0f}, score={last['total_score']:+.1f})"
        elif direction == "SHORT":
            expl = f"Bearish sentiment regime (FNG={last['fng_value']:.0f}, score={last['total_score']:+.1f})"
        else:
            expl = f"Sentiment neutral (FNG={last['fng_value']:.0f}, score={last['total_score']:+.1f})"

        return StrategySignal(direction, conf, expl, details)
