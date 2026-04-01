"""Strategy 3: Macro Cross-Asset Risk Regime.

Thesis: BTC is a risk-on macro asset correlated with NASDAQ. DXY strength
crushes BTC. VIX spikes signal risk-off. Composite scoring of 7 macro
sub-signals determines the regime.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class MacroRegime(BaseStrategy):

    name = "Macro Risk Regime"
    description = "Trades BTC based on cross-asset risk-on/risk-off regime"
    data_files = [
        "macro_equities.csv",
        "macro_fx.csv",
        "macro_commodities.csv",
        "macro_rates.csv",
        "macro_credit.csv",
    ]

    # Tunable parameters
    LONG_THRESHOLD = 2.0
    SHORT_THRESHOLD = -2.0
    VIX_LOW = 18
    VIX_HIGH = 28
    RETURN_LOOKBACK = 5  # Use 5d returns instead of 1d
    SPX_WEIGHT = 1.0
    NDX_SPREAD_WEIGHT = 0.5
    DXY_WEIGHT = 1.0
    VIX_WEIGHT = 1.0
    GOLD_WEIGHT = 0.5
    YIELD_CURVE_WEIGHT = 0.3
    CREDIT_WEIGHT = 0.5

    def _build_daily_macro(self, data: dict) -> pd.DataFrame:
        """Merge all macro sources into a single daily DataFrame."""
        # Equities
        eq = data.get("macro_equities.csv", pd.DataFrame())
        fx = data.get("macro_fx.csv", pd.DataFrame())
        comm = data.get("macro_commodities.csv", pd.DataFrame())
        rates = data.get("macro_rates.csv", pd.DataFrame())
        credit = data.get("macro_credit.csv", pd.DataFrame())

        frames = {}

        if not eq.empty and "GSPC_close" in eq.columns:
            eq = eq[["date"]].copy()
            eq_full = data["macro_equities.csv"]
            eq["date"] = pd.to_datetime(eq_full["date"]).dt.date
            for col in ["GSPC_close", "IXIC_close"]:
                if col in eq_full.columns:
                    eq[col] = pd.to_numeric(eq_full[col], errors="coerce")
            eq = eq.drop_duplicates("date", keep="last")
            frames["eq"] = eq

        if not fx.empty:
            fx_out = pd.DataFrame()
            fx["date_parsed"] = pd.to_datetime(fx["date"]).dt.date
            for col in ["DXYNYB_close"]:
                if col in fx.columns:
                    fx[col] = pd.to_numeric(fx[col], errors="coerce")
            fx_out = fx[["date_parsed"] + [c for c in ["DXYNYB_close"] if c in fx.columns]].copy()
            fx_out = fx_out.rename(columns={"date_parsed": "date"}).drop_duplicates("date", keep="last")
            frames["fx"] = fx_out

        if not comm.empty:
            comm_out = pd.DataFrame()
            comm["date_parsed"] = pd.to_datetime(comm["date"]).dt.date
            cols = [c for c in ["GCF_close"] if c in comm.columns]
            for c in cols:
                comm[c] = pd.to_numeric(comm[c], errors="coerce")
            comm_out = comm[["date_parsed"] + cols].copy()
            comm_out = comm_out.rename(columns={"date_parsed": "date"}).drop_duplicates("date", keep="last")
            frames["comm"] = comm_out

        if not rates.empty:
            rates["date_parsed"] = pd.to_datetime(rates["date"]).dt.date
            rate_cols = [c for c in ["VIXCLS", "T10Y2Y"] if c in rates.columns]
            for c in rate_cols:
                rates[c] = pd.to_numeric(rates[c], errors="coerce")
            rates_out = rates[["date_parsed"] + rate_cols].copy()
            rates_out = rates_out.rename(columns={"date_parsed": "date"}).drop_duplicates("date", keep="last")
            frames["rates"] = rates_out

        if not credit.empty:
            credit["date_parsed"] = pd.to_datetime(credit["date"]).dt.date
            cred_cols = [c for c in ["BAMLH0A0HYM2"] if c in credit.columns]
            for c in cred_cols:
                credit[c] = pd.to_numeric(credit[c], errors="coerce")
            credit_out = credit[["date_parsed"] + cred_cols].copy()
            credit_out = credit_out.rename(columns={"date_parsed": "date"}).drop_duplicates("date", keep="last")
            frames["credit"] = credit_out

        if not frames:
            return pd.DataFrame()

        # Merge all on date
        merged = None
        for key, df in frames.items():
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged = merged.ffill()  # Forward-fill missing values
        return merged

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        df = self._build_daily_macro(data)
        if df.empty or len(df) < 10:
            return pd.DataFrame()

        # Compute daily returns and sub-signals
        scores = pd.DataFrame(index=df.index)

        # 1. SPX 5d return (less whipsaw than daily)
        lb = self.RETURN_LOOKBACK
        if "GSPC_close" in df.columns:
            df["spx_ret"] = df["GSPC_close"].pct_change(lb)
            scores["spx_score"] = np.where(df["spx_ret"] > 0.005, self.SPX_WEIGHT,
                                           np.where(df["spx_ret"] < -0.005, -self.SPX_WEIGHT, 0))
        else:
            scores["spx_score"] = 0

        # 2. NASDAQ-SPX spread (5d cumulative)
        if "GSPC_close" in df.columns and "IXIC_close" in df.columns:
            spx_5d = df["GSPC_close"].pct_change(lb)
            ndx_5d = df["IXIC_close"].pct_change(lb)
            spread = ndx_5d - spx_5d
            scores["ndx_spread_score"] = np.where(spread > 0.005, self.NDX_SPREAD_WEIGHT,
                                                   np.where(spread < -0.005, -self.NDX_SPREAD_WEIGHT, 0))
        else:
            scores["ndx_spread_score"] = 0

        # 3. DXY 5d return (inverted: weak dollar = bullish BTC)
        if "DXYNYB_close" in df.columns:
            df["dxy_ret"] = df["DXYNYB_close"].pct_change(lb)
            scores["dxy_score"] = np.where(df["dxy_ret"] < -0.002, self.DXY_WEIGHT,
                                           np.where(df["dxy_ret"] > 0.002, -self.DXY_WEIGHT, 0))
        else:
            scores["dxy_score"] = 0

        # 4. VIX level + direction
        if "VIXCLS" in df.columns:
            vix = df["VIXCLS"]
            vix_dir = vix.diff()
            vix_signal = np.zeros(len(df))
            for i in range(len(df)):
                v = vix.iloc[i]
                d = vix_dir.iloc[i]
                if pd.isna(v):
                    continue
                if v < self.VIX_LOW or (not pd.isna(d) and d < 0 and v < 22):
                    vix_signal[i] = self.VIX_WEIGHT  # Risk-on
                elif v > self.VIX_HIGH or (not pd.isna(d) and d > 0 and v > 20):
                    vix_signal[i] = -self.VIX_WEIGHT  # Risk-off
            scores["vix_score"] = vix_signal
        else:
            scores["vix_score"] = 0

        # 5. Gold return + DXY direction (debasement narrative)
        if "GCF_close" in df.columns:
            df["gold_ret"] = df["GCF_close"].pct_change()
            gold_up = df["gold_ret"] > 0
            dxy_down = df.get("dxy_ret", pd.Series(0, index=df.index)) < 0
            scores["gold_score"] = np.where(gold_up & dxy_down, self.GOLD_WEIGHT,
                                            np.where(~gold_up & ~dxy_down, -self.GOLD_WEIGHT * 0.5, 0))
        else:
            scores["gold_score"] = 0

        # 6. 10Y-2Y spread change (steepening = optimism)
        if "T10Y2Y" in df.columns:
            spread_chg = df["T10Y2Y"].diff()
            scores["yield_score"] = np.where(spread_chg > 0, self.YIELD_CURVE_WEIGHT,
                                             np.where(spread_chg < 0, -self.YIELD_CURVE_WEIGHT, 0))
        else:
            scores["yield_score"] = 0

        # 7. Credit spread change (tightening = risk-on)
        if "BAMLH0A0HYM2" in df.columns:
            credit_chg = df["BAMLH0A0HYM2"].diff()
            scores["credit_score"] = np.where(credit_chg < 0, self.CREDIT_WEIGHT,
                                              np.where(credit_chg > 0, -self.CREDIT_WEIGHT, 0))
        else:
            scores["credit_score"] = 0

        # Total score
        score_cols = [c for c in scores.columns if c.endswith("_score")]
        df["total_score"] = scores[score_cols].sum(axis=1)

        # Smooth with 5-day EMA to reduce noise
        df["score_ema"] = df["total_score"].ewm(span=5, min_periods=2).mean()

        # Generate signals (lag by 1 day for look-ahead bias prevention)
        df["score_lagged"] = df["score_ema"].shift(1)

        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))

        for i in range(1, len(df)):
            s = df["score_lagged"].iloc[i]
            if pd.isna(s):
                continue
            if s > self.LONG_THRESHOLD:
                signal[i] = 1
                confidence[i] = min(0.55 + (s - self.LONG_THRESHOLD) * 0.08, 0.85)
            elif s < self.SHORT_THRESHOLD:
                signal[i] = -1
                confidence[i] = min(0.55 + abs(s - self.SHORT_THRESHOLD) * 0.08, 0.85)

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence

        return df[["date", "signal", "confidence", "total_score", "score_ema"]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient macro data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "score": f"{last['total_score']:+.2f}",
            "score_ema": f"{last['score_ema']:+.2f}",
        }

        if direction == "LONG":
            expl = f"Risk-on macro environment (score={last['score_ema']:+.1f})"
        elif direction == "SHORT":
            expl = f"Risk-off macro environment (score={last['score_ema']:+.1f})"
        else:
            expl = f"Macro regime neutral (score={last['score_ema']:+.1f})"

        return StrategySignal(direction, conf, expl, details)
