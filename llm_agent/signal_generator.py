"""ML Model Inference — load trained models and generate signals on latest data.

Loads bullish models from models/v23/ and (if available) bearish models from
models/bearish/. Both configs use the same feature matrix built from market_context_data.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.ta_core import compute_ta_features
from features.cross_exchange import build_cross_exchange_features
from features.funding import build_funding_features
from features.open_interest import build_open_interest_features
from features.positioning import build_positioning_features
from features.volatility_implied import build_implied_vol_features
from features.macro import build_macro_features
from features.sentiment import build_sentiment_features
from features.onchain import build_onchain_features
from features.defi import build_defi_features
from features.coinalyze import build_coinalyze_features
from features.dydx_trades import build_dydx_trades_features
from features.liquidations import build_liquidation_features

log = logging.getLogger(__name__)

MODELS_DIR = Path("models/v23")
CONFIG_FILE = MODELS_DIR / "production_config_v23.json"
BEARISH_MODELS_DIR = Path("models/bearish")
BEARISH_CONFIG_FILE = BEARISH_MODELS_DIR / "production_config_bearish.json"
CONTEXT_DIR = Path("market_context_data")

# Need enough history for the longest rolling window (288 candles = 24h)
MIN_CANDLES = 350


def _load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def _load_bearish_config() -> dict | None:
    """Load bearish model config if it exists. Returns None if not yet trained."""
    if not BEARISH_CONFIG_FILE.exists():
        return None
    try:
        with open(BEARISH_CONFIG_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not load bearish config: %s", e)
        return None


def _load_klines(filename: str, ts_col: str = "open_time_ms") -> pd.DataFrame:
    """Load a klines CSV from market_context_data/ with standard preprocessing."""
    path = CONTEXT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)

    # Normalize timestamp to open_time_ms (int64 milliseconds)
    if ts_col == "timestamp" and "timestamp" in df.columns:
        # dYdX uses ISO strings like "2026-02-24T12:40:00.000Z"
        df["open_time_ms"] = (
            pd.to_datetime(df["timestamp"], utc=True)
            .astype("int64") // 10**6
        )
    elif ts_col not in df.columns and "open_time_ms" not in df.columns:
        raise KeyError(f"No timestamp column found in {filename}")

    # Standard float columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("open_time_ms").drop_duplicates(subset="open_time_ms", keep="last")
    return df


def _build_features() -> pd.DataFrame:
    """Build the full feature DataFrame from market context klines.

    Returns a DataFrame indexed by open_time_ms with all features needed
    by any of the 25 models.
    """
    # Load dYdX candles as master grid (same role as in build_dataset.py)
    dydx = _load_klines("dydx_candles_5m.csv", ts_col="timestamp")
    dydx_ohlcv = dydx[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()
    if "trades" in dydx.columns:
        dydx_ohlcv["trades"] = pd.to_numeric(dydx["trades"], errors="coerce")
    if "usd_volume" in dydx.columns:
        dydx_ohlcv["quote_volume"] = pd.to_numeric(dydx["usd_volume"], errors="coerce")

    if len(dydx_ohlcv) < MIN_CANDLES:
        log.error("Only %d dYdX candles (need %d for reliable features) — "
                  "skipping ML inference this cycle",
                  len(dydx_ohlcv), MIN_CANDLES)
        return pd.DataFrame()

    # Compute dYdX TA features (no prefix — these are the primary features)
    log.info("Computing dYdX TA features on %d candles...", len(dydx_ohlcv))
    dydx_feats = compute_ta_features(dydx_ohlcv, prefix="")

    # Load Binance futures for prefixed features
    try:
        bnc = _load_klines("binance_futures_klines_5m.csv")
        bnc_ohlcv = bnc[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()
        for col in ["taker_buy_volume", "quote_volume", "trades"]:
            if col in bnc.columns:
                bnc_ohlcv[col] = pd.to_numeric(bnc[col], errors="coerce")
        log.info("Computing Binance TA features on %d candles...", len(bnc_ohlcv))
        bnc_feats = compute_ta_features(bnc_ohlcv, prefix="bnc_")
    except Exception as e:
        log.warning("Binance TA features failed: %s", e)
        bnc_feats = None

    # Start with dYdX features as base
    result = dydx_feats.copy()

    # Merge Binance features
    if bnc_feats is not None:
        result = result.merge(bnc_feats, on="open_time_ms", how="left")

    # Supplementary features — all builders use alignment.DATA_DIR to find CSVs.
    # Temporarily point it at market_context_data/ instead of raw_data/.
    import features.alignment as alignment_mod
    orig_data_dir = alignment_mod.DATA_DIR
    try:
        alignment_mod.DATA_DIR = str(CONTEXT_DIR)

        spot = _load_klines("binance_spot_klines_5m.csv")
        spot_close = spot[["open_time_ms", "close"]].copy()
        grid = result[["open_time_ms"]].copy()

        builders = [
            ("Cross-exchange", lambda: build_cross_exchange_features(grid, spot_close)),
            ("Funding",        lambda: build_funding_features(grid)),
            ("Open Interest",  lambda: build_open_interest_features(grid)),
            ("Positioning",    lambda: build_positioning_features(grid)),
            ("Implied Vol",    lambda: build_implied_vol_features(grid, dydx_ohlcv)),
            ("Macro",          lambda: build_macro_features(grid, spot_close)),
            ("Sentiment",      lambda: build_sentiment_features(grid)),
            ("On-chain",       lambda: build_onchain_features(grid)),
            ("DeFi",           lambda: build_defi_features(grid)),
            ("Coinalyze",      lambda: build_coinalyze_features(grid)),
            ("dYdX Trades",    lambda: build_dydx_trades_features(grid, spot_close)),
            ("Liquidations",   lambda: build_liquidation_features(grid)),
        ]

        for name, builder in builders:
            try:
                feat_df = builder()
                new_cols = [c for c in feat_df.columns if c != "open_time_ms"]
                if new_cols:
                    result = result.merge(feat_df, on="open_time_ms", how="left")
                log.info("  %-20s +%d features", name, len(new_cols))
            except FileNotFoundError as e:
                log.warning("  %-20s SKIPPED (missing: %s)", name, os.path.basename(str(e)))
            except Exception as e:
                log.warning("  %-20s FAILED: %s", name, e)
    finally:
        alignment_mod.DATA_DIR = orig_data_dir

    return result


def generate_signals() -> dict:
    """Run inference on all bullish + bearish models and return structured signals.

    Returns:
        {
            "signals": {model_name: {prob, threshold, signal, ...}, ...},
            "consensus": {bullish_count, bearish_count, neutral_count, weighted_score},
            "timestamp_ms": int,
            "text_summary": str  # formatted for LLM prompt
        }
    """
    config = _load_config()
    models_list = config["models"]
    model_weights = config.get("model_weights", {})

    # Load bearish models if available (graceful degradation)
    bearish_config = _load_bearish_config()
    bearish_models_list = bearish_config["models"] if bearish_config else []
    if bearish_models_list:
        log.info("Bearish models available: %d", len(bearish_models_list))

    # Build feature matrix (shared by all models)
    features_df = _build_features()
    if features_df.empty:
        return {"signals": {}, "consensus": {}, "timestamp_ms": 0,
                "text_summary": "ML SIGNALS: Feature computation failed — no data."}

    latest_ts = int(features_df["open_time_ms"].iloc[-1])
    log.info("Latest candle timestamp: %d", latest_ts)

    # Run bullish models
    signals = {}
    bullish = []
    bearish = []
    neutral = []

    for model_def in models_list:
        name = model_def["name"]
        try:
            result = _run_single_model(model_def, features_df, model_weights)
            signals[name] = result
            if result["signal"] == "BULLISH":
                bullish.append(result)
            elif result["signal"] == "BEARISH":
                bearish.append(result)
            else:
                neutral.append(result)
        except Exception as e:
            log.warning("Model %s failed: %s", name, e)
            signals[name] = {"error": str(e), "signal": "ERROR"}

    # Run bearish models (from models/bearish/ if trained)
    for model_def in bearish_models_list:
        name = model_def["name"]
        try:
            result = _run_single_model(
                model_def, features_df, {},
                models_dir=BEARISH_MODELS_DIR,
                force_direction="BEARISH",
            )
            signals[name] = result
            if result["signal"] == "BEARISH":
                bearish.append(result)
            else:
                neutral.append(result)
        except Exception as e:
            log.warning("Bearish model %s failed: %s", name, e)
            signals[name] = {"error": str(e), "signal": "ERROR"}

    # Compute weighted consensus score:
    # bullish models push score positive, bearish models push it negative
    bull_weighted = 0.0
    bear_weighted = 0.0
    weight_total = 0.0
    for s in signals.values():
        if "error" in s:
            continue
        w = s.get("quality_weight", 1.0)
        weight_total += w
        if s["signal"] == "BULLISH":
            bull_weighted += w * s["prob"]
        elif s["signal"] == "BEARISH":
            bear_weighted += w * s["prob"]

    if weight_total > 0:
        weighted_score = (bull_weighted - bear_weighted) / weight_total
    else:
        weighted_score = 0.0

    total_models = len(models_list) + len(bearish_models_list)
    consensus = {
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "total": total_models,
        "weighted_score": round(weighted_score, 4),
    }

    text = _format_signals_text(signals, consensus)

    return {
        "signals": signals,
        "consensus": consensus,
        "timestamp_ms": latest_ts,
        "text_summary": text,
    }


def _run_single_model(model_def: dict, features_df: pd.DataFrame,
                       model_weights: dict,
                       models_dir: Path | None = None,
                       force_direction: str | None = None) -> dict:
    """Run a single model on the latest row of features_df.

    Args:
        model_def: Model config dict with name, lgb_file, features, etc.
        features_df: Full feature matrix (latest row used for inference).
        model_weights: Optional quality-weight overrides (bullish models only).
        models_dir: Directory to load model files from. Defaults to MODELS_DIR.
        force_direction: If set ("BEARISH"), override the signal direction when firing.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    name = model_def["name"]
    lgb_path = models_dir / model_def["lgb_file"]
    feature_list = model_def["features"]
    prob_threshold = model_def["prob_threshold"]
    horizon = model_def["horizon"]
    target = model_def["target"]
    use_ensemble = model_def.get("use_ensemble", False)
    lgb_weight = model_def.get("lgb_weight", 1.0)
    quality_weight = model_weights.get(name, model_def.get("quality_weight", 1.0))

    # Check which features are available
    available = [f for f in feature_list if f in features_df.columns]
    missing = [f for f in feature_list if f not in features_df.columns]
    if missing:
        log.debug("Model %s missing %d/%d features: %s",
                  name, len(missing), len(feature_list), missing[:5])

    if len(available) < len(feature_list) * 0.5:
        raise ValueError(f"Too many missing features ({len(missing)}/{len(feature_list)})")

    # Extract latest row feature vector
    X = features_df[available].iloc[[-1]].copy()

    # Fill missing features with NaN (LightGBM handles NaN natively)
    for f in missing:
        X[f] = np.nan
    X = X[feature_list]  # reorder to match training order

    # Reject if too many features are NaN (likely insufficient warmup)
    nan_count = int(X.iloc[0].isna().sum())
    if nan_count > len(feature_list) * 0.3:
        raise ValueError(
            f"Latest row has {nan_count}/{len(feature_list)} NaN features "
            f"({nan_count / len(feature_list):.0%}) — likely insufficient warmup")

    # Load and predict — LightGBM
    lgb_model = joblib.load(lgb_path)
    lgb_prob = float(lgb_model.predict_proba(X)[:, 1][0])

    # CatBoost ensemble if applicable
    final_prob = lgb_prob
    if use_ensemble and model_def.get("cb_file"):
        cb_path = models_dir / model_def["cb_file"]
        if cb_path.exists():
            cb_model = joblib.load(cb_path)
            cb_prob = float(cb_model.predict_proba(X)[:, 1][0])
            final_prob = lgb_weight * lgb_prob + (1 - lgb_weight) * cb_prob
        else:
            log.warning("CatBoost file missing for %s: %s", name, cb_path)

    # Determine signal
    firing = final_prob >= prob_threshold

    # Parse target info for human-readable description
    # target format: target_up_12_0002 → horizon=12 candles (60min), threshold=0.2%
    #                target_down_12_0003 → horizon=12 candles, threshold=0.3% (bearish)
    # Threshold label is str(float).replace(".", ""):
    #   0.002→"0002", 0.003→"0003", 0.005→"0005", 0.01→"001"
    horizon_minutes = horizon * 5
    parts = target.replace("target_", "").split("_")
    tgt_direction = parts[0]  # "up", "fav", or "down"
    thresh_str = parts[-1]
    try:
        threshold_decimal = float(thresh_str[0] + "." + thresh_str[1:])
    except (ValueError, IndexError):
        threshold_decimal = 0.0
    threshold_pct = threshold_decimal * 100

    # Signal strength classification
    if not firing:
        strength = "NOT_FIRING"
        signal = "NEUTRAL"
    else:
        excess = final_prob - prob_threshold
        if excess > 0.3:
            strength = "STRONG"
        elif excess > 0.15:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        # Bearish (down_*) targets or forced-bearish direction → BEARISH signal
        if force_direction == "BEARISH" or tgt_direction == "down":
            signal = "BEARISH"
        else:
            signal = "BULLISH"

    return {
        "prob": round(final_prob, 4),
        "threshold": prob_threshold,
        "signal": signal,
        "strength": strength,
        "target": target,
        "direction": tgt_direction,
        "horizon_candles": horizon,
        "horizon_minutes": horizon_minutes,
        "threshold_pct": threshold_pct,
        "quality_weight": round(quality_weight, 2),
        "missing_features": len(missing),
    }


def _format_signals_text(signals: dict, consensus: dict) -> str:
    """Format signals into structured text for the LLM prompt."""
    lines = []
    total = consensus["total"]
    b_count = consensus["bullish_count"]
    bear_count = consensus["bearish_count"]
    lines.append(f"ML MODEL SIGNALS ({total} models, latest 5-min candle):")

    # Bullish signals (firing)
    bullish = [(k, v) for k, v in signals.items()
               if v.get("signal") == "BULLISH"]
    if bullish:
        lines.append(f"  BULLISH signals ({len(bullish)} firing):")
        for name, s in sorted(bullish, key=lambda x: -x[1]["prob"]):
            lines.append(
                f"    {name}: prob={s['prob']:.2f} (thresh={s['threshold']:.2f}) "
                f"| {s['horizon_minutes']}min +{s['threshold_pct']}% "
                f"| weight={s['quality_weight']:.2f} | {s['strength']}"
            )

    # Bearish signals (firing)
    bearish_firing = [(k, v) for k, v in signals.items()
                      if v.get("signal") == "BEARISH"]
    if bearish_firing:
        lines.append(f"  BEARISH signals ({len(bearish_firing)} firing — downside models):")
        for name, s in sorted(bearish_firing, key=lambda x: -x[1]["prob"]):
            lines.append(
                f"    {name}: prob={s['prob']:.2f} (thresh={s['threshold']:.2f}) "
                f"| {s['horizon_minutes']}min -{s['threshold_pct']}% "
                f"| weight={s['quality_weight']:.2f} | {s['strength']}"
            )

    # Neutral signals (not firing)
    neutrals = [(k, v) for k, v in signals.items()
                if v.get("signal") == "NEUTRAL"]
    if neutrals:
        lines.append(f"  NEUTRAL/not-firing ({len(neutrals)}):")
        for name, s in sorted(neutrals, key=lambda x: -x[1].get("prob", 0)):
            prob = s.get("prob", 0)
            thresh = s.get("threshold", 0)
            hm = s.get("horizon_minutes", 0)
            tp = s.get("threshold_pct", 0)
            w = s.get("quality_weight", 0)
            lines.append(
                f"    {name}: prob={prob:.2f} (thresh={thresh:.2f}) "
                f"| {hm}min | weight={w:.2f}"
            )

    # Errors
    errors = [(k, v) for k, v in signals.items() if v.get("signal") == "ERROR"]
    if errors:
        lines.append(f"  ERRORS ({len(errors)} models failed):")
        for name, s in errors:
            lines.append(f"    {name}: {s.get('error', 'unknown')}")

    ws = consensus["weighted_score"]
    lines.append(
        f"  Consensus: {b_count}/{total} bullish, "
        f"{bear_count}/{total} bearish, "
        f"net weighted score: {ws:+.4f} "
        f"(positive=bullish bias, negative=bearish bias)"
    )

    return "\n".join(lines)
