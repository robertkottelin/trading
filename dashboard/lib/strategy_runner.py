"""Strategy runner — wraps the production StrategyEngine for the dashboard."""

from __future__ import annotations

from typing import Any

from dashboard.lib import state


def run_strategies(data_dir: str | None = None) -> dict[str, Any]:
    """Instantiate the StrategyEngine and run all 18 strategies once.

    Args:
        data_dir: which data directory to load from. Defaults to
                  market_context_data/ (the live pipeline source).

    Returns:
        {
          "signals": {name: {"direction", "confidence", "explanation", "details"}},
          "consensus": {"long_count", "short_count", "inactive_count"},
          "text_summary": str,
          "error": str | None,
        }
    """
    from strategies.engine import StrategyEngine

    d = data_dir or str(state.MARKET_DATA_DIR)
    engine = StrategyEngine(data_dir=d)
    try:
        result = engine.generate_signals()
    except Exception as e:
        return {"signals": {}, "consensus": {}, "text_summary": "",
                "error": f"{type(e).__name__}: {e}"}

    out_signals = {}
    for name, sig in (result.get("signals") or {}).items():
        if hasattr(sig, "direction"):
            out_signals[name] = {
                "direction": sig.direction,
                "confidence": sig.confidence,
                "explanation": sig.explanation,
                "details": dict(sig.details or {}),
            }
        else:
            out_signals[name] = dict(sig) if isinstance(sig, dict) else {"raw": str(sig)}

    return {
        "signals": out_signals,
        "consensus": result.get("consensus", {}),
        "text_summary": result.get("text_summary", ""),
        "error": None,
    }


def reload_engine_params(engine) -> None:
    """Force reload of strategy_params.yaml into the running engine."""
    if hasattr(engine, "reload_params"):
        engine.reload_params()
