# Trading Dashboard

Local Streamlit control plane for the dYdX trading bot.

## Start

```bash
bash dashboard/run_dashboard.sh             # default port 8501
PORT=8502 bash dashboard/run_dashboard.sh   # custom port
```

Open <http://127.0.0.1:8501> in a browser. The dashboard binds to `127.0.0.1`
only and is never reachable from outside the machine.

## Pages

| # | Page | What it does |
|---|---|---|
| Home | Overview | Top-line state: heartbeat, equity, latest decision, retrain status |
| 2 | Bot Control | Start / Stop / Pause / Resume / Restart with CLI flag picker |
| 3 | Logs | Live tail of `logs/pipeline_*.log` with level + regex filter |
| 4 | Pipeline | 7-stage DAG with per-stage timing, next-run countdown |
| 5 | Data Explorer | Browse CSVs / Parquet under `market_context_data/`, `processed_data/`, `raw_data/` with auto-detected timestamp plotting |
| 6 | ML Models | Production config metadata, 69+ model weights bar chart, firing history, live inference button |
| 7 | Strategies | 18 conventional strategies' live LONG/SHORT/INACTIVE direction + confidence + explanation |
| 8 | Positions | Equity curve + drawdown, win-rate, P&L by direction, trades ledger |
| 9 | Backtests | Trigger `python -m strategies.backtest_v2 ...` as a background job with live log tail |
| 10 | Retrain | View `state_data/retrain_state.json`; trigger `retrain_manager.run_retrain_chain` |
| 11 | Config | Edit `config/*.yaml` with diff preview and timestamped backups |

## Architecture

- The dashboard is a **separate process** from the bot.
- It **reads** existing artifacts (`state_data/*.jsonl`, `llm_agent/decision*.json`, `logs/`, `market_context_data/`, `models/*/production_config_*.json`).
- It **writes** only `state_data/.bot_pid.json` (PID tracking), `state_data/.pause_flag.json` (pause signal), and `config/*.yaml` + `config/.backups/*` on user-initiated edits.
- It **spawns** subprocesses for the bot (`run_pipeline.py --loop`), backtests (`strategies.backtest_v2`), and retraining (`retraining.retrain_manager.run_retrain_chain`).

## Bot control mechanics

- **Start**: `subprocess.Popen(start_new_session=True)` so the bot has its own process group. PID + command + start time persisted to `state_data/.bot_pid.json`. stdout/stderr piped to `logs/dashboard_bot_<UTC>.log`.
- **Stop**: SIGTERM to the process group; SIGKILL after a 10 s grace period. PID file cleared.
- **Pause**: writes `state_data/.pause_flag.json`. The reasoning agent reads this at the very top of every cycle and, if present, skips Stages 1–7. **Stage 0 (position monitoring, orphan cleanup, trailing stops) still runs** — open positions stay protected.
- **Resume**: deletes the pause flag.
- **Restart**: stop + sleep 0.5 s + start.
- **Stale-PID protection**: `psutil` is used to verify the saved PID is alive AND its command line contains `run_pipeline.py`. A PID that has been reused by an unrelated process is treated as `stopped` and not signalled.

## Config editing

- YAML is parsed (`yaml.safe_load`) before any write. Invalid YAML is rejected.
- Every successful write creates a timestamped backup under `config/.backups/<name>.YYYYMMDD_HHMMSS.bak`.
- The running bot picks up the new config on its **next cycle** (no live signal needed — `run_pipeline.py` and the strategies re-read YAML every cycle).

## Running tests

```bash
pytest dashboard/tests/ -v
```

All 62 tests (state loaders, log tail, pipeline parser, data loaders, bot supervisor with real subprocesses, config editor, job runner) should pass.

## Troubleshooting

- **`Port 8501 is not available`** — set `PORT=8599 bash dashboard/run_dashboard.sh`.
- **PID file stuck** — go to Bot Control → "Clear stale PID file" or delete `state_data/.bot_pid.json` by hand.
- **Pause flag stuck** — go to Bot Control → "Resume". To force-clear, delete `state_data/.pause_flag.json`.
- **`.cmd` or `signal_generator` import fails on ML Models live inference** — Live inference loads all .pkl files (~200 MB). Make sure `models/v23/` is populated (run `retraining.retrain_manager.run_retrain_chain` once or use the dashboard's Retrain page).
- **Auto-refresh feels slow** — turn off in the page sidebar, or raise the interval slider.

## Dependencies

Added by this dashboard (see `requirements.txt`):

```
streamlit>=1.32
plotly>=5.18
pyarrow>=14
streamlit-autorefresh>=1.0
streamlit-ace>=0.1.1
psutil>=5.9
pytest>=7.4
```

## Files

```
dashboard/
├── app.py                          entry: page config + sidebar status
├── run_dashboard.sh                launcher (binds to 127.0.0.1)
├── README.md                       this file
├── pages/                          11 Streamlit pages
└── lib/
    ├── state.py                    JSONL/JSON loaders + arrow_safe()
    ├── log_tail.py                 reverse-scan log tailer + filters
    ├── pipeline_parser.py          log → stage status extractor
    ├── data_loaders.py             CSV/Parquet/JSON readers
    ├── ml_inspector.py             production_config_*.json parser
    ├── strategy_runner.py          wraps strategies.engine.StrategyEngine
    ├── bot_supervisor.py           start/stop/pause/resume/status
    ├── job_runner.py               background subprocess + log capture
    ├── config_editor.py            safe YAML read/diff/backup/save
    └── plots.py                    reusable Plotly figures
```
