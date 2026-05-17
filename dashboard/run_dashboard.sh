#!/usr/bin/env bash
# Launch the local trading dashboard.
#
# Usage:
#   bash dashboard/run_dashboard.sh                       # loopback only (default)
#   PORT=8502 bash dashboard/run_dashboard.sh             # custom port
#   ADDR=192.168.1.42 bash dashboard/run_dashboard.sh     # bind to LAN IP
#   ADDR=0.0.0.0 bash dashboard/run_dashboard.sh          # all interfaces (CAUTION)
#
# Default ADDR=127.0.0.1 binds to loopback only — never reachable from outside
# this machine. Override ADDR only on a trusted network: the dashboard has NO
# authentication and exposes start/stop/pause/config-edit controls. Anyone who
# can reach the port can switch you from paper→live, testnet→mainnet, etc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-8501}"
ADDR="${ADDR:-127.0.0.1}"

if [[ "$ADDR" != "127.0.0.1" && "$ADDR" != "localhost" ]]; then
    echo "⚠️  Binding to $ADDR — dashboard has NO authentication." >&2
    echo "    Anyone on this network who can reach $ADDR:$PORT gets full bot control." >&2
fi

cd "$REPO_ROOT"
exec streamlit run dashboard/app.py \
    --server.port "$PORT" \
    --server.address "$ADDR" \
    --server.headless true \
    --browser.gatherUsageStats false
