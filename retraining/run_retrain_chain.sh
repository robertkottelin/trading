#!/bin/bash
# Full retraining chain: dataset rebuild → bullish staging → bearish staging → strategy tuning → signal ready
# Called by retrain_manager.py. Runs sequentially; exits immediately on any failure.
set -e

cd "$(dirname "$(dirname "$(realpath "$0")")")"   # cd to project root

LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"

# On any failure, mark state as failed
_fail_handler() {
    EXIT_CODE=$?
    FAIL_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"
    echo "$FAIL_PREFIX  RETRAIN CHAIN FAILED (exit=$EXIT_CODE)"
    python - <<PYEOF
import json, time
from pathlib import Path
state_path = Path("state_data/retrain_state.json")
try:
    state = json.loads(state_path.read_text())
except Exception:
    state = {}
state["status"] = "failed"
state["failure_reason"] = "chain exited with code $EXIT_CODE"
state["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
state_path.write_text(json.dumps(state, indent=2))
PYEOF
}
trap '_fail_handler' ERR
echo ""
echo "========================================================================"
echo "$LOG_PREFIX  RETRAIN CHAIN START"
echo "========================================================================"

# Step 1: Rebuild training dataset
echo "$LOG_PREFIX  Step 1/4: Rebuilding training dataset..."
python build_dataset.py
echo "$LOG_PREFIX  Step 1/4: Dataset build complete."

# Step 2: Train bullish models (staging)
echo "$LOG_PREFIX  Step 2/4: Training bullish models → models/v23_staging/ ..."
python retraining/train_v2_staging.py
echo "$LOG_PREFIX  Step 2/4: Bullish models trained."

# Step 3: Train bearish models (staging)
echo "$LOG_PREFIX  Step 3/4: Training bearish models → models/bearish_staging/ ..."
python retraining/train_bearish_staging.py
echo "$LOG_PREFIX  Step 3/4: Bearish models trained."

# Step 4: Tune strategy parameters
echo "$LOG_PREFIX  Step 4/4: Tuning strategy parameters → config/strategy_params_staging.yaml ..."
python retraining/strategy_tuner.py
echo "$LOG_PREFIX  Step 4/4: Strategy tuning complete."

# Signal completion — update retrain_state.json to 'ready_to_deploy'
python - <<'PYEOF'
import json, time
from pathlib import Path

state_path = Path("state_data/retrain_state.json")
try:
    state = json.loads(state_path.read_text())
except Exception:
    state = {}

state["status"] = "ready_to_deploy"
state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
state_path.write_text(json.dumps(state, indent=2))
print(f"State updated: ready_to_deploy")
PYEOF

echo ""
echo "========================================================================"
echo "$LOG_PREFIX  RETRAIN CHAIN COMPLETE — ready_to_deploy"
echo "========================================================================"
