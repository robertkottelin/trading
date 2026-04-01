#!/usr/bin/env bash
# Daily ML retraining script — download data, rebuild features, retrain models.
#
# Usage:
#   bash scripts/daily_retrain.sh            # run interactively
#   0 4 * * * cd /home/compute/trading && bash scripts/daily_retrain.sh   # cron (4 AM UTC)
#
# Logs are written to logs/retrain_YYYY-MM-DD.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATE=$(date -u +%Y-%m-%d)
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/retrain_${DATE}.log"

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG_FILE"
}

log "=== Daily retrain started ==="

# Step 1: Incremental data download
log "Step 1/3: Downloading raw data (incremental)..."
if python -m downloaders.download_all >> "$LOG_FILE" 2>&1; then
    log "Step 1/3: Download complete"
else
    log "Step 1/3: Download had errors (continuing anyway)"
fi

# Step 2: Rebuild feature dataset
log "Step 2/3: Building feature dataset..."
if python build_dataset.py >> "$LOG_FILE" 2>&1; then
    log "Step 2/3: Dataset build complete"
else
    log "Step 2/3: Dataset build FAILED"
    exit 1
fi

# Step 3: Retrain ML models
log "Step 3/3: Training models..."
if python -m model_training.train_model_v23 >> "$LOG_FILE" 2>&1; then
    log "Step 3/3: Training complete"
else
    log "Step 3/3: Training FAILED"
    exit 1
fi

log "=== Daily retrain finished ==="
