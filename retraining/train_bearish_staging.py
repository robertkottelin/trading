"""Bearish model training shim — redirects output to models/bearish_staging/.

Imports train_bearish, overrides MODEL_DIR to the staging directory, runs main().
The bearish trainer already writes production_config_bearish.json so no rename needed.

Usage: python retraining/train_bearish_staging.py
"""

import logging
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path (required when run from retraining/ subdir)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [train_bearish_staging] %(levelname)s: %(message)s")

STAGING_DIR = "models/bearish_staging"

# Clean out any previous incomplete staging run
if Path(STAGING_DIR).exists():
    log.info("Removing stale staging dir: %s", STAGING_DIR)
    shutil.rmtree(STAGING_DIR)
os.makedirs(STAGING_DIR, exist_ok=True)

# Override module-level constants BEFORE running
import model_training.train_bearish as trainer  # noqa: E402
trainer.MODEL_DIR = STAGING_DIR
trainer.LOG_FILE = os.path.join("model_training", "logs", "train_bearish_staging.log")

log.info("Starting bearish model training → %s", STAGING_DIR)
trainer.main()

# Verify output
config_path = Path(STAGING_DIR) / "production_config_bearish.json"
if config_path.exists():
    log.info("Bearish staging training complete: %s", STAGING_DIR)
else:
    log.warning("production_config_bearish.json not found — check training output")
