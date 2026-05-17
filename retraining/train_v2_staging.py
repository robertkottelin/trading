"""Bullish model training shim — redirects output to models/v23_staging/.

Imports train_v2_all, overrides MODEL_DIR to the staging directory,
runs main(), then renames production_config.json → production_config_v23.json
so that signal_generator.py can find it after atomic deployment.

Usage: python retraining/train_v2_staging.py
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
                    format="%(asctime)s [train_v2_staging] %(levelname)s: %(message)s")

STAGING_DIR = "models/v23_staging"

# Clean out any previous incomplete staging run
if Path(STAGING_DIR).exists():
    log.info("Removing stale staging dir: %s", STAGING_DIR)
    shutil.rmtree(STAGING_DIR)
os.makedirs(STAGING_DIR, exist_ok=True)

# Override module-level constants BEFORE importing to avoid side effects
import model_training.train_v2_all as trainer  # noqa: E402
trainer.MODEL_DIR = STAGING_DIR
trainer.LOG_FILE = os.path.join("model_training", "logs", "v2_all_staging.log")
trainer.FINDINGS_FILE = os.path.join("model_training", "findings_staging.md")
trainer.VERSION = "v23_staging"

log.info("Starting bullish model training → %s", STAGING_DIR)
trainer.main()

# Rename config so signal_generator.py can find it after deployment
config_src = Path(STAGING_DIR) / "production_config.json"
config_dst = Path(STAGING_DIR) / "production_config_v23.json"
if config_src.exists() and not config_dst.exists():
    config_src.rename(config_dst)
    log.info("Renamed production_config.json → production_config_v23.json")
elif config_dst.exists():
    log.info("production_config_v23.json already present")
else:
    log.warning("production_config.json not found in staging dir — check training output")

log.info("Bullish staging training complete: %s", STAGING_DIR)
