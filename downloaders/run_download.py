r"""
Run the full data download and log output to a file that Claude can read.
Run this in a separate terminal:
    cd C:\Users\rober\repos\trading
    python run_download.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Redirect stdout/stderr to log file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(project_root, "downloaders", "logs", "download_full.log")
print(f"Logging to: {log_path}")
print("Starting full download of all 14 data sources...")
print("This will take 30-60 minutes. Check the log file for progress.")
print()

import logging

# Set up root logger to capture everything
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

# Now run the download
from downloaders.download_all import main
sys.argv = ["download_all", "--full"]
main()
