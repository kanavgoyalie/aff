#!/usr/bin/env bash
# Reproduce every number, table and figure in the manuscript (about 15-25 minutes on a laptop CPU).
set -euo pipefail
pip install -r requirements.txt
python download_data.py
python speeches_yields_capstone.py --speeches data/fed_speeches.csv --yields data/DGS10.csv --out outputs | tee outputs/run_log.txt
