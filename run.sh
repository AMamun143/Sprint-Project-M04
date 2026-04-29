#!/usr/bin/env bash
set -euo pipefail

mkdir -p figs

# Rebuild the final semantic-map figure from scratch with a local venv.
python3 -m venv .venv_m04
. .venv_m04/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas matplotlib numpy scikit-learn
python scripts/generate_plot.py

echo "Done. Figure generated at figs/scatter.png"
