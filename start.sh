#!/usr/bin/env bash
set -e
# Regenerate the source data (fast, ~10s). Models, thresholds, reference
# and all evaluation results are committed, so nothing heavy runs here.
if [ ! -f data/events.csv ]; then
  echo "generating dataset (seed 42)..."
  python -m src.generate_data --seed 42
fi
exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}
