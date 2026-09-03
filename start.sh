#!/usr/bin/env bash
set -e
# data/ is gitignored, so the server regenerates it from the fixed seed.
# Identical output every time — same seed, same data.
if [ ! -f data/events.csv ]; then
  echo "generating dataset (seed 42)..."
  python -m src.generate_data --seed 42
fi
if [ ! -f models/model.pkl ]; then
  echo "training model..."
  python -m src.model --seed 42
  python -m eval.build_reference
  python -m eval.cost_model
  python -m eval.evaluate
  python -m eval.ablation
  python -m eval.redteam
fi
exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
