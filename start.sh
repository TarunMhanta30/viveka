#!/usr/bin/env bash
set -e
# Everything is precomputed and committed. Just start the server.
exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}
