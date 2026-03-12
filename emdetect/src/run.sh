#!/usr/bin/env bash
# Start the EmDetect API server.
# Usage: bash run.sh [port]
#
# Set API key via env var before running:
#   export EMDETECT_API_KEY=your-secret-key
#   bash run.sh 8000

PORT=${1:-8000}
cd "$(dirname "$0")"
uvicorn server:app --host 0.0.0.0 --port "$PORT" --reload
