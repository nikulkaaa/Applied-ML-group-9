#!/usr/bin/env bash
set -euo pipefail

# Start Uvicorn server in background with logs streamed
conda run -n predict_env --no-capture-output \
    uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# Forward termination signals to Uvicorn for graceful shutdown
trap 'echo "Stopping Uvicorn..."; kill $UVICORN_PID' SIGINT SIGTERM

# Wait for Uvicorn to exit, keeping container alive
wait $UVICORN_PID
