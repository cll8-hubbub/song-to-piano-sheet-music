#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Starting the piano sheet music app (localhost only, not visible to other devices on your network)..."
echo "Once you see 'Application startup complete', open: http://localhost:8000"
uvicorn app:app --host 127.0.0.1 --port 8000
