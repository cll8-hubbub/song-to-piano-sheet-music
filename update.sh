#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Pulling latest changes from GitHub..."
git pull
source .venv/bin/activate
echo "Installing any new or changed dependencies (skips anything already up to date)..."
uv pip install -r requirements.txt
echo ""
echo "Done. Run ./run.sh to start the updated app."
