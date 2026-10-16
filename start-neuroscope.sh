#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
printf '\nNeuroScope local app: http://127.0.0.1:8000\nKeep this terminal open. Press Ctrl+C to stop.\n\n'
exec .venv/bin/python -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
