#!/usr/bin/env bash
set -euo pipefail

# Stop any zombie servers
pkill -f "uvicorn app.main:app" >/dev/null 2>&1 || true

# Clean Python caches (fast)
find app -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find app -name "*.pyc" -delete 2>/dev/null || true

# Start uvicorn watching ONLY your source tree, excluding everything noisy
uvicorn app.main:app \
  --reload \
  --reload-dir app \
  --reload-include "app/*.py" \
  --reload-include "app/**/*.py" \
  --reload-exclude ".venv/*" \
  --reload-exclude "**/site-packages/*" \
  --reload-exclude "**/__pycache__/*" \
  --reload-exclude "*.pyc"
