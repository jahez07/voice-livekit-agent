#!/bin/bash
set -e

echo "Running service initialization..."
uv run python init_services.py

echo "Starting agent..."
exec uv run python src/agent.py start