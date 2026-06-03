#!/usr/bin/env bash
set -euo pipefail

GENERATION_MODEL="${GENERATION_MODEL:-llama3.1:8b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed."
  echo "Install it from https://ollama.com/download, then rerun this script."
  exit 1
fi

echo "Pulling generation model: ${GENERATION_MODEL}"
ollama pull "${GENERATION_MODEL}"

echo "Pulling embedding model: ${EMBEDDING_MODEL}"
ollama pull "${EMBEDDING_MODEL}"

echo "Models are ready."
echo "Start Ollama if it is not already running, then run:"
echo "  python3 tools/ai_artifact_assistant.py doctor"
