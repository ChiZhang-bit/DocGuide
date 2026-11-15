#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATASET_PATH="dataset/HybridQA"
EMBEDDING_MODEL_PATH="model/embedding"
RERANKER_MODEL_PATH="model/reranker"
MAX_WORKERS=4

args=("$@")
if [[ $# -eq 0 ]]; then
  args=(
    --data "$DATASET_PATH"
    --embedding-model-path "$EMBEDDING_MODEL_PATH"
    --reranker-model-path "$RERANKER_MODEL_PATH"
    --workers "$MAX_WORKERS"
  )
fi

export PYTHONDONTWRITEBYTECODE=1
"${PYTHON:-python3}" -c 'import sys; sys.exit("DocGuide requires Python 3.10 or newer; activate the environment or set PYTHON.") if sys.version_info < (3, 10) else None'
exec "${PYTHON:-python3}" -m scripts.run "${args[@]}"
