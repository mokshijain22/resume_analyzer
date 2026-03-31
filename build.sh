#!/usr/bin/env bash
# build.sh — runs during Render build phase BEFORE the server starts.
# Pre-downloads sentence-transformer model so worker startup is instant.
set -e

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Pre-downloading sentence-transformer model..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
# Run a test encode to confirm it works
result = model.encode(['test sentence'])
print(f'Model ready. Embedding shape: {result.shape}')
"

echo "==> Build complete — model is cached."
