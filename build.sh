#!/usr/bin/env bash
# build.sh — Pre-downloads the sentence-transformer model during build.
# This means zero model download time on first request.
set -e

echo "==> Installing dependencies..."
pip install -r requirements.txt

echo "==> Pre-downloading sentence-transformer model..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
result = model.encode(['test sentence'])
print(f'Model ready. Shape: {result.shape}')
"

echo "==> Verifying other imports..."
python -c "
import flask, groq, pdfplumber, plotly
print('All imports OK')
"

echo "==> Build complete."