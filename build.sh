#!/usr/bin/env bash
# build.sh — Render build script.
# No heavy model downloads. TF-IDF is pure scikit-learn, loads instantly.
set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Verifying key imports..."
python -c "
import flask, groq, pdfplumber, plotly
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
print('All imports OK')

# Quick smoke test of TF-IDF scorer
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
m = v.fit_transform(['python machine learning flask', 'python nlp flask docker'])
from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(m[0], m[1])[0][0]
print(f'TF-IDF smoke test: similarity={score:.3f} (should be ~0.4-0.7)')
"

echo "==> Build complete."