"""
matcher.py — Production-grade ATS scoring.
Optimizations:
  - Singleton model (loads ONCE at startup, never reloaded)
  - Request-level embedding cache (hash-keyed)
  - Timing logs on every stage
  - Categorized missing skills: critical / important / optional
"""
import re
import time
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── SINGLETON MODEL — loads once, reused forever ──────────────────────────────
_MODEL: SentenceTransformer | None = None
_EMBED_CACHE: dict = {}   # hash → embedding vector

def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        t0 = time.time()
        print("[matcher] Loading sentence-transformer model...", flush=True)
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"[matcher] Model ready in {time.time()-t0:.2f}s", flush=True)
    return _MODEL

# Pre-load at import time (Flask startup), not on first request
_get_model()

# ── KEYWORD BANK WITH PRIORITY TIERS ─────────────────────────────────────────
CRITICAL_KEYWORDS = [
    "python", "machine learning", "deep learning", "nlp",
    "pytorch", "tensorflow", "scikit-learn", "sql", "docker",
    "rest api", "flask", "fastapi", "git", "aws",
]
IMPORTANT_KEYWORDS = [
    "huggingface", "transformers", "pandas", "numpy", "computer vision",
    "neural network", "bert", "llm", "fine-tuning", "classification",
    "regression", "clustering", "feature engineering", "kubernetes",
    "ci/cd", "gcp", "azure", "mlflow", "airflow", "spark",
    "mongodb", "postgresql", "redis", "langchain",
]
OPTIONAL_KEYWORDS = [
    "typescript", "java", "scala", "kotlin", "go", "rust",
    "opencv", "nltk", "spacy", "xgboost", "lightgbm", "dbt",
    "kafka", "hadoop", "elasticsearch", "terraform", "linux",
    "matplotlib", "seaborn", "plotly", "tableau", "power bi",
    "communication", "teamwork", "leadership", "agile", "research",
]
ALL_KEYWORDS = CRITICAL_KEYWORDS + IMPORTANT_KEYWORDS + OPTIONAL_KEYWORDS


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _get_embedding(text: str) -> np.ndarray:
    """Cached embedding — avoids recomputing same text."""
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _EMBED_CACHE:
        _EMBED_CACHE[key] = _get_model().encode([text])[0]
    return _EMBED_CACHE[key]


def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    """
    Returns full ATS scoring dict with timing breakdown.
    All stages timed and logged.
    """
    timings: dict = {}
    t_total = time.time()

    resume_clean = _clean(resume_text)
    jd_clean     = _clean(jd_text)

    # ── Stage 1: Semantic similarity ─────────────────────────────────────────
    t0 = time.time()
    emb_resume = _get_embedding(resume_clean)
    emb_jd     = _get_embedding(jd_clean)
    cos_sim    = float(cosine_similarity([emb_resume], [emb_jd])[0][0])
    cos_score  = cos_sim * 100
    timings["embedding_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 2: Keyword matching ─────────────────────────────────────────────
    t0 = time.time()
    jd_critical  = [kw for kw in CRITICAL_KEYWORDS  if kw in jd_clean]
    jd_important = [kw for kw in IMPORTANT_KEYWORDS if kw in jd_clean]
    jd_optional  = [kw for kw in OPTIONAL_KEYWORDS  if kw in jd_clean]

    matched_critical  = [kw for kw in jd_critical  if kw in resume_clean]
    matched_important = [kw for kw in jd_important if kw in resume_clean]
    matched_optional  = [kw for kw in jd_optional  if kw in resume_clean]

    missing_critical  = [kw for kw in jd_critical  if kw not in resume_clean]
    missing_important = [kw for kw in jd_important if kw not in resume_clean]
    missing_optional  = [kw for kw in jd_optional  if kw not in resume_clean]

    all_jd      = jd_critical + jd_important + jd_optional
    all_matched = matched_critical + matched_important + matched_optional
    kw_score    = (len(all_matched) / len(all_jd) * 100) if all_jd else 50.0
    timings["keyword_ms"] = round((time.time() - t0) * 1000)

    # ── Stage 3: Blended ATS + derived scores ────────────────────────────────
    t0 = time.time()
    ats_score = round(0.6 * cos_score + 0.4 * kw_score, 2)

    # Job fit penalises hard critical gaps
    critical_penalty = len(missing_critical) * 7
    job_fit_score    = max(0, round(ats_score - critical_penalty, 1))

    # Percentile heuristic
    if ats_score >= 85: percentile = 90
    elif ats_score >= 75: percentile = 75
    elif ats_score >= 65: percentile = 55
    elif ats_score >= 55: percentile = 35
    elif ats_score >= 45: percentile = 20
    else: percentile = 10

    timings["scoring_ms"] = round((time.time() - t0) * 1000)
    timings["total_ms"]   = round((time.time() - t_total) * 1000)

    def cap(lst): return [kw.title() for kw in lst]

    print(f"[matcher] Scoring done: {timings}", flush=True)

    return {
        "ats_score":         ats_score,
        "cosine_similarity": round(cos_sim, 4),
        "keyword_score":     round(kw_score, 2),
        "job_fit_score":     job_fit_score,
        "percentile":        percentile,
        "matched_keywords":  cap(all_matched),
        "missing_keywords":  cap(missing_critical + missing_important + missing_optional),
        "missing_skills_categorized": {
            "critical":  cap(missing_critical),
            "important": cap(missing_important),
            "optional":  cap(missing_optional),
        },
        "timings": timings,
    }


if __name__ == "__main__":
    RESUME = "Python PyTorch scikit-learn Flask pandas numpy Git NLP HuggingFace machine learning deep learning REST API"
    JD     = "ML Engineer Python PyTorch NLP HuggingFace Flask Docker AWS SQL MLflow required"
    r = compute_ats_score(RESUME, JD)
    print(f"ATS     : {r['ats_score']}%")
    print(f"Job Fit : {r['job_fit_score']}%")
    print(f"Timings : {r['timings']}")