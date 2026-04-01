"""
matcher.py — Lightweight ATS scoring using TF-IDF cosine similarity.
NO sentence-transformers. NO torch. NO heavy models.
RAM usage: ~10 MB (vs ~1.2 GB with sentence-transformers).
Render free tier compatible (512 MB limit).

Scoring logic (defensible in interviews):
  - TF-IDF vectorizes resume + JD into term-frequency vectors
  - Cosine similarity measures semantic overlap (not just exact keywords)
  - Keyword matching adds precision for role-specific terms
  - Blended: 55% TF-IDF similarity + 45% keyword match
"""
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Keyword bank with priority tiers ─────────────────────────────────────────
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


def _clean(text: str) -> str:
    """Lowercase, remove special chars, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_ats_score(resume_text: str, jd_text: str) -> dict:
    """
    Returns full ATS scoring dict.
    Uses TF-IDF cosine similarity — no external model download needed.
    """
    t_total = time.time()

    resume_clean = _clean(resume_text)
    jd_clean     = _clean(jd_text)

    # ── Stage 1: TF-IDF cosine similarity ────────────────────────────────────
    t0 = time.time()
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams for better context
            stop_words="english",
            max_features=5000,
        )
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
        cos_sim      = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
        cos_score    = cos_sim * 100
    except Exception:
        cos_score = 50.0
        cos_sim   = 0.5
    tfidf_ms = round((time.time() - t0) * 1000)

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
    kw_ms       = round((time.time() - t0) * 1000)

    # ── Stage 3: Blended ATS + derived scores ────────────────────────────────
    # 55% TF-IDF (semantic) + 45% keyword (precision)
    ats_score = round(0.55 * cos_score + 0.45 * kw_score, 2)

    # Job fit — penalise missing critical skills heavily
    critical_penalty = len(missing_critical) * 7
    job_fit_score    = max(0, round(ats_score - critical_penalty, 1))

    # Percentile heuristic
    if ats_score >= 85:   percentile = 90
    elif ats_score >= 75: percentile = 75
    elif ats_score >= 65: percentile = 55
    elif ats_score >= 55: percentile = 35
    elif ats_score >= 45: percentile = 20
    else:                 percentile = 10

    total_ms = round((time.time() - t_total) * 1000)

    print(
        f"[matcher] tfidf={tfidf_ms}ms kw={kw_ms}ms total={total_ms}ms "
        f"cos={cos_sim:.3f} kw={kw_score:.1f} ats={ats_score}",
        flush=True,
    )

    def cap(lst): return [kw.title() for kw in lst]

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
    }


if __name__ == "__main__":
    RESUME = "Python PyTorch scikit-learn Flask pandas numpy Git NLP HuggingFace machine learning deep learning REST API deployed on Render"
    JD     = "ML Engineer Python PyTorch NLP HuggingFace Flask Docker AWS SQL required"
    r = compute_ats_score(RESUME, JD)
    print(f"ATS Score   : {r['ats_score']}%")
    print(f"Job Fit     : {r['job_fit_score']}%")
    print(f"Matched     : {r['matched_keywords']}")
    print(f"Critical gap: {r['missing_skills_categorized']['critical']}")