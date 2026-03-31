import plotly.graph_objects as go
import json


SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "c++", "c#", "r", "scala", "kotlin", "swift", "go", "rust", "typescript"
    ],
    "ML / AI": [
        "machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning",
        "neural network", "transformer", "bert", "llm", "generative ai", "fine-tuning",
        "classification", "regression", "clustering", "feature engineering"
    ],
    "Frameworks & Libraries": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "huggingface", "langchain",
        "flask", "fastapi", "django", "react", "node", "spring", "opencv", "nltk", "spacy"
    ],
    "Data & Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "dbt"
    ],
    "Cloud & DevOps": [
        "aws", "gcp", "azure", "docker", "kubernetes", "ci/cd", "git", "github",
        "terraform", "linux", "bash", "render", "heroku", "vercel", "mlflow"
    ],
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving", "analytical",
        "project management", "agile", "collaboration", "critical thinking", "research"
    ]
}


def compute_category_scores(resume_text: str, jd_text: str) -> dict:
    """
    For each skill category, compute:
      - resume_score  : % of category keywords found in resume  (0–100)
      - jd_score      : % of category keywords found in JD      (0–100)
      - overlap_score : % of JD-required keywords also in resume (0–100)
    """
    resume_lower = resume_text.lower()
    jd_lower     = jd_text.lower()

    results = {}
    for category, keywords in SKILL_CATEGORIES.items():
        in_resume = [kw for kw in keywords if kw in resume_lower]
        in_jd     = [kw for kw in keywords if kw in jd_lower]

        resume_score = round(len(in_resume) / len(keywords) * 100, 1)
        jd_score     = round(len(in_jd)     / len(keywords) * 100, 1)

        if in_jd:
            matched_jd = [kw for kw in in_jd if kw in resume_lower]
            overlap_score = round(len(matched_jd) / len(in_jd) * 100, 1)
        else:
            overlap_score = 0.0

        results[category] = {
            "resume_score":  resume_score,
            "jd_score":      jd_score,
            "overlap_score": overlap_score,
            "matched":       in_resume,
            "jd_required":   in_jd,
        }

    return results


def generate_radar_chart(resume_text: str, jd_text: str, candidate_name: str = "Candidate") -> str:
    """
    Builds a Plotly radar chart comparing resume coverage vs JD requirements
    across all skill categories.

    Returns the chart as a JSON string (for embedding in Flask via Plotly JS).
    """
    scores  = compute_category_scores(resume_text, jd_text)
    cats    = list(scores.keys())

    resume_vals  = [scores[c]["resume_score"]  for c in cats]
    jd_vals      = [scores[c]["jd_score"]      for c in cats]
    overlap_vals = [scores[c]["overlap_score"] for c in cats]

    # Close the polygon
    cats_closed         = cats + [cats[0]]
    resume_vals_closed  = resume_vals  + [resume_vals[0]]
    jd_vals_closed      = jd_vals      + [jd_vals[0]]
    overlap_vals_closed = overlap_vals + [overlap_vals[0]]

    fig = go.Figure()

    # Trace 1 — JD requirements (what the job wants)
    fig.add_trace(go.Scatterpolar(
        r    = jd_vals_closed,
        theta= cats_closed,
        fill = "toself",
        name = "JD Requirements",
        line = dict(color="#6366f1", width=2),
        fillcolor="rgba(99, 102, 241, 0.15)",
        hovertemplate="<b>%{theta}</b><br>JD coverage: %{r:.1f}%<extra></extra>"
    ))

    # Trace 2 — Resume coverage (what candidate has)
    fig.add_trace(go.Scatterpolar(
        r    = resume_vals_closed,
        theta= cats_closed,
        fill = "toself",
        name = "Your Resume",
        line = dict(color="#10b981", width=2),
        fillcolor="rgba(16, 185, 129, 0.15)",
        hovertemplate="<b>%{theta}</b><br>Resume coverage: %{r:.1f}%<extra></extra>"
    ))

    # Trace 3 — Overlap (resume ∩ JD)
    fig.add_trace(go.Scatterpolar(
        r    = overlap_vals_closed,
        theta= cats_closed,
        fill = "toself",
        name = "Matched Skills",
        line = dict(color="#f59e0b", width=2, dash="dot"),
        fillcolor="rgba(245, 158, 11, 0.10)",
        hovertemplate="<b>%{theta}</b><br>Skill match: %{r:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible   = True,
                range     = [0, 100],
                ticksuffix= "%",
                tickfont  = dict(size=10, color="#6b7280"),
                gridcolor = "rgba(107, 114, 128, 0.3)",
                linecolor = "rgba(107, 114, 128, 0.3)",
            ),
            angularaxis=dict(
                tickfont  = dict(size=12, color="#374151"),
                gridcolor = "rgba(107, 114, 128, 0.2)",
                linecolor = "rgba(107, 114, 128, 0.3)",
            )
        ),
        showlegend   = True,
        legend       = dict(
            orientation="h",
            yanchor="bottom",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        title=dict(
            text = f"Skills Coverage — {candidate_name}",
            font = dict(size=16, color="#111827"),
            x    = 0.5
        ),
        margin    = dict(t=80, b=80, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font         = dict(family="Inter, sans-serif"),
        height       = 520,
    )

    return fig.to_json()


def get_chart_summary(resume_text: str, jd_text: str) -> dict:
    """
    Returns a plain-dict summary of category scores — useful for
    passing to Groq or rendering as a table in the UI.

    Example return value:
    {
      "ML / AI":        {"resume": 45.0, "jd": 60.0, "overlap": 75.0},
      "Cloud & DevOps": {"resume": 20.0, "jd": 40.0, "overlap": 50.0},
      ...
    }
    """
    scores = compute_category_scores(resume_text, jd_text)
    return {
        cat: {
            "resume":  data["resume_score"],
            "jd":      data["jd_score"],
            "overlap": data["overlap_score"],
        }
        for cat, data in scores.items()
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE_RESUME = """
    Python developer with experience in machine learning, deep learning, and NLP.
    Worked with TensorFlow, PyTorch, scikit-learn, and HuggingFace transformers.
    Built REST APIs using Flask and FastAPI. Deployed models on AWS and Docker.
    Proficient in SQL, pandas, numpy. Used Git, GitHub for version control.
    Strong analytical and communication skills. Experience with agile teams.
    """

    SAMPLE_JD = """
    We are looking for an ML Engineer with Python, deep learning, and NLP experience.
    Must know PyTorch or TensorFlow, HuggingFace, and LLM fine-tuning.
    Experience with Flask or FastAPI for model serving. Docker and AWS required.
    SQL and data pipeline experience (Spark, Kafka) is a plus.
    Strong communication and teamwork skills essential.
    """

    print("── Category Scores ──")
    summary = get_chart_summary(SAMPLE_RESUME, SAMPLE_JD)
    for cat, vals in summary.items():
        print(f"  {cat:<28}  resume={vals['resume']:5.1f}%  jd={vals['jd']:5.1f}%  overlap={vals['overlap']:5.1f}%")

    chart_json = generate_radar_chart(SAMPLE_RESUME, SAMPLE_JD, candidate_name="Mokshi Jain")
    print(f"\n── Chart JSON length: {len(chart_json)} chars (Plotly payload ready)")
    print("✓ charts.py working correctly")