"""
charts.py — Plotly radar chart for skill coverage.
Uses simple keyword matching (no ML models needed).
"""
import re
import plotly.graph_objects as go

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "c++", "c#", "r", "scala",
        "kotlin", "swift", "go", "rust", "typescript",
    ],
    "ML / AI": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "reinforcement learning", "neural network", "transformer", "bert",
        "llm", "generative ai", "fine-tuning", "classification",
        "regression", "clustering", "feature engineering",
    ],
    "Frameworks & Libraries": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "huggingface",
        "langchain", "flask", "fastapi", "django", "react", "opencv",
        "nltk", "spacy", "xgboost", "lightgbm",
    ],
    "Data & Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis",
        "elasticsearch", "pandas", "numpy", "spark", "hadoop",
        "kafka", "airflow", "dbt",
    ],
    "Cloud & DevOps": [
        "aws", "gcp", "azure", "docker", "kubernetes", "ci/cd",
        "git", "github", "terraform", "linux", "mlflow",
        "render", "heroku",
    ],
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "analytical", "agile", "collaboration", "research",
    ],
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def compute_category_scores(resume_text: str, jd_text: str) -> dict:
    resume_lower = _clean(resume_text)
    jd_lower     = _clean(jd_text)

    results = {}
    for category, keywords in SKILL_CATEGORIES.items():
        in_resume = [kw for kw in keywords if kw in resume_lower]
        in_jd     = [kw for kw in keywords if kw in jd_lower]

        resume_score = round(len(in_resume) / len(keywords) * 100, 1)
        jd_score     = round(len(in_jd)     / len(keywords) * 100, 1)

        if in_jd:
            matched_jd    = [kw for kw in in_jd if kw in resume_lower]
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


def generate_radar_chart(resume_text: str, jd_text: str,
                          candidate_name: str = "Candidate") -> str:
    scores = compute_category_scores(resume_text, jd_text)
    cats   = list(scores.keys())

    resume_vals  = [scores[c]["resume_score"]  for c in cats]
    jd_vals      = [scores[c]["jd_score"]      for c in cats]
    overlap_vals = [scores[c]["overlap_score"] for c in cats]

    cats_c   = cats + [cats[0]]
    res_c    = resume_vals  + [resume_vals[0]]
    jd_c     = jd_vals      + [jd_vals[0]]
    over_c   = overlap_vals + [overlap_vals[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=jd_c, theta=cats_c, fill="toself", name="JD Requirements",
        line=dict(color="#6366f1", width=2),
        fillcolor="rgba(99,102,241,0.12)",
        hovertemplate="<b>%{theta}</b><br>JD coverage: %{r:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=res_c, theta=cats_c, fill="toself", name="Your Resume",
        line=dict(color="#10b981", width=2),
        fillcolor="rgba(16,185,129,0.12)",
        hovertemplate="<b>%{theta}</b><br>Resume coverage: %{r:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=over_c, theta=cats_c, fill="toself", name="Matched Skills",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        fillcolor="rgba(245,158,11,0.07)",
        hovertemplate="<b>%{theta}</b><br>Skill match: %{r:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100], ticksuffix="%",
                tickfont=dict(size=10, color="#6b7280"),
                gridcolor="rgba(107,114,128,0.25)",
                linecolor="rgba(107,114,128,0.25)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#374151"),
                gridcolor="rgba(107,114,128,0.18)",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
        margin=dict(t=30, b=50, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif"),
        height=400,
    )
    return fig.to_json()


def get_chart_summary(resume_text: str, jd_text: str) -> dict:
    scores = compute_category_scores(resume_text, jd_text)
    return {
        cat: {
            "resume":  data["resume_score"],
            "jd":      data["jd_score"],
            "overlap": data["overlap_score"],
        }
        for cat, data in scores.items()
    }


if __name__ == "__main__":
    RESUME = "Python machine learning NLP PyTorch Flask scikit-learn pandas numpy Git"
    JD     = "Python PyTorch NLP Docker AWS machine learning REST API SQL"
    summary = get_chart_summary(RESUME, JD)
    for cat, vals in summary.items():
        print(f"{cat:<28} resume={vals['resume']:5.1f}% jd={vals['jd']:5.1f}% overlap={vals['overlap']:5.1f}%")
    print("\nChart JSON length:", len(generate_radar_chart(RESUME, JD)), "chars")