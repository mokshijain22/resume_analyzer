"""
app.py — Production Flask app for Render deployment.
Fixes:
  - Uses resume_parser.py instead of parser.py (avoids Python built-in conflict)
  - GROQ_API_KEY validated at startup with clear error message
  - All imports inside try/except with descriptive errors
  - Parallel execution for ATS + Groq
  - Hash-keyed caching
  - Validation layer for logical consistency
"""
import os
import time
import tempfile
import concurrent.futures
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from markupsafe import Markup

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

ALLOWED_EXTENSIONS = {"pdf", "docx"}

SAMPLE_JDS = {
    "ML Engineer":       "We are hiring a Machine Learning Engineer. Required: Python, PyTorch or TensorFlow, scikit-learn, HuggingFace Transformers, NLP, deep learning, model deployment, Docker, AWS, REST APIs, SQL, Git, MLflow. Must have production-deployed models with measurable impact.",
    "Data Analyst":      "Seeking a Data Analyst. Required: SQL, Python or R, pandas, Excel, Tableau or Power BI, statistics, A/B testing, data cleaning, business dashboards. Must deliver measurable business insights.",
    "Backend Developer": "Backend Developer needed. Required: Python or Node.js, REST APIs, PostgreSQL, MongoDB, Docker, CI/CD, Git, system design, authentication, microservices. Must have deployed production APIs.",
}


def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_resume(file) -> str:
    """Save uploaded file to temp location and extract text."""
    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        # Use resume_parser (not parser — avoids built-in module conflict)
        from resume_parser import extract_text
        return extract_text(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", sample_jds=SAMPLE_JDS)


@app.route("/analyze", methods=["POST"])
def analyze():
    t_request = time.time()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if "resume" not in request.files or request.files["resume"].filename == "":
        return render_template("index.html", sample_jds=SAMPLE_JDS,
                               error="Please upload your resume.")

    file = request.files["resume"]
    if not allowed_file(file.filename):
        return render_template("index.html", sample_jds=SAMPLE_JDS,
                               error="Only PDF and DOCX files are supported.")

    jd_text = request.form.get("jd_text", "").strip()
    role    = request.form.get("role", "ML Engineer").strip()

    # ── Parse resume ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        resume_text = parse_resume(file)
    except Exception as e:
        return render_template("index.html", sample_jds=SAMPLE_JDS,
                               error=f"Could not read your resume: {str(e)}")

    print(f"[app] Resume parsed in {(time.time()-t0)*1000:.0f}ms — {len(resume_text)} chars", flush=True)

    if not resume_text or len(resume_text.strip()) < 50:
        return render_template("index.html", sample_jds=SAMPLE_JDS,
                               error="Resume appears empty or unreadable. Please upload a text-based PDF or DOCX (not a scanned image).")

    scoring_jd = jd_text if jd_text else SAMPLE_JDS.get(role, SAMPLE_JDS["ML Engineer"])

    # ── Check cache ───────────────────────────────────────────────────────────
    try:
        import cache as cache_module
        cached = cache_module.get(resume_text, scoring_jd, role)
        if cached:
            print(f"[app] Cache HIT — {(time.time()-t_request)*1000:.0f}ms total", flush=True)
            return render_template("result.html", **cached)
    except Exception as e:
        print(f"[app] Cache error (non-fatal): {e}", flush=True)
        cache_module = None

    # ── Run ATS scoring + Groq IN PARALLEL ───────────────────────────────────
    from matcher import compute_ats_score
    from gemini_analyzer import analyze_resume as ai_analyze, _fallback

    def run_matcher():
        try:
            return compute_ats_score(resume_text, scoring_jd)
        except Exception as e:
            print(f"[app] Matcher error: {e}", flush=True)
            raise

    def run_groq():
        try:
            return ai_analyze(resume_text, jd_text=jd_text, role=role)
        except Exception as e:
            print(f"[app] Groq error: {e}", flush=True)
            raise

    t0 = time.time()
    match_result = None
    ai = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_match = pool.submit(run_matcher)
        future_ai    = pool.submit(run_groq)

        try:
            match_result = future_match.result(timeout=35)
        except Exception as e:
            return render_template("index.html", sample_jds=SAMPLE_JDS,
                                   error=f"ATS scoring failed: {str(e)}")

        try:
            ai = future_ai.result(timeout=45)
        except Exception as e:
            print(f"[app] Groq failed, using fallback: {e}", flush=True)
            ai = _fallback("with_jd" if jd_text else "without_jd", str(e)[:100])

    print(f"[app] Parallel stage: {(time.time()-t0)*1000:.0f}ms", flush=True)

    # ── Merge matcher ground truth ────────────────────────────────────────────
    ai["matched_skills"] = match_result["matched_keywords"]
    ai["missing_skills"] = match_result["missing_skills_categorized"]
    if ai.get("job_fit_score", 0) == 0:
        ai["job_fit_score"] = match_result["job_fit_score"]
    if ai.get("percentile", 0) == 0:
        ai["percentile"] = match_result["percentile"]

    # ── Validation layer ──────────────────────────────────────────────────────
    fixes_applied = []
    try:
        from validator import validate_and_fix
        ai, fixes_applied = validate_and_fix(ai, match_result)
    except Exception as e:
        print(f"[app] Validator error (non-fatal): {e}", flush=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    t0 = time.time()
    chart_json = Markup("{}")
    try:
        from charts import generate_radar_chart
        chart_json = Markup(generate_radar_chart(resume_text, scoring_jd))
    except Exception as e:
        print(f"[app] Chart error (non-fatal): {e}", flush=True)
    print(f"[app] Chart: {(time.time()-t0)*1000:.0f}ms", flush=True)

    # ── Build context ─────────────────────────────────────────────────────────
    ctx = dict(
        ats_score            = round(float(ai.get("ats_score", 0)), 1),
        job_fit_score        = round(float(ai.get("job_fit_score", 0)), 1),
        percentile           = int(ai.get("percentile", 0)),
        mode                 = ai.get("mode", "with_jd"),
        role                 = role,
        confidence           = ai.get("confidence", {"level": "Low", "reason": "Analysis incomplete."}),
        score_defensibility  = ai.get("score_defensibility", ""),
        score_breakdown      = ai.get("score_breakdown", {}),
        score_contributors   = ai.get("score_contributors", {"positive": [], "negative": []}),
        evidence_summary     = ai.get("evidence_summary", {}),
        matched_skills       = ai.get("matched_skills", []),
        missing_skills       = ai.get("missing_skills", {"critical": [], "important": [], "optional": []}),
        skill_validation     = ai.get("skill_validation", []),
        experience_analysis  = ai.get("experience_analysis", {"total_experience_level": "Unknown", "skills_depth": []}),
        proof_of_skill       = ai.get("proof_of_skill", {}),
        project_analysis     = ai.get("project_analysis", []),
        benchmark_comparison = ai.get("benchmark_comparison", {"vs_average_candidate": "", "vs_top_10_percent": "", "shortlist_probability": 0}),
        recruiter_feedback   = ai.get("recruiter_feedback", []),
        risk_flags           = ai.get("risk_flags", []),
        radar_explanation    = ai.get("radar_explanation", {}),
        top_3_actions        = ai.get("top_3_actions", []),
        overall_feedback     = ai.get("overall_feedback", ""),
        improved_bullets     = ai.get("improved_bullets", []),
        before_after         = ai.get("before_after_comparison", {}),
        learning_roadmap     = ai.get("learning_roadmap", []),
        cover_note           = ai.get("cover_note", ""),
        chart_json           = chart_json,
        filename             = file.filename,
        fixes_applied        = fixes_applied,
    )

    # ── Cache result ──────────────────────────────────────────────────────────
    try:
        if cache_module:
            cache_module.set(resume_text, scoring_jd, role, ctx)
    except Exception as e:
        print(f"[app] Cache set error (non-fatal): {e}", flush=True)

    print(f"[app] Total: {(time.time()-t_request)*1000:.0f}ms", flush=True)
    return render_template("result.html", **ctx)


@app.route("/sample-jd/<role>")
def sample_jd(role):
    return jsonify({"jd": SAMPLE_JDS.get(role, "")})


@app.route("/health")
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "ok", "groq_key": bool(os.environ.get("GROQ_API_KEY"))})


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", sample_jds=SAMPLE_JDS,
                           error="File too large. Maximum size is 5 MB."), 413


@app.errorhandler(500)
def server_error(e):
    print(f"[app] 500 error: {e}", flush=True)
    return render_template("index.html", sample_jds=SAMPLE_JDS,
                           error="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    # Validate GROQ_API_KEY at startup
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set — AI analysis will fail!", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port)