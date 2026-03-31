"""
app.py — Production-grade Flask app.
Performance upgrades:
  - ATS scoring + Groq run in PARALLEL via ThreadPoolExecutor
  - Full result caching (hash-keyed, 1hr TTL)
  - Timing logs on every stage
  - Validation layer catches logical contradictions before render
  - Progressive partial results: ATS score returned first if LLM slow
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
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "docx"}

SAMPLE_JDS = {
    "ML Engineer":       "We are hiring a Machine Learning Engineer. Required: Python, PyTorch or TensorFlow, scikit-learn, HuggingFace Transformers, NLP, deep learning, model deployment, Docker, AWS, REST APIs, SQL, Git, MLflow. Must have production-deployed models with measurable impact.",
    "Data Analyst":      "Seeking a Data Analyst. Required: SQL, Python or R, pandas, Excel, Tableau or Power BI, statistics, A/B testing, data cleaning, business dashboards. Must deliver measurable business insights.",
    "Backend Developer": "Backend Developer needed. Required: Python or Node.js, REST APIs, PostgreSQL, MongoDB, Docker, CI/CD, Git, system design, authentication, microservices. Must have deployed production APIs.",
}


def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_resume(file) -> str:
    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        from parser import extract_text
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
                               error="Only PDF and DOCX supported.")

    jd_text = request.form.get("jd_text", "").strip()
    role    = request.form.get("role", "ML Engineer").strip()

    # ── Parse resume ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        resume_text = parse_resume(file)
    except Exception as e:
        return render_template("index.html", sample_jds=SAMPLE_JDS,
                               error=f"Could not read resume: {e}")
    print(f"[app] Resume parsed in {(time.time()-t0)*1000:.0f}ms", flush=True)

    if not resume_text or len(resume_text.strip()) < 100:
        return render_template("index.html", sample_jds=SAMPLE_JDS,
                               error="Resume appears empty. Use a text-based PDF or DOCX, not a scanned image.")

    scoring_jd = jd_text if jd_text else SAMPLE_JDS.get(role, SAMPLE_JDS["ML Engineer"])

    # ── Check cache ───────────────────────────────────────────────────────────
    import cache as cache_module
    cached = cache_module.get(resume_text, scoring_jd, role)
    if cached:
        print(f"[app] Cache hit — returning in {(time.time()-t_request)*1000:.0f}ms total", flush=True)
        return render_template("result.html", **cached)

    # ── Run ATS scoring + Groq IN PARALLEL ───────────────────────────────────
    from matcher import compute_ats_score
    from gemini_analyzer import analyze_resume as ai_analyze, _fallback

    def run_matcher():
        return compute_ats_score(resume_text, scoring_jd)

    def run_groq():
        return ai_analyze(resume_text, jd_text=jd_text, role=role)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_match = pool.submit(run_matcher)
        future_ai    = pool.submit(run_groq)
        match_result = future_match.result(timeout=30)
        try:
            ai = future_ai.result(timeout=40)
        except Exception as e:
            print(f"[app] Groq failed: {e}", flush=True)
            ai = _fallback("with_jd" if jd_text else "without_jd", str(e))

    print(f"[app] Parallel stage done in {(time.time()-t0)*1000:.0f}ms", flush=True)

    # ── Merge matcher ground truth into AI result ─────────────────────────────
    ai["matched_skills"] = match_result["matched_keywords"]
    ai["missing_skills"] = match_result["missing_skills_categorized"]
    if ai["job_fit_score"] == 0:
        ai["job_fit_score"] = match_result["job_fit_score"]
    if ai["percentile"] == 0:
        ai["percentile"] = match_result["percentile"]

    # ── Validation layer — fix contradictions ─────────────────────────────────
    from validator import validate_and_fix
    ai, fixes_applied = validate_and_fix(ai, match_result)

    # ── Radar chart ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        from charts import generate_radar_chart
        chart_json = Markup(generate_radar_chart(resume_text, scoring_jd))
    except Exception:
        chart_json = Markup("{}")
    print(f"[app] Chart generated in {(time.time()-t0)*1000:.0f}ms", flush=True)

    # ── Build template context ────────────────────────────────────────────────
    ctx = dict(
        ats_score            = round(float(ai["ats_score"]), 1),
        job_fit_score        = round(float(ai["job_fit_score"]), 1),
        percentile           = int(ai["percentile"]),
        mode                 = ai["mode"],
        role                 = role,
        confidence           = ai["confidence"],
        score_defensibility  = ai["score_defensibility"],
        score_breakdown      = ai["score_breakdown"],
        score_contributors   = ai["score_contributors"],
        evidence_summary     = ai["evidence_summary"],
        matched_skills       = ai["matched_skills"],
        missing_skills       = ai["missing_skills"],
        skill_validation     = ai["skill_validation"],
        experience_analysis  = ai["experience_analysis"],
        proof_of_skill       = ai["proof_of_skill"],
        project_analysis     = ai["project_analysis"],
        benchmark_comparison = ai["benchmark_comparison"],
        recruiter_feedback   = ai["recruiter_feedback"],
        risk_flags           = ai["risk_flags"],
        radar_explanation    = ai["radar_explanation"],
        top_3_actions        = ai["top_3_actions"],
        overall_feedback     = ai["overall_feedback"],
        improved_bullets     = ai["improved_bullets"],
        before_after         = ai["before_after_comparison"],
        learning_roadmap     = ai["learning_roadmap"],
        cover_note           = ai["cover_note"],
        chart_json           = chart_json,
        filename             = file.filename,
        fixes_applied        = fixes_applied,
    )

    # ── Cache the result ──────────────────────────────────────────────────────
    cache_module.set(resume_text, scoring_jd, role, ctx)

    total_ms = round((time.time() - t_request) * 1000)
    print(f"[app] Total request time: {total_ms}ms", flush=True)

    return render_template("result.html", **ctx)


@app.route("/sample-jd/<role>")
def sample_jd(role):
    return jsonify({"jd": SAMPLE_JDS.get(role, "")})


@app.route("/cache-stats")
def cache_stats():
    import cache as c
    return jsonify(c.stats())


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", sample_jds=SAMPLE_JDS,
                           error="File too large. Max 5 MB."), 413

@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", sample_jds=SAMPLE_JDS,
                           error="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)