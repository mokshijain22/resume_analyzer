"""
app.py — Full production Flask app with sentence-transformers.
For servers with >2GB RAM (not Render free tier).
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
                               error="Resume appears empty or unreadable. Use a text-based PDF or DOCX.")

    scoring_jd = jd_text if jd_text else SAMPLE_JDS.get(role, SAMPLE_JDS["ML Engineer"])

    # ── Check cache ───────────────────────────────────────────────────────────
    cache_module = None
    try:
        import cache as cache_module
        cached = cache_module.get(resume_text, scoring_jd, role)
        if cached:
            print(f"[app] Cache HIT — {(time.time()-t_request)*1000:.0f}ms", flush=True)
            return render_template("result.html", **cached)
    except Exception as e:
        print(f"[app] Cache error (non-fatal): {e}", flush=True)
        cache_module = None

    # ── Run ATS scoring + Groq IN PARALLEL ───────────────────────────────────
    from matcher import compute_ats_score
    from gemini_analyzer import analyze_resume as ai_analyze, _fallback

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_match = pool.submit(compute_ats_score, resume_text, scoring_jd)
        future_ai    = pool.submit(ai_analyze, resume_text, jd_text, role)

        try:
            match_result = future_match.result(timeout=30)
            print(f"[app] Matcher: ats={match_result['ats_score']}", flush=True)
        except Exception as e:
            return render_template("index.html", sample_jds=SAMPLE_JDS,
                                   error=f"Scoring failed: {str(e)}")

        try:
            ai = future_ai.result(timeout=50)
            print(f"[app] Groq: ats={ai.get('ats_score', 'N/A')}", flush=True)
        except Exception as e:
            print(f"[app] Groq fallback: {e}", flush=True)
            ai = _fallback("with_jd" if jd_text else "without_jd", str(e)[:80])

    print(f"[app] Parallel stage: {(time.time()-t0)*1000:.0f}ms", flush=True)

    # ── Matcher is ground truth for keyword scores ────────────────────────────
    final_ats_score  = match_result["ats_score"]
    final_job_fit    = match_result["job_fit_score"]
    final_percentile = match_result["percentile"]

    # Blend with Groq if it returned a valid score
    groq_ats = float(ai.get("ats_score", 0))
    if groq_ats > 0 and abs(groq_ats - final_ats_score) <= 30:
        final_ats_score = round(0.7 * final_ats_score + 0.3 * groq_ats, 1)

    # Use matcher's categorized skills
    ai["matched_skills"] = match_result["matched_keywords"]
    ai["missing_skills"] = match_result["missing_skills_categorized"]

    # ── Validation layer ──────────────────────────────────────────────────────
    fixes_applied = []
    try:
        from validator import validate_and_fix
        ai, fixes_applied = validate_and_fix(ai, match_result)
    except Exception as e:
        print(f"[app] Validator error (non-fatal): {e}", flush=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    chart_json = Markup("{}")
    try:
        from charts import generate_radar_chart
        chart_json = Markup(generate_radar_chart(resume_text, scoring_jd))
    except Exception as e:
        print(f"[app] Chart error (non-fatal): {e}", flush=True)

    # ── Score breakdown fallback ──────────────────────────────────────────────
    score_breakdown = ai.get("score_breakdown", {})
    if not score_breakdown or all(v == 0 for v in score_breakdown.values()):
        kw  = match_result.get("keyword_score", 50)
        cos = match_result.get("cosine_similarity", 0.5) * 100
        score_breakdown = {
            "skills_match":     round(min(kw / 100 * 30, 30), 1),
            "experience_depth": round(min(cos / 100 * 25, 25), 1),
            "projects_quality": 7.5,
            "tools_tech_stack": round(min(len(match_result["matched_keywords"]) / 10 * 10, 10), 1),
            "resume_quality":   6.0,
            "proof_of_work":    5.0,
        }

    # ── Build template context ────────────────────────────────────────────────
    ctx = dict(
        ats_score            = round(final_ats_score, 1),
        job_fit_score        = round(final_job_fit, 1),
        percentile           = int(final_percentile),
        mode                 = ai.get("mode", "with_jd"),
        role                 = role,
        confidence           = ai.get("confidence", {"level": "Medium", "reason": "Analysis complete."}),
        score_defensibility  = ai.get("score_defensibility", ""),
        score_breakdown      = score_breakdown,
        score_contributors   = ai.get("score_contributors", {"positive": [], "negative": []}),
        evidence_summary     = ai.get("evidence_summary", {
            "projects_detected": 0, "projects_with_metrics": 0,
            "projects_with_deployment": 0, "tools_detected": [],
            "metrics_found": [], "links_detected": False,
            "github_detected": False, "summary": "",
        }),
        matched_skills       = match_result["matched_keywords"],
        missing_skills       = match_result["missing_skills_categorized"],
        skill_validation     = ai.get("skill_validation", []),
        experience_analysis  = ai.get("experience_analysis", {
            "total_experience_level": "Junior", "skills_depth": []
        }),
        proof_of_skill       = ai.get("proof_of_skill", {
            "has_deployed_projects": False, "has_live_links": False,
            "has_metrics": False, "has_real_world_datasets": False, "summary": "",
        }),
        project_analysis     = ai.get("project_analysis", []),
        benchmark_comparison = ai.get("benchmark_comparison", {
            "vs_average_candidate": "",
            "vs_top_10_percent":    "",
            "shortlist_probability": int(final_percentile * 0.6),
        }),
        recruiter_feedback   = ai.get("recruiter_feedback", []),
        risk_flags           = ai.get("risk_flags", []),
        radar_explanation    = ai.get("radar_explanation", {}),
        top_3_actions        = ai.get("top_3_actions", []),
        overall_feedback     = ai.get("overall_feedback", ""),
        improved_bullets     = ai.get("improved_bullets", []),
        before_after         = {
            "ats_score_before":      round(final_ats_score, 1),
            "ats_score_after":       min(round(final_ats_score + 12, 1), 95),
            "skill_coverage_before": round(match_result.get("keyword_score", 50), 1),
            "skill_coverage_after":  min(round(match_result.get("keyword_score", 50) + 15, 1), 95),
            "key_improvements":      ai.get("before_after_comparison", {}).get("key_improvements", []),
        },
        learning_roadmap     = ai.get("learning_roadmap", []),
        cover_note           = ai.get("cover_note", ""),
        chart_json           = chart_json,
        filename             = file.filename,
        fixes_applied        = fixes_applied,
    )

    # ── Cache result ──────────────────────────────────────────────────────────
    if cache_module:
        try:
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
    return jsonify({
        "status":   "ok",
        "groq_key": bool(os.environ.get("GROQ_API_KEY")),
    })


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", sample_jds=SAMPLE_JDS,
                           error="File too large. Maximum 5 MB."), 413


@app.errorhandler(500)
def server_error(e):
    print(f"[app] 500: {e}", flush=True)
    return render_template("index.html", sample_jds=SAMPLE_JDS,
                           error="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set!", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port)