"""
gemini_analyzer.py — Fixed Groq analyzer.
Key fixes:
  - Prompt produces MUCH smaller JSON (was hitting token limit mid-response)
  - max_tokens raised to 2048 with temperature 0.3
  - Fallback extracts partial data if JSON is incomplete
  - All fields have safe defaults
"""
import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_CLIENT = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MARKET_PROFILES = {
    "ML Engineer":       "Python, PyTorch/TensorFlow, scikit-learn, NLP, model deployment, Docker, AWS/GCP, MLflow, REST APIs, SQL, Git. Expected: production models, measurable accuracy, real datasets.",
    "Data Analyst":      "SQL, Python/R, pandas, Excel, Tableau/Power BI, statistics, A/B testing, dashboards. Expected: business impact metrics, stakeholder reports.",
    "Backend Developer": "Python/Node.js, REST APIs, PostgreSQL/MongoDB, Docker, CI/CD, Git, system design. Expected: production APIs, deployed apps, unit tests.",
}


def analyze_resume(resume_text: str, jd_text: str = "", role: str = "") -> dict:
    mode   = "with_jd" if jd_text.strip() else "without_jd"
    target = jd_text.strip()[:800] if mode == "with_jd" else MARKET_PROFILES.get(role, MARKET_PROFILES["ML Engineer"])
    resume = resume_text.strip()[:2000]
    role_label = role or "ML Engineer"

    t0 = time.time()

    # ── Compact prompt — strict JSON, minimal fields to avoid truncation ──────
    prompt = f"""You are a senior hiring manager. Analyze this resume against the {"job description" if mode=="with_jd" else f"{role_label} market profile"}.

RESUME:
{resume}

{"JOB DESCRIPTION" if mode=="with_jd" else "MARKET PROFILE"}:
{target}

Return ONLY valid compact JSON. No markdown. No explanation. No extra text before or after.
Every string value must be SHORT (under 100 chars). Arrays max 4 items each.

{{
  "confidence_level": "High or Medium or Low",
  "confidence_reason": "one short sentence why",
  "ats_score": <number 0-100>,
  "score_defensibility": "short explanation of score with 2-3 specific reasons",
  "skills_match": <0-30>,
  "experience_depth": <0-25>,
  "projects_quality": <0-15>,
  "tools_tech_stack": <0-10>,
  "resume_quality": <0-10>,
  "proof_of_work": <0-10>,
  "positive_contributors": ["reason 1", "reason 2"],
  "negative_contributors": ["reason 1", "reason 2"],
  "projects_detected": <int>,
  "projects_with_metrics": <int>,
  "projects_with_deployment": <int>,
  "tools_detected": ["tool1", "tool2", "tool3"],
  "metrics_found": ["metric1"],
  "github_detected": <true or false>,
  "evidence_summary": "one sentence describing what was found in resume",
  "job_fit_score": <0-100>,
  "percentile": <0-100>,
  "matched_skills": ["skill1", "skill2", "skill3", "skill4"],
  "critical_missing": ["skill1", "skill2"],
  "important_missing": ["skill1", "skill2"],
  "optional_missing": ["skill1"],
  "experience_level": "Fresher or Junior or Mid or Senior",
  "skills_depth": [
    {{"skill": "Python", "level": "Intermediate", "reason": "used in 2 projects", "has_metrics": false}}
  ],
  "has_deployed_projects": <true or false>,
  "has_metrics": <true or false>,
  "project_analysis": [
    {{"name": "project name", "score": <0-10>, "prod_score": <0-10>, "deployed": <bool>, "has_metrics": <bool>, "feedback": "short feedback"}}
  ],
  "vs_average": "one sentence comparison to average candidate",
  "vs_top_10": "one sentence comparison to top 10 percent",
  "shortlist_probability": <0-100>,
  "recruiter_feedback": ["specific rejection reason 1", "specific rejection reason 2"],
  "risk_flags": ["flag 1", "flag 2"],
  "action_1_title": "action title",
  "action_1_why": "why this action",
  "action_1_how": "how to do it referencing their projects",
  "action_2_title": "action title",
  "action_2_why": "why",
  "action_2_how": "how",
  "action_3_title": "action title",
  "action_3_why": "why",
  "action_3_how": "how",
  "overall_feedback": "3 sentences max evidence-based summary",
  "bullet_before_1": "original weak bullet",
  "bullet_after_1": "improved metric-driven version",
  "bullet_before_2": "original weak bullet",
  "bullet_after_2": "improved version",
  "ats_before": <current score>,
  "ats_after": <projected score after fixes, max +20>,
  "coverage_before": <0-100>,
  "coverage_after": <0-100>,
  "week1_focus": "topic", "week1_task": "specific task referencing their project", "week1_outcome": "deliverable",
  "week2_focus": "topic", "week2_task": "specific task", "week2_outcome": "deliverable",
  "week3_focus": "topic", "week3_task": "specific task", "week3_outcome": "deliverable",
  "week4_focus": "topic", "week4_task": "specific task", "week4_outcome": "deliverable",
  "cover_note": "2 sentence cover note referencing actual projects"
}}"""

    try:
        response = _CLIENT.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 2048,
            temperature = 0.3,
        )
        elapsed = round((time.time() - t0) * 1000)
        raw = response.choices[0].message.content.strip()
        print(f"[groq] {elapsed}ms, {len(raw)} chars, tokens={response.usage.total_tokens}", flush=True)

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()

        data = json.loads(raw)
        return _build_result(data, mode)

    except json.JSONDecodeError as e:
        print(f"[groq] JSON error: {e} — attempting partial recovery", flush=True)
        # Try to recover partial JSON by truncating at last complete field
        if 'raw' in dir():
            partial = _recover_partial_json(raw)
            if partial:
                return _build_result(partial, mode)
        return _fallback(mode, f"JSON parse error — please retry")

    except Exception as e:
        print(f"[groq] API error: {e}", flush=True)
        return _fallback(mode, str(e)[:80])


def _recover_partial_json(raw: str) -> dict | None:
    """Try to close truncated JSON by finding last complete key-value pair."""
    try:
        # Find the last complete string value or number
        truncated = raw.rstrip()
        # Try adding closing brace
        for suffix in ["}", '"}', '"}}', '}}}']:
            try:
                return json.loads(truncated + suffix)
            except Exception:
                pass
        return None
    except Exception:
        return None


def _build_result(d: dict, mode: str) -> dict:
    """Convert flat Groq response dict into the nested structure the template expects."""

    def sg(k, default):
        v = d.get(k, default)
        return v if v is not None else default

    def sgl(k):
        v = d.get(k, [])
        return v if isinstance(v, list) else []

    # Build improved bullets
    improved_bullets = []
    for i in range(1, 4):
        before = sg(f"bullet_before_{i}", "")
        after  = sg(f"bullet_after_{i}",  "")
        if before and after:
            improved_bullets.append(f"BEFORE: {before} | AFTER: {after}")
        elif after:
            improved_bullets.append(after)

    # Build roadmap
    roadmap = []
    for w in range(1, 5):
        focus   = sg(f"week{w}_focus",   "")
        task    = sg(f"week{w}_task",    "")
        outcome = sg(f"week{w}_outcome", "")
        if focus and task:
            roadmap.append({"week": w, "focus": focus, "task": task, "outcome": outcome})

    # Build top 3 actions
    top_3 = []
    for i in range(1, 4):
        title = sg(f"action_{i}_title", "")
        why   = sg(f"action_{i}_why",   "")
        how   = sg(f"action_{i}_how",   "")
        if title:
            top_3.append({"priority": i, "action": title, "why": why, "how": how})

    # Build project analysis
    proj_raw = sgl("project_analysis")
    projects = []
    for p in proj_raw[:4]:
        if isinstance(p, dict):
            projects.append({
                "name":                    str(p.get("name", "Project")),
                "score":                   float(p.get("score", 5)),
                "production_readiness_score": float(p.get("prod_score", 5)),
                "impact_score":            float(p.get("score", 5)),
                "has_deployment":          bool(p.get("deployed", False)),
                "has_metrics":             bool(p.get("has_metrics", False)),
                "real_world_relevance":    "Medium",
                "tech_stack_depth":        "Moderate",
                "feedback":                str(p.get("feedback", "")),
            })

    # Build skills_depth
    depth_raw = sgl("skills_depth")
    skills_depth = []
    for s in depth_raw[:6]:
        if isinstance(s, dict):
            skills_depth.append({
                "skill":       str(s.get("skill", "")),
                "level":       str(s.get("level", "Beginner")),
                "reason":      str(s.get("reason", "")),
                "has_metrics": bool(s.get("has_metrics", False)),
            })

    ats = float(sg("ats_score", 50))
    ats_before = float(sg("ats_before", ats))
    ats_after  = min(float(sg("ats_after", ats + 10)), 97)

    return {
        "mode":    mode,
        "confidence": {
            "level":  str(sg("confidence_level", "Medium")),
            "reason": str(sg("confidence_reason", "Analysis complete.")),
        },
        "ats_score":          ats,
        "score_defensibility": str(sg("score_defensibility", "")),
        "score_breakdown": {
            "skills_match":     float(sg("skills_match",     0)),
            "experience_depth": float(sg("experience_depth", 0)),
            "projects_quality": float(sg("projects_quality", 0)),
            "tools_tech_stack": float(sg("tools_tech_stack", 0)),
            "resume_quality":   float(sg("resume_quality",   0)),
            "proof_of_work":    float(sg("proof_of_work",    0)),
        },
        "score_contributors": {
            "positive": sgl("positive_contributors"),
            "negative": sgl("negative_contributors"),
        },
        "evidence_summary": {
            "projects_detected":        int(sg("projects_detected",        0)),
            "projects_with_metrics":    int(sg("projects_with_metrics",    0)),
            "projects_with_deployment": int(sg("projects_with_deployment", 0)),
            "tools_detected":           sgl("tools_detected"),
            "metrics_found":            sgl("metrics_found"),
            "links_detected":           bool(sg("github_detected", False)),
            "github_detected":          bool(sg("github_detected", False)),
            "summary":                  str(sg("evidence_summary", "")),
        },
        "job_fit_score": float(sg("job_fit_score", 50)),
        "percentile":    float(sg("percentile",    50)),
        "matched_skills": sgl("matched_skills"),
        "missing_skills": {
            "critical":  sgl("critical_missing"),
            "important": sgl("important_missing"),
            "optional":  sgl("optional_missing"),
        },
        "skill_validation": [],   # skipped to save tokens
        "experience_analysis": {
            "total_experience_level": str(sg("experience_level", "Junior")),
            "skills_depth": skills_depth,
        },
        "proof_of_skill": {
            "has_deployed_projects":  bool(sg("has_deployed_projects",  False)),
            "has_live_links":         bool(sg("github_detected",        False)),
            "has_metrics":            bool(sg("has_metrics",            False)),
            "has_real_world_datasets": False,
            "summary":                str(sg("evidence_summary", "")),
        },
        "project_analysis": projects,
        "benchmark_comparison": {
            "vs_average_candidate": str(sg("vs_average", "")),
            "vs_top_10_percent":    str(sg("vs_top_10",  "")),
            "shortlist_probability": int(sg("shortlist_probability", 40)),
        },
        "recruiter_feedback":   sgl("recruiter_feedback"),
        "risk_flags":           sgl("risk_flags"),
        "radar_explanation":    {},
        "top_3_actions":        top_3,
        "overall_feedback":     str(sg("overall_feedback", "")),
        "improved_bullets":     improved_bullets,
        "before_after_comparison": {
            "ats_score_before":      ats_before,
            "ats_score_after":       ats_after,
            "skill_coverage_before": float(sg("coverage_before", 50)),
            "skill_coverage_after":  float(sg("coverage_after",  65)),
            "key_improvements":      [],
        },
        "learning_roadmap": roadmap,
        "cover_note":       str(sg("cover_note", "")),
    }


def _fallback(mode: str, detail: str = "") -> dict:
    return {
        "mode": mode,
        "confidence": {"level": "Low", "reason": f"Analysis failed: {detail}"},
        "ats_score": 0, "score_defensibility": "",
        "score_breakdown": {"skills_match":0,"experience_depth":0,"projects_quality":0,"tools_tech_stack":0,"resume_quality":0,"proof_of_work":0},
        "score_contributors": {"positive":[],"negative":[]},
        "evidence_summary": {"projects_detected":0,"projects_with_metrics":0,"projects_with_deployment":0,"tools_detected":[],"metrics_found":[],"links_detected":False,"github_detected":False,"summary":""},
        "job_fit_score":0,"percentile":0,
        "matched_skills":[],"missing_skills":{"critical":[],"important":[],"optional":[]},
        "skill_validation":[],"experience_analysis":{"total_experience_level":"Unknown","skills_depth":[]},
        "proof_of_skill":{"has_deployed_projects":False,"has_live_links":False,"has_metrics":False,"has_real_world_datasets":False,"summary":""},
        "project_analysis":[],"benchmark_comparison":{"vs_average_candidate":"","vs_top_10_percent":"","shortlist_probability":0},
        "recruiter_feedback":[],"risk_flags":[],"radar_explanation":{},"top_3_actions":[],
        "overall_feedback": f"AI analysis unavailable — {detail}. Please retry.",
        "improved_bullets":[],"before_after_comparison":{"ats_score_before":0,"ats_score_after":0,"skill_coverage_before":0,"skill_coverage_after":0,"key_improvements":[]},
        "learning_roadmap":[],"cover_note":"",
    }


if __name__ == "__main__":
    RESUME = """
    Mokshi Jain — B.Tech CSE AI/ML, MUJ 2027.
    Python, PyTorch, scikit-learn, HuggingFace, Flask, pandas, numpy, Git.
    Fake News Detector: BERT NLP, 94% accuracy, deployed on Render.
    MNIST: CNN PyTorch 99.1%. Real Estate: regression scikit-learn deployed.
    GitHub: github.com/mokshijain
    """
    JD = "ML Engineer Python PyTorch NLP HuggingFace Docker AWS MLflow required"
    t0 = time.time()
    r = analyze_resume(RESUME, JD)
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"ATS: {r['ats_score']} | Confidence: {r['confidence']['level']}")
    print(f"Feedback: {r['overall_feedback'][:100]}")
    print(f"Roadmap weeks: {len(r['learning_roadmap'])}")