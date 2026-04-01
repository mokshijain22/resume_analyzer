"""
gemini_analyzer.py — Structured recruiter-grade analyzer.
Uses a strict evidence-based prompt with consistency checks built in.
Parses flat JSON response into the nested structure the templates expect.
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
    "ML Engineer":       "Python, PyTorch/TensorFlow, scikit-learn, NLP, model deployment, Docker, AWS/GCP, MLflow, REST APIs, SQL, Git. Expected: production models, measurable accuracy, real datasets, end-to-end pipelines.",
    "Data Analyst":      "SQL, Python/R, pandas, Excel, Tableau/Power BI, statistics, A/B testing, dashboards. Expected: business impact metrics, stakeholder reports, measurable insights.",
    "Backend Developer": "Python/Node.js, REST APIs, PostgreSQL/MongoDB, Docker, CI/CD, Git, system design, auth, microservices. Expected: production APIs, deployed apps, unit tests.",
}


def analyze_resume(resume_text: str, jd_text: str = "", role: str = "") -> dict:
    mode       = "with_jd" if jd_text.strip() else "without_jd"
    target     = jd_text.strip()[:800] if mode == "with_jd" else MARKET_PROFILES.get(role, MARKET_PROFILES["ML Engineer"])
    resume     = resume_text.strip()[:2500]
    role_label = role or "ML Engineer"

    t0 = time.time()

    # ── Structured recruiter prompt ───────────────────────────────────────────
    prompt = f"""You are an expert technical recruiter and ATS evaluator.
You are given raw resume text and data about a candidate.

IMPORTANT RULES:
- You MUST ONLY use the provided resume text and data.
- DO NOT assume or hallucinate skills or projects.
- DO NOT give generic advice.
- Every statement MUST be backed by evidence from the resume.
- Maintain strict logical consistency between ATS score, percentile, and shortlist probability.
- Be analytical and realistic. Avoid motivational or vague language.
- No skill should appear in both matched AND missing lists.

INPUT DATA:
Resume Text: {resume}

Job Description / Market Profile ({role_label}):
{target}

STEP 1: PROJECT EXTRACTION (MANDATORY)
Extract ALL projects mentioned. Return each separately. Do NOT merge. Do NOT hallucinate.

STEP 2: ANALYSIS

1. OVERALL ASSESSMENT: 2-3 lines, evidence-based. Critical if weak, justified if strong.

2. SHORTLIST DECISION — derive from ATS score strictly:
   - ATS < 50 → shortlist ≤ 40%
   - ATS 50-70 → shortlist 40-70%
   - ATS > 70 → shortlist 60-90%
   Adjust down for missing critical skills. Give short reasoning.

3. SKILL GAP ANALYSIS: Exact missing skills vs JD. Explain impact briefly.

4. TOP 3 IMPROVEMENT ACTIONS — SPECIFIC and JD-aware:
   BAD: "Learn AWS"
   GOOD: "Missing AWS deployment required by JD — deploy your existing Flask/ML project on AWS EC2 or Lambda to demonstrate this skill."
   Each action must state: what is missing, why it matters, how to fix using their actual projects.

5. SCORE BREAKDOWN — weighted: skills 30% + experience 25% + projects 15% + tools 10% + resume quality 10% + proof of work 10%

6. EXPERIENCE DEPTH — for each detected skill: Beginner/Intermediate/Advanced with reason from resume evidence.

7. LEARNING ROADMAP — 4 weeks, execution-based, referencing their actual projects.

8. IMPROVED BULLET POINTS — 2 before/after pairs. Use metrics and action verbs.

9. COVER NOTE — 2 sentences referencing their actual projects and this specific role.

CONSISTENCY CHECK (apply before returning):
- No skill in both matched and missing
- Shortlist % must follow ATS rules above
- No generic statements

Return ONLY valid compact JSON. No markdown. No text before or after.
Keep every string value under 150 chars. Arrays max 4 items.

{{
  "projects": ["project name + one line description"],
  "assessment": "2-3 sentence evidence-based evaluation",
  "shortlist_percentage": <number 0-100>,
  "shortlist_reason": "short reasoning tied to ATS score and gaps",
  "skill_gaps": ["specific gap with impact explanation"],
  "actions": [
    "Action 1: what missing | why matters | how to fix using their actual projects",
    "Action 2: what missing | why matters | how to fix",
    "Action 3: what missing | why matters | how to fix"
  ],
  "confidence_level": "High or Medium or Low",
  "confidence_reason": "one sentence from resume evidence",
  "score_defensibility": "why this exact score with 2-3 specific evidence points",
  "skills_match": <0-30>,
  "experience_depth_score": <0-25>,
  "projects_quality_score": <0-15>,
  "tools_tech_stack_score": <0-10>,
  "resume_quality_score": <0-10>,
  "proof_of_work_score": <0-10>,
  "positive_contributors": ["evidence-backed positive reason"],
  "negative_contributors": ["evidence-backed penalty reason"],
  "projects_detected": <int>,
  "projects_with_metrics": <int>,
  "projects_with_deployment": <int>,
  "tools_detected": ["tool1", "tool2", "tool3"],
  "metrics_found": ["e.g. 94% accuracy"],
  "github_detected": <true or false>,
  "evidence_summary": "X projects found, Y deployed, Z with metrics — tools: list",
  "job_fit_score": <0-100>,
  "experience_level": "Fresher or Junior or Mid or Senior",
  "skills_depth": [
    {{"skill": "Python", "level": "Intermediate", "reason": "evidence from resume", "has_metrics": false}}
  ],
  "has_deployed_projects": <true or false>,
  "has_metrics": <true or false>,
  "project_details": [
    {{"name": "project", "score": <0-10>, "prod_score": <0-10>, "deployed": <bool>, "has_metrics": <bool>, "feedback": "honest specific feedback"}}
  ],
  "vs_average": "specific comparison to average candidate with numbers",
  "vs_top_10": "specific gap to top 10 percent candidates",
  "recruiter_feedback": ["decisive specific rejection/concern reason"],
  "risk_flags": ["specific flag with resume evidence"],
  "bullet_before_1": "original weak bullet",
  "bullet_after_1": "improved metric-driven version",
  "bullet_before_2": "original weak bullet",
  "bullet_after_2": "improved version",
  "week1_focus": "topic", "week1_task": "specific task using their actual project", "week1_outcome": "deliverable",
  "week2_focus": "topic", "week2_task": "specific task", "week2_outcome": "deliverable",
  "week3_focus": "topic", "week3_task": "specific task", "week3_outcome": "deliverable",
  "week4_focus": "topic", "week4_task": "specific task", "week4_outcome": "deliverable",
  "cover_note": "2 sentences referencing actual projects and specific role"
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
        print(f"[groq] {elapsed}ms {len(raw)} chars tokens={response.usage.total_tokens}", flush=True)

        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",          "", raw).strip()

        data = json.loads(raw)
        return _build_result(data, mode)

    except json.JSONDecodeError as e:
        print(f"[groq] JSON error at {e.pos}: {e.msg} — trying recovery", flush=True)
        try:
            partial = _recover_partial(raw)
            if partial:
                return _build_result(partial, mode)
        except Exception:
            pass
        return _fallback(mode, "JSON parse error — please retry")

    except Exception as e:
        print(f"[groq] error: {e}", flush=True)
        return _fallback(mode, str(e)[:80])


def _recover_partial(raw: str) -> dict | None:
    for suffix in ["}", '"}', '"}}']: 
        try:
            return json.loads(raw.rstrip() + suffix)
        except Exception:
            pass
    return None


def _build_result(d: dict, mode: str) -> dict:
    def sg(k, default):
        v = d.get(k, default)
        return v if v is not None else default

    def sgl(k):
        v = d.get(k, [])
        return v if isinstance(v, list) else []

    # ── Improved bullets ──────────────────────────────────────────────────────
    improved_bullets = []
    for i in range(1, 4):
        before = sg(f"bullet_before_{i}", "")
        after  = sg(f"bullet_after_{i}",  "")
        if before and after:
            improved_bullets.append(f"BEFORE: {before} | AFTER: {after}")
        elif after:
            improved_bullets.append(after)

    # ── Roadmap ───────────────────────────────────────────────────────────────
    roadmap = []
    for w in range(1, 5):
        focus   = sg(f"week{w}_focus",   "")
        task    = sg(f"week{w}_task",    "")
        outcome = sg(f"week{w}_outcome", "")
        if focus and task:
            roadmap.append({"week": w, "focus": focus, "task": task, "outcome": outcome})

    # ── Top 3 actions — from "actions" array ─────────────────────────────────
    top_3 = []
    raw_actions = sgl("actions")
    for i, action_str in enumerate(raw_actions[:3], 1):
        # Parse "What | Why | How" format if present
        parts = str(action_str).split("|")
        if len(parts) >= 3:
            top_3.append({
                "priority": i,
                "action":   parts[0].replace("Action", "").strip().lstrip("0123456789: "),
                "why":      parts[1].strip(),
                "how":      parts[2].strip(),
            })
        else:
            # Plain string — put it all in action field
            top_3.append({
                "priority": i,
                "action":   str(action_str)[:100],
                "why":      "",
                "how":      "",
            })

    # ── Project analysis ──────────────────────────────────────────────────────
    projects = []
    for p in sgl("project_details")[:5]:
        if isinstance(p, dict):
            projects.append({
                "name":                       str(p.get("name", "Project")),
                "score":                      float(p.get("score", 5)),
                "production_readiness_score": float(p.get("prod_score", 5)),
                "impact_score":               float(p.get("score", 5)),
                "has_deployment":             bool(p.get("deployed", False)),
                "has_metrics":                bool(p.get("has_metrics", False)),
                "real_world_relevance":       "Medium",
                "tech_stack_depth":           "Moderate",
                "feedback":                   str(p.get("feedback", "")),
            })

    # ── Skills depth ──────────────────────────────────────────────────────────
    skills_depth = []
    for s in sgl("skills_depth")[:8]:
        if isinstance(s, dict):
            skills_depth.append({
                "skill":       str(s.get("skill", "")),
                "level":       str(s.get("level", "Beginner")),
                "reason":      str(s.get("reason", "")),
                "has_metrics": bool(s.get("has_metrics", False)),
            })

    # ── Skill gaps → missing_skills (categorised by order) ───────────────────
    # The prompt returns a flat skill_gaps list; map first 2 to critical, rest to important
    skill_gaps = sgl("skill_gaps")
    # We don't overwrite matcher's categorised missing — this is used for display only

    ats = float(sg("ats_score", 50) if sg("ats_score", 0) else 50)

    # ── Extracted projects list for display ───────────────────────────────────
    extracted_projects = sgl("projects")

    return {
        "mode": mode,
        "extracted_projects": extracted_projects,
        "skill_gaps_display": skill_gaps,
        "confidence": {
            "level":  str(sg("confidence_level", "Medium")),
            "reason": str(sg("confidence_reason", "Analysis complete.")),
        },
        "ats_score":           ats,
        "score_defensibility": str(sg("score_defensibility", "")),
        "score_breakdown": {
            "skills_match":     float(sg("skills_match",             0)),
            "experience_depth": float(sg("experience_depth_score",   0)),
            "projects_quality": float(sg("projects_quality_score",   0)),
            "tools_tech_stack": float(sg("tools_tech_stack_score",   0)),
            "resume_quality":   float(sg("resume_quality_score",     0)),
            "proof_of_work":    float(sg("proof_of_work_score",      0)),
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
        "percentile":    50.0,   # overridden by matcher in app.py
        "matched_skills":  [],   # overridden by matcher
        "missing_skills":  {"critical": [], "important": [], "optional": []},  # overridden
        "skill_validation": [],
        "experience_analysis": {
            "total_experience_level": str(sg("experience_level", "Junior")),
            "skills_depth": skills_depth,
        },
        "proof_of_skill": {
            "has_deployed_projects":   bool(sg("has_deployed_projects", False)),
            "has_live_links":          bool(sg("github_detected",       False)),
            "has_metrics":             bool(sg("has_metrics",           False)),
            "has_real_world_datasets": False,
            "summary":                 str(sg("evidence_summary", "")),
        },
        "project_analysis": projects,
        "benchmark_comparison": {
            "vs_average_candidate":  str(sg("vs_average", "")),
            "vs_top_10_percent":     str(sg("vs_top_10",  "")),
            "shortlist_probability": int(sg("shortlist_percentage", 40)),
        },
        "recruiter_feedback":  sgl("recruiter_feedback"),
        "risk_flags":          sgl("risk_flags"),
        "radar_explanation":   {},
        "top_3_actions":       top_3,
        "overall_feedback":    str(sg("assessment", "")),
        "improved_bullets":    improved_bullets,
        "before_after_comparison": {
            "ats_score_before":      ats,
            "ats_score_after":       min(ats + 12, 95),
            "skill_coverage_before": 50.0,
            "skill_coverage_after":  65.0,
            "key_improvements":      [],
        },
        "learning_roadmap": roadmap,
        "cover_note":       str(sg("cover_note", "")),
    }


def _fallback(mode: str, detail: str = "") -> dict:
    return {
        "mode": mode, "extracted_projects": [], "skill_gaps_display": [],
        "confidence": {"level": "Low", "reason": f"Analysis failed: {detail}"},
        "ats_score": 0, "score_defensibility": "",
        "score_breakdown": {"skills_match":0,"experience_depth":0,"projects_quality":0,"tools_tech_stack":0,"resume_quality":0,"proof_of_work":0},
        "score_contributors": {"positive":[],"negative":[]},
        "evidence_summary": {"projects_detected":0,"projects_with_metrics":0,"projects_with_deployment":0,"tools_detected":[],"metrics_found":[],"links_detected":False,"github_detected":False,"summary":""},
        "job_fit_score":0,"percentile":0,"matched_skills":[],"missing_skills":{"critical":[],"important":[],"optional":[]},
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
    Mokshi Jain — B.Tech CSE AI/ML, Manipal University Jaipur, 2027.
    Python, PyTorch, scikit-learn, HuggingFace Transformers, Flask, pandas, numpy, Git.
    Projects:
    - Fake News Detector: BERT-based NLP classifier, 94% accuracy, deployed on Render.
    - MNIST Digit Recognizer: CNN in PyTorch, 99.1% test accuracy.
    - Real Estate Price Predictor: Ridge regression scikit-learn, deployed on Render.
    GitHub: github.com/mokshijain
    """
    JD = "ML Engineer — Python, PyTorch, NLP, HuggingFace, Docker, AWS, MLflow required."
    t0 = time.time()
    r = analyze_resume(RESUME, JD)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Confidence   : {r['confidence']['level']} — {r['confidence']['reason']}")
    print(f"Assessment   : {r['overall_feedback'][:120]}")
    print(f"Projects     : {r['extracted_projects']}")
    print(f"Skill gaps   : {r['skill_gaps_display']}")
    print(f"Top 3 actions: {[a['action'] for a in r['top_3_actions']]}")
    print(f"Roadmap      : {len(r['learning_roadmap'])} weeks")