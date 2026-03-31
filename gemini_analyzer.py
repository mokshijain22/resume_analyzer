"""
gemini_analyzer.py — Performance + intelligence upgraded analyzer.
Optimizations:
  - Prompt trimmed to essential fields only (faster LLM response)
  - Hard max_tokens cap (1800) for speed
  - Temperature 0.4 (more deterministic, less rambling)
  - All outputs must be evidence-backed — strict prompt instructions
  - _sanitize() provides safe defaults for every field
  - _fallback() used on any error
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
    "ML Engineer":       "Python, PyTorch/TensorFlow, scikit-learn, NLP, computer vision, model deployment, Docker, AWS/GCP, MLflow, REST APIs, SQL, Git. Expected: production models, measurable accuracy, real datasets, end-to-end pipelines.",
    "Data Analyst":      "SQL, Python/R, pandas, Excel, Tableau/Power BI, statistics, A/B testing, dashboards. Expected: business impact metrics, stakeholder reports, measurable insights.",
    "Backend Developer": "Python/Node.js, REST APIs, PostgreSQL/MongoDB, Docker, CI/CD, Git, system design, auth, microservices. Expected: production APIs, deployed apps, unit tests.",
}


def analyze_resume(resume_text: str, jd_text: str = "", role: str = "") -> dict:
    mode   = "with_jd" if jd_text.strip() else "without_jd"
    target = jd_text.strip()[:1200] if mode == "with_jd" else MARKET_PROFILES.get(role, MARKET_PROFILES["ML Engineer"])
    resume = resume_text.strip()[:2800]
    role_label = role or "ML Engineer"

    t0 = time.time()

    # ── Compact prompt — removes verbose instructions, keeps strict rules ─────
    prompt = f"""Senior hiring manager + ATS expert. Analyze this resume. Every output MUST reference specific resume evidence. Zero generic statements.

RESUME:
{resume}

{"JOB DESCRIPTION" if mode=="with_jd" else f"ROLE PROFILE ({role_label})"}:
{target}

Rules:
- BAD: "Strong profile" | GOOD: "3 projects found, 1 deployed, 2 with metrics"
- BAD: "Could improve Docker" | GOOD: "Docker absent — required for this role, missing from all 3 projects"
- BAD: "Good Python skills" | GOOD: "Python used in 3 projects (NLP, regression, CNN) — Intermediate level"
- Recruiter feedback must be decisive: "NOT ready for X role because..." not "could improve"
- top_3_actions must reference candidate's actual project names from resume
- learning_roadmap tasks must reference their actual existing projects

Return ONLY valid JSON (no markdown, no backticks):

{{
  "confidence": {{"level":"High/Medium/Low","reason":"specific reason from resume evidence"}},
  "ats_score": <0-100 weighted: skills 30% experience 25% projects 15% tools 10% resume_quality 10% proof 10%>,
  "score_defensibility": "Why this exact score: list specific +/- contributors with evidence",
  "score_breakdown": {{"skills_match":<0-30>,"experience_depth":<0-25>,"projects_quality":<0-15>,"tools_tech_stack":<0-10>,"resume_quality":<0-10>,"proof_of_work":<0-10>}},
  "score_contributors": {{"positive":["evidence-backed +reason"],"negative":["evidence-backed -reason"]}},
  "evidence_summary": {{
    "projects_detected":<int>,"projects_with_metrics":<int>,"projects_with_deployment":<int>,
    "tools_detected":["actual tools found"],"metrics_found":["exact metrics like 94% accuracy"],
    "links_detected":<bool>,"github_detected":<bool>,
    "summary":"e.g. 3 projects: Fake News Detector (deployed, 94% acc), MNIST (99.1%), Real Estate (deployed)"
  }},
  "job_fit_score":<0-100>,
  "percentile":<0-100>,
  "matched_skills":["skills with project evidence"],
  "missing_skills":{{"critical":["must-have missing"],"important":["important missing"],"optional":["nice to have"]}},
  "skill_validation":[
    {{"skill":"name","listed":<bool>,"evidenced":<bool>,"proof_strength":"Strong/Moderate/Weak/None","reason":"specific evidence from resume"}}
  ],
  "experience_analysis":{{
    "total_experience_level":"Fresher/Junior/Mid/Senior",
    "skills_depth":[{{"skill":"name","level":"Beginner/Intermediate/Advanced","reason":"evidence-based: complexity, scale, ownership, impact","has_metrics":<bool>}}]
  }},
  "proof_of_skill":{{
    "has_deployed_projects":<bool>,"has_live_links":<bool>,"has_metrics":<bool>,"has_real_world_datasets":<bool>,
    "summary":"evidence count: X projects, Y deployed, Z with metrics"
  }},
  "project_analysis":[
    {{"name":"project name","score":<0-10>,"production_readiness_score":<0-10>,"impact_score":<0-10>,
      "has_deployment":<bool>,"has_metrics":<bool>,"real_world_relevance":"Low/Medium/High",
      "tech_stack_depth":"Shallow/Moderate/Deep","feedback":"specific honest assessment"}}
  ],
  "benchmark_comparison":{{
    "vs_average_candidate":"specific comparison with numbers",
    "vs_top_10_percent":"specific gap with concrete missing items",
    "shortlist_probability":<0-100>
  }},
  "recruiter_feedback":["decisive specific rejection reason referencing actual resume gaps — no soft language"],
  "risk_flags":["specific flag with evidence, e.g. 0 of 3 projects have deployment"],
  "radar_explanation":{{
    "Programming Languages":"how calculated e.g. Python confirmed in 3 projects, no other lang detected",
    "ML / AI":"evidence-based",
    "Frameworks & Libraries":"evidence-based",
    "Data & Databases":"evidence-based",
    "Cloud & DevOps":"evidence-based",
    "Soft Skills":"evidence-based"
  }},
  "top_3_actions":[
    {{"priority":1,"action":"specific title","why":"evidence-backed reason from resume","how":"execution step using their actual project names"}}
  ],
  "overall_feedback":"3-4 sentences, evidence-based, references specific projects and gaps",
  "improved_bullets":["BEFORE: original weak bullet | AFTER: metric-driven ATS-optimized version"],
  "before_after_comparison":{{
    "ats_score_before":<current>,"ats_score_after":<projected after top 3 actions, max +20pts>,
    "skill_coverage_before":<0-100>,"skill_coverage_after":<0-100>,
    "key_improvements":["specific improvement driving the delta"]
  }},
  "learning_roadmap":[
    {{"week":<int>,"focus":"topic","task":"execution step referencing their actual projects","outcome":"concrete deliverable"}}
  ],
  "cover_note":"2-3 sentences referencing their actual projects and the specific role"
}}"""

    try:
        response = _CLIENT.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": prompt}],
            max_tokens  = 1800,
            temperature = 0.4,
        )
        elapsed = round((time.time() - t0) * 1000)
        print(f"[groq] Response in {elapsed}ms, tokens={response.usage.total_tokens}", flush=True)

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$",         "", raw).strip()

        data = json.loads(raw)
        print(f"[groq] JSON parsed OK", flush=True)
        return _sanitize(data, mode)

    except json.JSONDecodeError as e:
        print(f"[groq] JSON parse error: {e}", flush=True)
        return _fallback(mode, f"JSON error: {str(e)[:80]}")
    except Exception as e:
        print(f"[groq] API error: {e}", flush=True)
        return _fallback(mode, str(e)[:100])


def _sanitize(d: dict, mode: str) -> dict:
    def sg(k, v): return d.get(k, v)

    def safe_list(v):
        if isinstance(v, list): return v
        if isinstance(v, str):  return [s.strip() for s in v.split("\n") if s.strip()]
        return []

    bullets = safe_list(sg("improved_bullets", []))
    bullets = [str(b) for b in bullets if str(b).strip()]

    ba = sg("before_after_comparison", {})
    ats = float(sg("ats_score", 50))

    missing = sg("missing_skills", {})
    if not isinstance(missing, dict):
        missing = {"critical": [], "important": [], "optional": []}

    conf = sg("confidence", {"level": "Medium", "reason": "Analysis complete."})
    if not isinstance(conf, dict):
        conf = {"level": "Medium", "reason": str(conf)}

    ev = sg("evidence_summary", {})

    return {
        "mode":               mode,
        "confidence":         conf,
        "ats_score":          ats,
        "score_defensibility": str(sg("score_defensibility", "")),
        "score_breakdown": {
            "skills_match":     float(sg("score_breakdown", {}).get("skills_match",     0)),
            "experience_depth": float(sg("score_breakdown", {}).get("experience_depth", 0)),
            "projects_quality": float(sg("score_breakdown", {}).get("projects_quality", 0)),
            "tools_tech_stack": float(sg("score_breakdown", {}).get("tools_tech_stack", 0)),
            "resume_quality":   float(sg("score_breakdown", {}).get("resume_quality",   0)),
            "proof_of_work":    float(sg("score_breakdown", {}).get("proof_of_work",    0)),
        },
        "score_contributors": sg("score_contributors", {"positive": [], "negative": []}),
        "evidence_summary": {
            "projects_detected":        int(ev.get("projects_detected",        0)),
            "projects_with_metrics":    int(ev.get("projects_with_metrics",    0)),
            "projects_with_deployment": int(ev.get("projects_with_deployment", 0)),
            "tools_detected":           list(ev.get("tools_detected",          [])),
            "metrics_found":            list(ev.get("metrics_found",           [])),
            "links_detected":           bool(ev.get("links_detected",          False)),
            "github_detected":          bool(ev.get("github_detected",         False)),
            "summary":                  str(ev.get("summary",                  "")),
        },
        "job_fit_score":   float(sg("job_fit_score", 50)),
        "percentile":      float(sg("percentile",    50)),
        "matched_skills":  safe_list(sg("matched_skills", [])),
        "missing_skills":  {
            "critical":  list(missing.get("critical",  [])),
            "important": list(missing.get("important", [])),
            "optional":  list(missing.get("optional",  [])),
        },
        "skill_validation":   safe_list(sg("skill_validation", [])),
        "experience_analysis": {
            "total_experience_level": str(sg("experience_analysis", {}).get("total_experience_level", "Junior")),
            "skills_depth":           safe_list(sg("experience_analysis", {}).get("skills_depth", [])),
        },
        "proof_of_skill": sg("proof_of_skill", {
            "has_deployed_projects": False, "has_live_links": False,
            "has_metrics": False,           "has_real_world_datasets": False,
            "summary": "Not assessed.",
        }),
        "project_analysis":   list(sg("project_analysis", []))[:5],
        "benchmark_comparison": sg("benchmark_comparison", {
            "vs_average_candidate": "N/A", "vs_top_10_percent": "N/A", "shortlist_probability": 40,
        }),
        "recruiter_feedback": safe_list(sg("recruiter_feedback", [])),
        "risk_flags":         safe_list(sg("risk_flags",         [])),
        "radar_explanation":  sg("radar_explanation", {}),
        "top_3_actions":      safe_list(sg("top_3_actions", []))[:3],
        "overall_feedback":   str(sg("overall_feedback", "")),
        "improved_bullets":   bullets[:6],
        "before_after_comparison": {
            "ats_score_before":      float(ba.get("ats_score_before",      ats)),
            "ats_score_after":       min(float(ba.get("ats_score_after",   ats + 12)), 97),
            "skill_coverage_before": float(ba.get("skill_coverage_before", 50)),
            "skill_coverage_after":  float(ba.get("skill_coverage_after",  65)),
            "key_improvements":      list(ba.get("key_improvements",       [])),
        },
        "learning_roadmap": safe_list(sg("learning_roadmap", []))[:6],
        "cover_note":       str(sg("cover_note", "")),
    }


def _fallback(mode: str, detail: str = "") -> dict:
    return {
        "mode": mode,
        "confidence": {"level": "Low", "reason": "Analysis failed — retry."},
        "ats_score": 0, "score_defensibility": "",
        "score_breakdown": {"skills_match":0,"experience_depth":0,"projects_quality":0,"tools_tech_stack":0,"resume_quality":0,"proof_of_work":0},
        "score_contributors": {"positive":[],"negative":[]},
        "evidence_summary": {"projects_detected":0,"projects_with_metrics":0,"projects_with_deployment":0,"tools_detected":[],"metrics_found":[],"links_detected":False,"github_detected":False,"summary":""},
        "job_fit_score":0,"percentile":0,"matched_skills":[],"missing_skills":{"critical":[],"important":[],"optional":[]},
        "skill_validation":[],"experience_analysis":{"total_experience_level":"Unknown","skills_depth":[]},
        "proof_of_skill":{"has_deployed_projects":False,"has_live_links":False,"has_metrics":False,"has_real_world_datasets":False,"summary":"Failed."},
        "project_analysis":[],"benchmark_comparison":{"vs_average_candidate":"N/A","vs_top_10_percent":"N/A","shortlist_probability":0},
        "recruiter_feedback":[],"risk_flags":[],"radar_explanation":{},"top_3_actions":[],
        "overall_feedback":f"AI analysis temporarily unavailable. {detail}",
        "improved_bullets":[],"before_after_comparison":{"ats_score_before":0,"ats_score_after":0,"skill_coverage_before":0,"skill_coverage_after":0,"key_improvements":[]},
        "learning_roadmap":[],"cover_note":"",
    }


if __name__ == "__main__":
    RESUME = """
    Mokshi Jain — B.Tech CSE (AI/ML), MUJ 2027.
    Python, PyTorch, scikit-learn, HuggingFace, Flask, pandas, numpy, Git.
    Fake News Detector: BERT-based NLP, 94% accuracy, deployed on Render.
    MNIST Digit Recognizer: CNN, PyTorch, 99.1% test accuracy.
    Real Estate Price Predictor: Ridge regression, scikit-learn, deployed Render.
    GitHub: github.com/mokshijain
    """
    JD = "ML Engineer — Python, PyTorch, NLP, HuggingFace, Docker, AWS, MLflow."
    t0 = time.time()
    r = analyze_resume(RESUME, JD)
    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"Confidence     : {r['confidence']['level']} — {r['confidence']['reason']}")
    print(f"ATS Score      : {r['ats_score']}")
    print(f"Defensibility  : {r['score_defensibility'][:120]}...")
    print(f"Evidence       : {r['evidence_summary']['summary']}")
    print(f"Top 3 Actions  : {[a['action'] for a in r['top_3_actions']]}")