"""
validator.py — Output validation layer.
Catches logical contradictions BEFORE results reach the user.
Examples:
  - AWS detected in resume but "no cloud skills" in feedback → FIXED
  - ats_score 85 but percentile 20 → FIXED
  - skill listed as matched AND missing → FIXED
  - score_breakdown components don't sum to ats_score range → WARNED
"""
import re


def validate_and_fix(ai: dict, match: dict) -> tuple[dict, list[str]]:
    """
    Returns (fixed_ai_dict, list_of_fixes_applied).
    Fixes logical contradictions in-place.
    """
    fixes: list[str] = []

    # ── 1. Matched skills must not appear in missing skills ───────────────────
    matched_lower = {s.lower() for s in match.get("matched_keywords", [])}
    for tier in ["critical", "important", "optional"]:
        tier_skills = ai["missing_skills"].get(tier, [])
        cleaned = [s for s in tier_skills if s.lower() not in matched_lower]
        if len(cleaned) != len(tier_skills):
            removed = [s for s in tier_skills if s.lower() in matched_lower]
            fixes.append(f"Removed from missing.{tier} (already matched): {removed}")
            ai["missing_skills"][tier] = cleaned

    # ── 2. Recruiter feedback must not contradict matched skills ──────────────
    rf = ai.get("recruiter_feedback", [])
    clean_rf: list[str] = []
    contradiction_patterns = [
        (r"no python", "python"),
        (r"no aws", "aws"),
        (r"no docker", "docker"),
        (r"no (deep learning|ml|machine learning)", "deep learning"),
        (r"no (nlp|natural language)", "nlp"),
        (r"no flask|no api", "flask"),
        (r"no cloud", "aws gcp azure"),
    ]
    for reason in rf:
        contradiction_found = False
        for pattern, skill_check in contradiction_patterns:
            if re.search(pattern, reason.lower()):
                skills_to_check = skill_check.split()
                if any(s in matched_lower for s in skills_to_check):
                    fixes.append(f"Removed contradicting recruiter_feedback: '{reason[:60]}...'")
                    contradiction_found = True
                    break
        if not contradiction_found:
            clean_rf.append(reason)
    ai["recruiter_feedback"] = clean_rf

    # ── 3. risk_flags must not contradict evidence_summary ───────────────────
    ev = ai.get("evidence_summary", {})
    rf2: list[str] = []
    for flag in ai.get("risk_flags", []):
        fl = flag.lower()
        if "no github" in fl and ev.get("github_detected"):
            fixes.append(f"Removed risk_flag 'no github' — github_detected=True")
            continue
        if "no metrics" in fl and ev.get("projects_with_metrics", 0) > 0:
            fixes.append(f"Removed risk_flag 'no metrics' — {ev['projects_with_metrics']} projects have metrics")
            continue
        if "no deployed" in fl and ev.get("projects_with_deployment", 0) > 0:
            fixes.append(f"Removed risk_flag 'no deployed' — {ev['projects_with_deployment']} deployed projects found")
            continue
        rf2.append(flag)
    ai["risk_flags"] = rf2

    # ── 4. Percentile must be consistent with ats_score ──────────────────────
    ats = ai.get("ats_score", 50)
    pct = ai.get("percentile", 50)
    expected_min_pct = max(0, ats - 20)
    expected_max_pct = min(100, ats + 15)
    if not (expected_min_pct <= pct <= expected_max_pct):
        corrected = int(ats * 0.9)
        fixes.append(f"Corrected percentile {pct} → {corrected} (inconsistent with ats_score {ats})")
        ai["percentile"] = corrected

    # ── 5. Score breakdown components must be internally consistent ───────────
    bd = ai.get("score_breakdown", {})
    component_sum = sum([
        bd.get("skills_match", 0),
        bd.get("experience_depth", 0),
        bd.get("projects_quality", 0),
        bd.get("tools_tech_stack", 0),
        bd.get("resume_quality", 0),
        bd.get("proof_of_work", 0),
    ])
    if component_sum > 0:
        scaled_sum = (component_sum / 100) * 100
        deviation = abs(scaled_sum - ats)
        if deviation > 25:
            fixes.append(f"score_breakdown sum {component_sum} deviates significantly from ats_score {ats} — may be miscalibrated")

    # ── 6. Job fit score must be ≤ ats_score (it's a stricter version) ────────
    jf = ai.get("job_fit_score", 50)
    if jf > ats:
        fixed_jf = round(ats * 0.92, 1)
        fixes.append(f"Corrected job_fit_score {jf} → {fixed_jf} (cannot exceed ats_score {ats})")
        ai["job_fit_score"] = fixed_jf

    # ── 7. Before-after ats scores must be realistic ─────────────────────────
    ba = ai.get("before_after_comparison", {})
    before = ba.get("ats_score_before", ats)
    after  = ba.get("ats_score_after", ats + 10)
    delta  = after - before
    if delta > 30:
        fixed_after = round(before + 20, 1)
        fixes.append(f"Capped before_after delta {delta:.0f}pts → 20pts (unrealistic improvement claim)")
        ai["before_after_comparison"]["ats_score_after"] = fixed_after
    if after > 98:
        ai["before_after_comparison"]["ats_score_after"] = 95
        fixes.append("Capped ats_score_after at 95 (100% is unrealistic)")

    # ── 8. Skill validation must not mark evidenced=True for missing skills ───
    sv = ai.get("skill_validation", [])
    for item in sv:
        skill_lower = item.get("skill", "").lower()
        if skill_lower in matched_lower and not item.get("evidenced"):
            item["evidenced"] = True
            item["proof_strength"] = item.get("proof_strength", "Moderate")
            fixes.append(f"Corrected skill_validation: {item['skill']} is matched → evidenced=True")
        if skill_lower not in matched_lower and item.get("proof_strength") == "Strong":
            item["proof_strength"] = "Moderate"
            fixes.append(f"Downgraded proof_strength for {item['skill']} — not in matched_keywords")

    if fixes:
        print(f"[validator] Applied {len(fixes)} consistency fixes:", flush=True)
        for f in fixes:
            print(f"  → {f}", flush=True)
    else:
        print("[validator] No contradictions found.", flush=True)

    return ai, fixes