"""
Run this from D:\resume-analyzer\ to find exactly where the slowdown is.
Command: python speed_test.py
"""
import time
import os
from dotenv import load_dotenv
load_dotenv()

SAMPLE_RESUME = """
Python developer with experience in machine learning, deep learning, and NLP.
Worked with TensorFlow, PyTorch, scikit-learn, and HuggingFace transformers.
Built REST APIs using Flask and FastAPI. Deployed models on AWS and Docker.
Proficient in SQL, pandas, numpy. Used Git and GitHub for version control.
Strong analytical and communication skills. Experience with agile teams.
"""

SAMPLE_JD = """
We are looking for an ML Engineer with Python, deep learning, and NLP experience.
Must know PyTorch or TensorFlow, HuggingFace, and LLM fine-tuning.
Experience with Flask or FastAPI for model serving. Docker and AWS required.
SQL and data pipeline experience is a plus. Strong communication skills essential.
"""

print("=" * 55)
print("  ResumeAI Speed Diagnostics")
print("=" * 55)

# ── Test 1: parser ─────────────────────────────────────────
print("\n[1/4] Testing parser.py ...")
t = time.time()
from parser import extract_text
print(f"      Import done in {time.time()-t:.2f}s")
# skip actual parse since we have no file, just measure import

# ── Test 2: sentence-transformer (the usual suspect) ───────
print("\n[2/4] Testing matcher.py (sentence-transformers load) ...")
t = time.time()
from matcher import compute_ats_score
elapsed = time.time() - t
print(f"      Import done in {elapsed:.2f}s  {'<-- SLOW HERE' if elapsed > 10 else 'OK'}")

t = time.time()
result = compute_ats_score(SAMPLE_RESUME, SAMPLE_JD)
elapsed = time.time() - t
print(f"      compute_ats_score() done in {elapsed:.2f}s  {'<-- SLOW HERE' if elapsed > 5 else 'OK'}")
print(f"      ATS score: {result.get('ats_score', 'N/A')}")

# ── Test 3: Groq API call ───────────────────────────────────
print("\n[3/4] Testing gemini_analyzer.py (Groq API call) ...")
t = time.time()
from gemini_analyzer import analyze_resume
elapsed = time.time() - t
print(f"      Import done in {elapsed:.2f}s")

t = time.time()
try:
    feedback = analyze_resume(SAMPLE_RESUME, SAMPLE_JD)
    elapsed = time.time() - t
    print(f"      analyze_resume() done in {elapsed:.2f}s  {'<-- SLOW HERE' if elapsed > 15 else 'OK'}")
    print(f"      Keys returned: {list(feedback.keys())}")
    fb_text = feedback.get('overall_feedback','')
    print(f"      Feedback preview: {fb_text[:80]}...")
except Exception as e:
    elapsed = time.time() - t
    print(f"      FAILED after {elapsed:.2f}s: {e}")

# ── Test 4: charts ─────────────────────────────────────────
print("\n[4/4] Testing charts.py ...")
t = time.time()
from charts import generate_radar_chart
elapsed = time.time() - t
print(f"      Import done in {elapsed:.2f}s")

t = time.time()
chart = generate_radar_chart(SAMPLE_RESUME, SAMPLE_JD)
elapsed = time.time() - t
print(f"      generate_radar_chart() done in {elapsed:.2f}s  {'<-- SLOW HERE' if elapsed > 5 else 'OK'}")

print("\n" + "=" * 55)
print("  Done. Share the output above so we can fix the slow step.")
print("=" * 55)