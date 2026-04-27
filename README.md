# Resume Analyzer

Resume Analyzer is a Flask web app that scores resumes against a job description
or role template, then returns ATS-focused feedback with evidence-based
recommendations.

## Features

- Upload PDF or DOCX resumes from the web UI.
- Compute ATS, job-fit, and percentile scores from embedding + keyword signals.
- Generate structured recruiter-style feedback using Groq.
- Show matched/missing skills, extracted projects, and radar-chart visualization.
- Apply post-processing validation to catch score/feedback contradictions.
- Cache analysis results in memory to speed up repeated runs.

## Tech Stack

- Python 3.11+
- Flask (web server and templating)
- sentence-transformers + scikit-learn (similarity/scoring)
- Groq API (LLM-generated feedback)
- Plotly (chart generation)

## Project Structure

- `app.py` - Flask routes, orchestration, rendering.
- `resume_parser.py` / `parser.py` - text extraction from uploaded files.
- `matcher.py` - ATS scoring and skill gap computation.
- `gemini_analyzer.py` - recruiter-style analysis via Groq.
- `validator.py` - consistency checks and auto-fixes on AI output.
- `charts.py` - radar chart JSON generation.
- `cache.py` - in-memory cache utility.
- `templates/` - UI templates.
- `speed_test.py` - performance diagnostics for import/runtime hotspots.

## Setup

1. Clone and enter the repository.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Configure environment variables.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Run Locally

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Code Quality & Validation

Run lint and syntax checks:

```bash
ruff check .
python -m compileall -q .
```

Optional performance diagnostic:

```bash
python speed_test.py
```

## Notes

- Maximum upload size is 5 MB.
- Supported file types are `.pdf` and `.docx`.
- Cache is in-memory and process-local; for production, replace with Redis.
