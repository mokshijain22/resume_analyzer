# AI Career Suite

AI-powered ATS resume analyzer and career intelligence platform that evaluates resumes against job roles using NLP similarity scoring, recruiter-style analysis, and AI-generated improvement recommendations.

---

## Live Demo

Frontend: https://resume-analyzer.impressment.in/

---

## Overview

AI Career Suite is a full-stack AI application that analyzes resumes against job descriptions or role templates and generates ATS-focused feedback with actionable career insights.

The platform combines TF-IDF similarity scoring, structured skill-gap analysis, recruiter-style LLM feedback, validation layers, caching, and visual analytics to create a practical AI-powered resume evaluation workflow.

---

## Features

* Upload PDF or DOCX resumes
* ATS compatibility scoring
* Role-specific resume matching
* Skill-gap detection
* Recruiter-style AI feedback
* Radar chart visualizations
* Project and experience extraction
* Groq-powered analysis generation
* Validation layer for contradiction checking
* Cached responses for faster repeated analysis
* Clean Flask-based web interface

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript
* Jinja Templates

### Backend

* Flask
* Python

### AI / NLP

* TF-IDF
* scikit-learn
* Groq API
* NLP parsing pipelines

### Visualization

* Plotly

### Deployment

* Docker
* Render

---

## System Flow

```text id="m9a6pq"
Resume Upload
   ↓
Text Extraction
   ↓
Skill Extraction
   ↓
TF-IDF Similarity Matching
   ↓
ATS Score Generation
   ↓
Groq Feedback Generation
   ↓
Validation Layer
   ↓
Final Career Report
```

---

## Why Lightweight NLP Instead of Heavy Models?

This project focuses on practical deployment efficiency and explainable scoring pipelines.

Benefits:

* Faster analysis
* Lower memory usage
* Better deployment compatibility
* More transparent scoring logic
* Stable behavior for repeated evaluations

---

## Project Structure

```text id="53g7f1"
resume-analyzer/
├── app.py
├── matcher.py
├── parser.py
├── resume_parser.py
├── validator.py
├── gemini_analyzer.py
├── charts.py
├── cache.py
├── project_extractor.py
├── templates/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Core Components

### matcher.py

Handles ATS scoring, TF-IDF similarity computation, and skill matching.

### validator.py

Applies post-processing checks to reduce contradictions and improve consistency in AI-generated feedback.

### gemini_analyzer.py

Generates recruiter-style recommendations and resume insights using Groq LLM APIs.

### cache.py

Implements in-memory caching for repeated analysis requests.

### charts.py

Creates radar-chart visualizations for role-fit analysis.

---

## Local Setup

### Clone Repository

```bash id="1m3r9o"
git clone https://github.com/mokshijain22/resume_analyzer.git
cd resume_analyzer
```

---

### Create Virtual Environment

```bash id="4r31do"
python -m venv .venv
```

#### Windows

```bash id="6m45f8"
.venv\Scripts\activate
```

#### Linux / macOS

```bash id="e2ah6n"
source .venv/bin/activate
```

---

### Install Dependencies

```bash id="m71jpl"
pip install -r requirements.txt
```

---

### Configure Environment Variables

Create a `.env` file:

```env id="bl4g2s"
GROQ_API_KEY=your_groq_api_key
```

---

### Run Locally

```bash id="e7f8ad"
python app.py
```

Open:

```text id="8fqqi7"
http://127.0.0.1:5000
```

---

## Environment Variables

```env id="pmq3h6"
GROQ_API_KEY=your_groq_api_key
```

---

## Key Learnings

* AI feedback systems require validation layers for consistency
* Structured scoring improves trust in generated recommendations
* Lightweight NLP pipelines can outperform heavier systems for practical deployment workflows
* Combining deterministic scoring with LLM reasoning creates better user experiences

---

## Future Improvements

* Multi-role resume benchmarking
* Resume history tracking
* Authentication system
* Persistent database storage
* PDF report export improvements
* Real-time analytics dashboard
* Vector-based semantic matching

---

## Notes

* Maximum upload size: 5 MB
* Supported formats: PDF and DOCX
* Current cache is in-memory and process-local
* Redis can be used for scalable production caching

---

## Author

Mokshi Jain
AI/ML Engineering Student

GitHub: https://github.com/mokshijain22


---

