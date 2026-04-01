import re

def extract_projects(text):
    projects = []

    # Extract PROJECTS section
    match = re.search(r"(projects|PROJECTS)(.*?)(education|skills|experience|$)", text, re.DOTALL)
    
    if match:
        section = match.group(2)
    else:
        section = text

    # 🔥 THIS FIXES YOUR ERROR
    lines = section.split("\n")

    for line in lines:
        line = line.strip()

        # Skip short lines
        if len(line) < 15:
            continue

        # Skip bullet points
        if line.startswith(("•", "-", "*")):
            continue

        # Skip links / noise
        if any(x in line.lower() for x in ["http", "github", "live:", "www", "@"]):
            continue

        # Only structured project titles
        if "|" in line:
            projects.append(line)

    # remove duplicates + limit
    projects = list(dict.fromkeys(projects))[:5]

    return projects