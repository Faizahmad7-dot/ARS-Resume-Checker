"""ATS Resume Checker - Streamlit + Gemini Flash.

Upload a resume (PDF or DOCX), optionally paste a job description, and get:
  * an overall ATS score (0-100) with a category breakdown
  * strengths, problems and concrete improvements
  * missing keywords (when a job description is given)
"""

import io
import json
import os
import re

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# The first model is tried first; if it is unavailable the next one is used.
# Override with GEMINI_MODEL in .streamlit/secrets.toml or an env variable.
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"]
MAX_FILE_MB = 5
MAX_CHARS = 20000  # resume text sent to the model

# Weights used to build the overall score (must add up to 100).
CATEGORIES = {
    "keywords": ("Keywords & skills match", 30),
    "formatting": ("Formatting & ATS readability", 20),
    "experience": ("Experience & achievements", 20),
    "sections": ("Structure & required sections", 15),
    "language": ("Language & clarity", 15),
}

PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and resume coach.
Analyse the resume below{jd_clause}. Be honest and strict; do not inflate scores.

Return ONLY a JSON object with exactly this structure:
{{
  "scores": {{
    "keywords": <integer 0-100>,
    "formatting": <integer 0-100>,
    "experience": <integer 0-100>,
    "sections": <integer 0-100>,
    "language": <integer 0-100>
  }},
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<short point>", ...],
  "problems": ["<short point>", ...],
  "improvements": [
    {{"priority": "High" | "Medium" | "Low",
      "issue": "<what is wrong>",
      "fix": "<specific action, with a rewritten example where useful>"}}
  ],
  "missing_keywords": ["<keyword>", ...],
  "found_keywords": ["<keyword>", ...]
}}

Scoring guide:
- keywords: relevant hard/soft skills and role keywords{jd_score_note}
- formatting: plain single-column layout, standard headings, no tables/graphics, parseable text
- experience: quantified achievements, action verbs, relevant impact
- sections: contact info, summary, education, experience, skills, projects
- language: grammar, concision, consistency, tense, no filler

Give 3-6 strengths, 3-8 problems, 5-10 improvements (highest priority first).
If no job description was given, base keywords on the role the resume appears to target
and leave "missing_keywords" for commonly expected keywords in that role.

{jd_block}RESUME:
\"\"\"
{resume}
\"\"\"
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    """Read a value from Streamlit secrets, then env variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file present
        pass
    return os.environ.get(name, default)


def extract_text(uploaded_file) -> str:
    """Extract plain text from a PDF or DOCX upload."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        # Include table text too (and note: tables can hurt ATS parsing).
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts).strip()

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")


def parse_json(text: str) -> dict:
    """Parse model output as JSON, tolerating code fences or extra text."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def clamp(value, low=0, high=100) -> int:
    try:
        return max(low, min(high, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def as_list(value) -> list:
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


def normalise(result: dict) -> dict:
    """Validate the model output and compute the weighted overall score."""
    raw_scores = result.get("scores") if isinstance(result.get("scores"), dict) else {}
    scores = {key: clamp(raw_scores.get(key, 0)) for key in CATEGORIES}
    overall = clamp(sum(scores[k] * CATEGORIES[k][1] for k in CATEGORIES) / 100)

    improvements = []
    for item in result.get("improvements", []) if isinstance(result.get("improvements"), list) else []:
        if isinstance(item, dict):
            priority = str(item.get("priority", "Medium")).title()
            if priority not in ("High", "Medium", "Low"):
                priority = "Medium"
            improvements.append(
                {
                    "priority": priority,
                    "issue": str(item.get("issue", "")).strip(),
                    "fix": str(item.get("fix", "")).strip(),
                }
            )
    order = {"High": 0, "Medium": 1, "Low": 2}
    improvements.sort(key=lambda i: order[i["priority"]])

    return {
        "overall": overall,
        "scores": scores,
        "summary": str(result.get("summary", "")).strip(),
        "strengths": as_list(result.get("strengths")),
        "problems": as_list(result.get("problems")),
        "improvements": improvements,
        "missing_keywords": as_list(result.get("missing_keywords")),
        "found_keywords": as_list(result.get("found_keywords")),
    }


def analyse_resume(resume_text: str, job_description: str, api_key: str) -> dict:
    """Send the resume to Gemini and return the normalised analysis."""
    client = genai.Client(api_key=api_key)

    jd = job_description.strip()
    prompt = PROMPT.format(
        jd_clause=" against the job description provided" if jd else "",
        jd_score_note=" (matched against the job description)" if jd else "",
        jd_block=f'JOB DESCRIPTION:\n"""\n{jd[:8000]}\n"""\n\n' if jd else "",
        resume=resume_text[:MAX_CHARS],
    )
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )

    custom = get_secret("GEMINI_MODEL")
    models = ([custom] if custom else []) + [m for m in DEFAULT_MODELS if m != custom]

    last_error = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            return normalise(parse_json(response.text))
        except json.JSONDecodeError as exc:
            last_error = exc  # malformed JSON: try the next model
        except Exception as exc:
            message = str(exc).lower()
            # Wrong/unavailable model name -> try next; anything else -> stop.
            if "404" in message or "not found" in message or "not supported" in message:
                last_error = exc
                continue
            raise
    raise RuntimeError(f"No Gemini model worked. Last error: {last_error}")


def score_label(score: int) -> tuple:
    if score >= 80:
        return "Excellent", "green"
    if score >= 65:
        return "Good", "blue"
    if score >= 50:
        return "Needs work", "orange"
    return "Poor", "red"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.caption("Upload your resume and get an ATS score with specific improvements.")

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        with st.sidebar:
            st.header("Settings")
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey",
            )

    col_left, col_right = st.columns(2)
    with col_left:
        uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    with col_right:
        job_description = st.text_area(
            "Job description (optional, improves keyword matching)",
            height=160,
            placeholder="Paste the job posting here...",
        )

    if not st.button("Analyse resume", type="primary", disabled=uploaded is None):
        st.info("Upload a resume and click **Analyse resume**.")
        return

    if not api_key:
        st.error("Please add your Gemini API key in the sidebar.")
        return
    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"File is too large. Maximum size is {MAX_FILE_MB} MB.")
        return

    try:
        with st.spinner("Reading your resume..."):
            text = extract_text(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    if len(text) < 100:
        st.error(
            "Almost no text could be extracted. The file may be a scanned image, "
            "which ATS systems also cannot read. Export it as a text-based PDF or DOCX."
        )
        return

    try:
        with st.spinner("Analysing with Gemini..."):
            result = analyse_resume(text, job_description, api_key)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return

    # ---- Results -----------------------------------------------------------
    label, colour = score_label(result["overall"])
    st.divider()
    top_left, top_right = st.columns([1, 2])
    with top_left:
        st.metric("ATS score", f"{result['overall']} / 100")
        st.markdown(f":{colour}[**{label}**]")
    with top_right:
        st.write(result["summary"])

    st.subheader("Score breakdown")
    for key, (title, weight) in CATEGORIES.items():
        value = result["scores"][key]
        st.write(f"**{title}** ({weight}% of score): {value}/100")
        st.progress(value / 100)

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Strengths")
        for item in result["strengths"] or ["None identified."]:
            st.markdown(f"- {item}")
    with right:
        st.subheader("⚠️ Problems")
        for item in result["problems"] or ["None identified."]:
            st.markdown(f"- {item}")

    st.subheader("🛠️ Improvements")
    icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
    for item in result["improvements"]:
        with st.expander(f"{icons[item['priority']]} {item['priority']}: {item['issue']}"):
            st.write(item["fix"])
    if not result["improvements"]:
        st.write("No improvements returned.")

    kw_left, kw_right = st.columns(2)
    with kw_left:
        st.subheader("Keywords found")
        st.write(", ".join(result["found_keywords"]) or "None identified.")
    with kw_right:
        st.subheader("Keywords missing")
        st.write(", ".join(result["missing_keywords"]) or "None identified.")

    with st.expander("Extracted resume text (what an ATS would see)"):
        st.text(text)

    st.caption(
        "This score is an AI estimate, not the result of any real ATS. "
        "Use it as guidance, not a guarantee."
    )


if __name__ == "__main__":
    main()
