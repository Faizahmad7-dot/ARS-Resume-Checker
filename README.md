# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness
and suggests specific improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF** or **DOCX**
- Optional **job description** for keyword matching
- Overall **ATS score (0-100)** with a 5-category breakdown
- Strengths, problems, and prioritised improvements (High / Medium / Low)
- Found and missing keywords
- Shows the extracted text, i.e. what an ATS would actually "see"

> The score is an AI estimate, not the output of a real ATS. Use it as guidance.

## How the score works

The model rates five categories from 0-100, and the app computes the final score
itself using fixed weights (so it is consistent and transparent):

| Category | Weight |
|---|---|
| Keywords & skills match | 30% |
| Formatting & ATS readability | 20% |
| Experience & achievements | 20% |
| Structure & required sections | 15% |
| Language & clarity | 15% |

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

3. Provide the key in one of two ways:
   - Paste it into the sidebar when the app opens, **or**
   - Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# Optional: force a specific model
# GEMINI_MODEL = "gemini-3.5-flash"
```

**Never commit `.streamlit/secrets.toml` or your key to GitHub.**

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (with `app.py` and `requirements.txt` in the root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, pick your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your-key-here"
```

5. Click **Deploy**.

## Project structure

```
app.py             # Streamlit app
requirements.txt   # Python dependencies
README.md
```

## Notes

- Scanned/image-only PDFs can't be read (ATS systems can't read them either).
- Resume text is sent to the Gemini API for analysis. Don't upload anything you
  aren't comfortable sharing with that service.
- If Gemini model names change, set `GEMINI_MODEL` in your secrets.
