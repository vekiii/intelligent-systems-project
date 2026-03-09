# ⚡ JobBoard ATS – MVP Setup Guide

A minimal AI-powered Applicant Tracking System.

---

## Architecture Overview

```
Browser (HTML/JS)  ──fetch──►  FastAPI Backend  ──►  SQLite DB
                                     │
                                     └──► Claude API (CV scoring)
```

---

## 🤖 AI Model Used: Claude Haiku

**Why Claude Haiku (`claude-haiku-4-5`)?**
- Fast (< 2 seconds per CV)
- Cheap (< $0.01 per CV analysis)
- Understands context — it reads a messy PDF and understands skills, experience, etc.
- You can swap in any other model by editing `score_cv_with_claude()` in `main.py`

**Free/cheap alternatives if you want to avoid the Anthropic API:**
| Model | How to use | Notes |
|---|---|---|
| **Mistral 7B** | Hugging Face Inference API (free tier) | ~50% as accurate as Haiku |
| **Ollama (local)** | Run `ollama run mistral` locally | 100% free, needs 8GB+ RAM |
| **sentence-transformers** | `pip install sentence-transformers` | Keyword similarity only, no reasoning |

The backend is structured so you only need to replace the `score_cv_with_claude()` function in `main.py`.

---

## 🚀 Setup

### 1. Clone / unzip the project

```
ats-app/
  backend/
    main.py
    requirements.txt
  frontend/
    index.html
    apply.html
    hr-login.html
    hr-dashboard.html
    style.css
```

### 2. Set your Anthropic API key

Get a key at https://console.anthropic.com

**Mac / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Windows:**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Start the backend

```bash
cd backend
python main.py
```

The API is now running at `http://localhost:8000`
Docs: `http://localhost:8000/docs`

### 5. Serve the frontend

**Option A – Python (simplest):**
```bash
cd frontend
python -m http.server 3000
```
Then open http://localhost:3000

**Option B – VS Code Live Server** (if you use VS Code, just right-click `index.html` → Open with Live Server)

---

## 🔑 HR Credentials (MVP)

| Username | Password | Company | Position |
|---|---|---|---|
| hr1 | password1 | TechCorp | Junior Frontend Developer |
| hr2 | password2 | StartupXYZ | Backend Python Intern |
| hr3 | password3 | DevAgency | DevOps Engineer |

---

## 📋 How it works

1. **Candidate** visits `index.html`, reads a job offer, clicks Apply
2. They fill out the form and upload a **PDF CV**
3. The backend:
   - Saves the PDF
   - Extracts all text with `pdfplumber`
   - Sends text + job requirements to **Claude Haiku**
   - Claude returns a **score 0–10** and a **2-sentence explanation**
   - Saves everything to **SQLite**
4. **HR** logs in and sees only candidates who scored **≥ 6/10**, sorted by score
5. Each candidate card shows name, email, phone, score, and the AI's explanation

---

## ⚙️ Configuration

Edit the top of `backend/main.py`:

```python
AI_PASS_THRESHOLD = 6.0   # Change to 7.0 to be stricter, 5.0 to be more lenient
SECRET_KEY = "..."         # Change before deploying!
```

---

## 🗂️ API Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/api/jobs` | List all 3 jobs |
| GET | `/api/jobs/{id}` | Single job detail |
| POST | `/api/apply` | Submit application (multipart form) |
| POST | `/api/hr/login` | HR login → returns JWT |
| GET | `/api/hr/applications` | Qualified candidates (auth required) |
| GET | `/api/hr/all-applications` | All candidates incl. rejected (debug) |

---

## 🔮 Possible next steps (post-MVP)

- [ ] Add real email notifications (send confirmation to candidates)
- [ ] Allow HR to mark candidates as "Interview scheduled" / "Rejected"
- [ ] Store CVs in S3 / cloud storage instead of local disk
- [ ] Move to PostgreSQL for production
- [ ] Add proper user management instead of hardcoded HR accounts
- [ ] Add a CV download button in the dashboard
