# ⚡ JobBoard ATS — AI-Powered Applicant Tracking System

An MVP Applicant Tracking System that uses a multi-component AI pipeline to automatically screen and score CVs. Built as a Master's student project for the course "Intelligent Systems".

---

## 🧠 AI Architecture

The system uses three AI components working together to evaluate every submitted CV:

| Component | Technology | Weight | Purpose |
|---|---|---|---|
| **CV Quality** | Random Forest + TF-IDF | 40% | Scores how well-written the CV is, trained on 2,484 real CVs |
| **Job Fit** | Ollama / Gemma 3:4b (LLM) | 60% | Scores how well the candidate matches the specific job requirements |
| **Display Breakdown** | Rule-based expert system | — | Generates the 6-criteria visual breakdown (not used in scoring) |

The final score is **0–10**. Candidates scoring below **6/10** are automatically filtered out and hidden from HR.

### CV Quality — Rule-based Display Breakdown
The dashboard shows a breakdown across 6 criteria:

| Criterion | Max Points | What is checked |
|---|---|---|
| Length | 20 | Optimal CV length (450–700 words scores highest) |
| Contact | 15 | Email, phone, LinkedIn/GitHub presence |
| Education | 15 | Degree level (PhD → Master → Bachelor → Diploma) |
| Experience | 20 | Years of experience + action verbs (80+ words) |
| Skills | 15 | Domain keywords across 24 job fields (150+ terms) |
| Structure | 15 | CV sections (summary, education, skills, projects...) |

### ML Model Training
The Random Forest model is trained using **weak supervision** — the rule-based scorer labels 2,484 CVs from the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) as HIGH / MEDIUM / LOW, and the model learns to generalise those patterns from the raw CV text.

---

## 🗂️ Project Structure

```
project-root/
├── backend/
│   ├── main.py              ← FastAPI server — all backend logic
│   ├── requirements.txt     ← Python dependencies
│   └── uploads/             ← Uploaded CVs (auto-created, gitignored)
├── frontend/
│   ├── index.html           ← Landing page
│   ├── jobs.html            ← Job listings
│   ├── apply.html           ← Job detail + application form
│   ├── hr-login.html        ← HR login page
│   └── hr-dashboard.html    ← HR candidate dashboard
├── scripts/
│   └── train.py             ← ML model training script
├── models/                  ← Trained model saved here (gitignored)
├── .gitignore
├── README.md
└── Resume.csv               ← Kaggle dataset for training (gitignored)
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Gemma 3:4b model pulled

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Install and start Ollama
Download Ollama from [ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull gemma3:4b
```
Make sure Ollama is running before starting the backend. Verify at: `http://localhost:11434`

### 4. Train the ML model (one-time setup)
Download `Resume.csv` from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and place it in the project root, then:
```bash
cd ..
python scripts/train.py --data Resume.csv
```
This takes ~5–10 minutes and saves the model to `models/cv_classifier.pkl`.

### 5. Start the backend
```bash
cd backend
python main.py
```
Backend runs at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### 6. Serve the frontend
Open a second terminal:
```bash
cd frontend
python -m http.server 3000
```
Open your browser at: **http://localhost:3000**

---

## 🔑 HR Credentials (MVP)

| Username | Password | Company | Position |
|---|---|---|---|
| hr1 | password1 | TechCorp | Junior Frontend Developer |
| hr2 | password2 | StartupXYZ | Backend Python Intern |
| hr3 | password3 | DevAgency | DevOps Engineer |

---

## 🗄️ Data Storage

All application data is stored in a **SQLite** database (`backend/ats.db`), which is created automatically on first startup. Uploaded CV PDFs are saved to `backend/uploads/`. Both are excluded from version control.

---

## 🌐 API Endpoints

| Method | URL | Description | Auth |
|---|---|---|---|
| GET | `/api/jobs` | List all job openings | Public |
| GET | `/api/jobs/{id}` | Get job details | Public |
| POST | `/api/apply` | Submit application + CV | Public |
| POST | `/api/hr/login` | HR login → returns JWT | Public |
| GET | `/api/hr/applications` | Qualified candidates (score ≥ 6) | HR |
| GET | `/api/hr/all-applications` | All candidates including rejected | HR |
| PATCH | `/api/hr/applications/{id}/status` | Update candidate status | HR |
| GET | `/api/status` | Backend + model status | Public |

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, Tailwind CSS, Alpine.js |
| Backend | Python, FastAPI, Uvicorn |
| Database | SQLite |
| PDF parsing | pdfplumber |
| ML model | scikit-learn (TF-IDF + Random Forest) |
| LLM | Ollama + Gemma 3:4b |
| Auth | JWT (python-jose) |

---

## ⚙️ Configuration

Key settings at the top of `backend/main.py`:

```python
AI_PASS_THRESHOLD = 6.0   # Candidates below this score are hidden from HR
MODEL_PATH = "models/cv_classifier.pkl"
SECRET_KEY = "change-this-in-production-please"
```

---

## 📋 Candidate Status System

HR can manually assign one of four statuses to each candidate:

| Status | Meaning |
|---|---|
| 🔵 New | Freshly submitted, not yet reviewed |
| 👁️ Read | HR has reviewed the application |
| ⭐ Starred | Shortlisted for interview |
| ❌ Rejected | Manually rejected by HR |
