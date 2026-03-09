"""
main.py — ATS MVP backend (v3 — Ollama edition, 100% free)

Three-component scoring pipeline:
  1. Rule-based CV quality scorer  (35%) — always runs
  2. ML model confidence           (15%) — runs if models/cv_classifier.pkl exists
  3. Ollama/Mistral job-fit scorer (50%) — always runs (requires Ollama running locally)

If the ML model is not found, weights auto-rebalance to:
  1. Rule-based  40%
  3. Ollama      60%

Setup:
  1. Install Ollama from https://ollama.com
  2. Run: ollama pull mistral
  3. Start backend normally — no API key needed
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3, os, uuid, json, re, pickle
from datetime import datetime, timedelta
from jose import JWTError, jwt
import pdfplumber
import urllib.request

app = FastAPI(title="ATS MVP v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY        = "change-this-in-production-please"
ALGORITHM         = "HS256"
UPLOAD_DIR        = "uploads"
MODEL_PATH        = "models/cv_classifier.pkl"
AI_PASS_THRESHOLD = 6.0

os.makedirs(UPLOAD_DIR, exist_ok=True)

HR_ACCOUNTS = {
    "hr1": {"password": "password1", "company": "TechCorp"},
    "hr2": {"password": "password2", "company": "StartupXYZ"},
    "hr3": {"password": "password3", "company": "DevAgency"},
}

# ─── Load ML model (optional) ─────────────────────────────────────────────────
_ml_model = None

def load_ml_model():
    global _ml_model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _ml_model = pickle.load(f)
        print(f"✅ ML model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  ML model not found at {MODEL_PATH} — will use 2-component scoring (rules + Claude)")

def ml_model_available() -> bool:
    return _ml_model is not None


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — CV QUALITY SCORER  (rule-based)
# Ported from friend's train.py — checks structure, contact, education, etc.
# Returns 0–100 score + breakdown
# ══════════════════════════════════════════════════════════════════════════════

def score_cv_quality(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"total": 0, "label": "LOW", "breakdown": {}, "word_count": 0}

    text_lower = text.lower()
    word_count = len(text_lower.split())
    scores = {}

    # 1. Length (max 20)
    if 300 <= word_count <= 1000:
        scores["length"] = 20
    elif 150 <= word_count < 300 or 1000 < word_count <= 1500:
        scores["length"] = 12
    elif word_count > 1500:
        scores["length"] = 6
    else:
        scores["length"] = 2

    # 2. Contact info (max 15)
    contact = 0
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        contact += 6
    if re.search(r"(\+?\d[\d\s\-().]{7,15}\d)", text):
        contact += 5
    if any(x in text_lower for x in ["linkedin", "github", "portfolio", "website"]):
        contact += 4
    scores["contact"] = min(contact, 15)

    # 3. Education (max 15)
    edu_map = {
        "phd": 15, "doctorate": 15, "ph.d": 15,
        "master": 12, "mba": 12, "msc": 12,
        "bachelor": 9, "b.s": 9, "b.a": 9, "bsc": 9,
        "university": 6, "college": 6,
        "diploma": 4, "certification": 3,
    }
    edu = max((v for k, v in edu_map.items() if k in text_lower), default=0)
    scores["education"] = min(edu, 15)

    # 4. Experience (max 20)
    exp = 0
    m = re.search(r"(\d+)\+?\s*years?\s*(of)?\s*experience", text_lower)
    if m:
        yrs = int(m.group(1))
        exp += 20 if yrs >= 10 else 15 if yrs >= 5 else 10 if yrs >= 2 else 6
    action_words = ["managed", "developed", "led", "designed", "implemented",
                    "achieved", "delivered", "increased", "reduced", "built"]
    exp += min(sum(1 for w in action_words if w in text_lower) * 2, 10)
    scores["experience"] = min(exp, 20)

    # 5. Technical skills (max 15)
    tech = ["python", "java", "javascript", "c++", "c#", "sql", "r", "scala",
            "react", "angular", "vue", "django", "flask", "spring", "node",
            "tensorflow", "pytorch", "sklearn", "pandas", "numpy",
            "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
            "tableau", "power bi", "excel", "spark", "hadoop", "typescript"]
    scores["skills"] = min(sum(1 for k in tech if k in text_lower) * 3, 15)

    # 6. Structure / sections (max 15)
    sections = ["summary", "objective", "profile", "education", "experience",
                "work history", "skills", "competencies", "projects",
                "achievements", "certifications", "awards", "publications"]
    scores["structure"] = min(sum(1 for s in sections if s in text_lower) * 3, 15)

    total = sum(scores.values())
    label = "HIGH" if total >= 65 else "MEDIUM" if total >= 35 else "LOW"

    return {
        "total":      total,
        "label":      label,
        "word_count": word_count,
        "breakdown": {
            "length":     {"score": scores["length"],     "max": 20},
            "contact":    {"score": scores["contact"],    "max": 15},
            "education":  {"score": scores["education"],  "max": 15},
            "experience": {"score": scores["experience"], "max": 20},
            "skills":     {"score": scores["skills"],     "max": 15},
            "structure":  {"score": scores["structure"],  "max": 15},
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — ML MODEL CONFIDENCE
# Uses the trained TF-IDF + RandomForest model from train.py
# Converts the model's HIGH/MEDIUM/LOW confidence into a 0–10 score
# ══════════════════════════════════════════════════════════════════════════════

def score_cv_ml(cv_text: str) -> dict:
    """
    Returns {"score": 0-10, "label": str, "confidence": dict}
    Score = weighted average of class probabilities:
      HIGH   → 10,  MEDIUM → 5,  LOW → 0
    """
    if not ml_model_available():
        return {"score": None, "label": None, "confidence": {}}

    try:
        proba = _ml_model.predict_proba([cv_text])[0]
        classes = _ml_model.classes_

        label_score = {"HIGH": 10.0, "MEDIUM": 5.0, "LOW": 0.0}
        weighted = sum(
            proba[i] * label_score.get(classes[i], 5.0)
            for i in range(len(classes))
        )

        confidence = {
            classes[i]: round(float(proba[i]) * 100, 1)
            for i in range(len(classes))
        }
        predicted_label = classes[proba.argmax()]

        return {
            "score":      round(weighted, 2),   # 0–10
            "label":      predicted_label,
            "confidence": confidence,            # e.g. {"HIGH": 72.3, "MEDIUM": 21.1, "LOW": 6.6}
        }
    except Exception as e:
        return {"score": None, "label": None, "confidence": {}, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — JOB-FIT SCORER  (Claude Haiku)
# Checks: how well the CV matches the specific job requirements
# Returns 0–10 score + 2-sentence summary
# ══════════════════════════════════════════════════════════════════════════════

def score_cv_job_fit(cv_text: str, job_title: str, requirements: str) -> dict:
    """
    Uses Ollama (local gemma3:4b model) to score the CV against job requirements.
    Returns overall score, 2-sentence summary, and per-skill breakdown.
    """
    OLLAMA_URL = "http://localhost:11434/api/generate"

    skills = [s.strip() for s in requirements.split(",") if s.strip()]
    skill_lines = "\n".join(f'    "{s}": <score 0-10>' for s in skills)

    prompt = f"""You are an expert technical recruiter. Evaluate the following CV against the job requirements.

JOB TITLE: {job_title}
REQUIRED SKILLS: {requirements}

CV CONTENT:
\"\"\"
{cv_text[:4000]}
\"\"\"

Instructions:
- Give an OVERALL score 0-10 (decimals allowed).
- Score EACH required skill 0-10: 0-3=not present, 4-6=partial, 7-10=clearly demonstrated.
- Write a 2-sentence summary explaining the overall score.

Respond ONLY with valid JSON, no markdown:
{{
  "score": <overall 0-10>,
  "summary": "<two sentences>",
  "breakdown": {{
{skill_lines}
  }}
}}"""

    payload = json.dumps({
        "model":  "gemma3:4b",
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw  = body.get("response", "").strip()
    except Exception as e:
        raise ValueError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Make sure Ollama is running and gemma3:4b is pulled. Error: {e}"
        )

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Find the JSON object
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    result = json.loads(raw)
    result["score"] = float(result["score"])

    # Normalise breakdown values to floats, fallback if missing
    if "breakdown" in result and isinstance(result["breakdown"], dict):
        result["breakdown"] = {k: float(v) for k, v in result["breakdown"].items()}
    else:
        result["breakdown"] = {s: result["score"] for s in skills}

    return result


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCORER
#
# With ML model:    35% quality (rules) + 15% ML confidence + 50% job fit
# Without ML model: 40% quality (rules) + 60% job fit
# ══════════════════════════════════════════════════════════════════════════════

def analyse_cv(cv_path: str, job_title: str, requirements: str) -> dict:
    # Extract text once
    cv_text = ""
    with pdfplumber.open(cv_path) as pdf:
        for page in pdf.pages:
            cv_text += (page.extract_text() or "") + "\n"
    cv_text = cv_text.strip()

    # Component 1 — quality rules
    quality        = score_cv_quality(cv_text)
    quality_score  = round((quality["total"] / 100) * 10, 2)   # normalise to 0–10

    # Component 2 — ML model
    ml             = score_cv_ml(cv_text)
    ml_available   = ml["score"] is not None

    # Component 3 — Claude job fit
    job_fit        = score_cv_job_fit(cv_text, job_title, requirements)

    # Weighted final score
    if ml_available:
        final_score = round(
            quality_score     * 0.35 +
            ml["score"]       * 0.15 +
            job_fit["score"]  * 0.50,
            2
        )
    else:
        final_score = round(
            quality_score    * 0.40 +
            job_fit["score"] * 0.60,
            2
        )

    return {
        "final_score":          final_score,
        # Component 1
        "quality_score":        quality_score,
        "quality_label":        quality["label"],
        "quality_breakdown":    quality["breakdown"],
        # Component 2
        "ml_score":             ml["score"],
        "ml_label":             ml["label"],
        "ml_confidence":        ml["confidence"],
        "ml_available":         ml_available,
        # Component 3
        "job_fit_score":        job_fit["score"],
        "job_fit_summary":      job_fit["summary"],
        "job_fit_breakdown":    job_fit.get("breakdown", {}),
        # Meta
        "word_count":           quality["word_count"],
    }


# ─── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("ats.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id           INTEGER PRIMARY KEY,
            title        TEXT NOT NULL,
            company      TEXT NOT NULL,
            description  TEXT NOT NULL,
            requirements TEXT NOT NULL,
            hr_username  TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id            INTEGER NOT NULL,
            name              TEXT NOT NULL,
            surname           TEXT NOT NULL,
            email             TEXT NOT NULL,
            phone             TEXT NOT NULL,
            cv_path           TEXT NOT NULL,
            final_score       REAL,
            quality_score     REAL,
            quality_label     TEXT,
            quality_breakdown TEXT,
            ml_score          REAL,
            ml_label          TEXT,
            ml_confidence     TEXT,
            ml_available      INTEGER,
            job_fit_score     REAL,
            job_fit_summary   TEXT,
            job_fit_breakdown TEXT,
            word_count        INTEGER,
            status            TEXT DEFAULT 'new',
            created_at        TEXT NOT NULL
        )
    """)

    # Migrate existing DB — add new columns if they don't exist yet
    existing = [row[1] for row in c.execute("PRAGMA table_info(applications)").fetchall()]
    if "job_fit_breakdown" not in existing:
        c.execute("ALTER TABLE applications ADD COLUMN job_fit_breakdown TEXT")
    if "status" not in existing:
        c.execute("ALTER TABLE applications ADD COLUMN status TEXT DEFAULT 'new'")

    if c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0:
        jobs = [
            (
                1, "Junior Frontend Developer", "TechCorp",
                (
                    "TechCorp is a product-first company building the next generation of SaaS tools. "
                    "We are looking for a passionate Junior Frontend Developer who loves crafting clean, "
                    "responsive UIs. You will work closely with our design and backend teams to ship "
                    "real features from day one.\n\n"
                    "Responsibilities:\n"
                    "• Build and maintain React components\n"
                    "• Collaborate with designers to implement pixel-perfect UIs\n"
                    "• Write clean, well-documented JavaScript/TypeScript\n"
                    "• Participate in code reviews\n\n"
                    "Nice to have: experience with Tailwind CSS, Figma, or any testing library."
                ),
                "React, JavaScript, HTML, CSS, Git, REST APIs, TypeScript", "hr1",
            ),
            (
                2, "Backend Python Intern", "StartupXYZ",
                (
                    "StartupXYZ is a fast-growing fintech startup. We are looking for a Backend Python "
                    "Intern who is comfortable with Python and eager to learn. You will contribute to "
                    "building and maintaining our REST APIs, help improve database performance, and write "
                    "automated tests.\n\n"
                    "Responsibilities:\n"
                    "• Develop and maintain REST APIs using FastAPI or Django\n"
                    "• Write SQL queries and migrations\n"
                    "• Write unit and integration tests\n"
                    "• Document your code and APIs\n\n"
                    "Nice to have: knowledge of Redis, Celery, or Docker."
                ),
                "Python, FastAPI or Django, SQL, REST APIs, Git, pytest", "hr2",
            ),
            (
                3, "DevOps Engineer", "DevAgency",
                (
                    "DevAgency helps enterprises modernise their infrastructure. We need a DevOps Engineer "
                    "to maintain, automate and improve our CI/CD pipelines and cloud infrastructure.\n\n"
                    "Responsibilities:\n"
                    "• Manage and improve Kubernetes clusters\n"
                    "• Build and maintain CI/CD pipelines (GitHub Actions / GitLab CI)\n"
                    "• Automate infrastructure with Terraform or Ansible\n"
                    "• Monitor services with Prometheus / Grafana\n\n"
                    "Nice to have: experience with AWS or GCP cost optimisation."
                ),
                "Docker, Kubernetes, CI/CD, Linux, AWS or GCP, Terraform, Bash scripting", "hr3",
            ),
        ]
        c.executemany("INSERT INTO jobs VALUES (?,?,?,?,?,?)", jobs)

    conn.commit()
    conn.close()


# ─── Auth ──────────────────────────────────────────────────────────────────────
def create_token(username: str) -> str:
    payload = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=8)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

security = HTTPBearer()

def get_current_hr(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload  = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username not in HR_ACCOUNTS:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ─── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    load_ml_model()


# ─── Public endpoints ──────────────────────────────────────────────────────────
@app.get("/api/jobs")
def list_jobs():
    conn = get_db()
    jobs = conn.execute(
        "SELECT id, title, company, description, requirements FROM jobs"
    ).fetchall()
    conn.close()
    return [dict(j) for j in jobs]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    conn = get_db()
    job  = conn.execute(
        "SELECT id, title, company, description, requirements FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    conn.close()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return dict(job)

@app.post("/api/apply")
async def apply(
    job_id:  int        = Form(...),
    name:    str        = Form(...),
    surname: str        = Form(...),
    email:   str        = Form(...),
    phone:   str        = Form(...),
    cv:      UploadFile = File(...),
):
    if not cv.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    filename = f"{uuid.uuid4()}.pdf"
    cv_path  = os.path.join(UPLOAD_DIR, filename)
    with open(cv_path, "wb") as f:
        f.write(await cv.read())

    conn = get_db()
    job  = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not job:
        conn.close()
        os.remove(cv_path)
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        result = analyse_cv(cv_path, job["title"], job["requirements"])
    except Exception as e:
        result = {
            "final_score": 0.0, "quality_score": 0.0, "quality_label": "LOW",
            "quality_breakdown": {}, "ml_score": None, "ml_label": None,
            "ml_confidence": {}, "ml_available": False,
            "job_fit_score": 0.0, "job_fit_summary": f"Analysis failed: {e}",
            "job_fit_breakdown": {}, "word_count": 0,
        }

    conn.execute(
        """INSERT INTO applications
           (job_id, name, surname, email, phone, cv_path,
            final_score, quality_score, quality_label, quality_breakdown,
            ml_score, ml_label, ml_confidence, ml_available,
            job_fit_score, job_fit_summary, job_fit_breakdown,
            word_count, status, created_at)
           VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?)""",
        (
            job_id, name, surname, email, phone, cv_path,
            result["final_score"], result["quality_score"], result["quality_label"],
            json.dumps(result["quality_breakdown"]),
            result["ml_score"], result["ml_label"],
            json.dumps(result["ml_confidence"]), int(bool(result["ml_available"])),
            result["job_fit_score"], result["job_fit_summary"],
            json.dumps(result.get("job_fit_breakdown", {})),
            result["word_count"], "new", datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    return {"message": "Application submitted! We will be in touch."}


# ─── HR endpoints ──────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class StatusUpdate(BaseModel):
    status: str   # "new" | "read" | "starred" | "rejected"

@app.patch("/api/hr/applications/{app_id}/status")
def update_status(app_id: int, body: StatusUpdate, current_hr: str = Depends(get_current_hr)):
    allowed = {"new", "read", "starred", "rejected"}
    if body.status not in allowed:
        raise HTTPException(status_code=400, detail=f"Status must be one of {allowed}")
    conn = get_db()
    # Verify the application belongs to this HR's job
    row = conn.execute(
        """SELECT a.id FROM applications a
           JOIN jobs j ON a.job_id = j.id
           WHERE a.id=? AND j.hr_username=?""",
        (app_id, current_hr)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Application not found")
    conn.execute("UPDATE applications SET status=? WHERE id=?", (body.status, app_id))
    conn.commit()
    conn.close()
    return {"id": app_id, "status": body.status}

@app.post("/api/hr/login")
def hr_login(data: LoginRequest):
    account = HR_ACCOUNTS.get(data.username)
    if not account or account["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {
        "token":         create_token(data.username),
        "company":       account["company"],
        "username":      data.username,
        "ml_available":  ml_model_available(),
    }

@app.get("/api/hr/applications")
def get_applications(current_hr: str = Depends(get_current_hr)):
    conn = get_db()
    job  = conn.execute("SELECT * FROM jobs WHERE hr_username=?", (current_hr,)).fetchone()
    if not job:
        conn.close()
        return {"job": None, "applications": [], "ml_available": ml_model_available()}

    apps = conn.execute(
        """SELECT id, name, surname, email, phone,
                  final_score, quality_score, quality_label, quality_breakdown,
                  ml_score, ml_label, ml_confidence, ml_available,
                  job_fit_score, job_fit_summary, job_fit_breakdown,
                  word_count, status, created_at
           FROM applications
           WHERE job_id=? AND final_score >= ?
           ORDER BY final_score DESC""",
        (job["id"], AI_PASS_THRESHOLD),
    ).fetchall()
    conn.close()

    result = []
    for a in apps:
        row = dict(a)
        row["quality_breakdown"]  = json.loads(row.get("quality_breakdown")  or "{}")
        row["ml_confidence"]      = json.loads(row.get("ml_confidence")      or "{}")
        row["job_fit_breakdown"]  = json.loads(row.get("job_fit_breakdown")   or "{}")
        result.append(row)

    return {
        "job":          dict(job),
        "applications": result,
        "ml_available": ml_model_available(),
    }

@app.get("/api/hr/all-applications")
def get_all_applications(current_hr: str = Depends(get_current_hr)):
    conn = get_db()
    job  = conn.execute("SELECT * FROM jobs WHERE hr_username=?", (current_hr,)).fetchone()
    if not job:
        conn.close()
        return []
    apps = conn.execute(
        "SELECT * FROM applications WHERE job_id=? ORDER BY final_score DESC",
        (job["id"],),
    ).fetchall()
    conn.close()
    return [dict(a) for a in apps]

@app.get("/api/status")
def status():
    return {
        "ml_model_loaded": ml_model_available(),
        "ml_model_path":   MODEL_PATH,
        "scoring_weights": (
            {"quality": 0.35, "ml": 0.15, "job_fit": 0.50}
            if ml_model_available() else
            {"quality": 0.40, "job_fit": 0.60}
        ),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
