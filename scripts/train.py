"""
train.py — Učitava Kaggle dataset, labeluje CV-jeve i trenira model.
Pokretanje: python scripts/train.py --data Resume.csv
"""

import argparse
import pickle
import re
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────

def score_cv_quality(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"total": 0, "label": "LOW", "breakdown": {}, "word_count": 0}

    text_lower = text.lower()
    word_count = len(text_lower.split())
    scores = {}

    # ── 1. LENGTH (max 20) ────────────────────────────────────────────────────
    # Sweet spot: 450–700 words (score 20). Degrades symmetrically in both
    # directions. Exactly 20 elif branches → one point per condition.
    if   450 <= word_count <= 700:   scores["length"] = 20
    elif 400 <= word_count <  450 or  701 <= word_count <= 750:  scores["length"] = 19
    elif 350 <= word_count <  400 or  751 <= word_count <= 800:  scores["length"] = 18
    elif 300 <= word_count <  350 or  801 <= word_count <= 850:  scores["length"] = 17
    elif 275 <= word_count <  300 or  851 <= word_count <= 900:  scores["length"] = 16
    elif 250 <= word_count <  275 or  901 <= word_count <= 950:  scores["length"] = 15
    elif 225 <= word_count <  250 or  951 <= word_count <= 1000: scores["length"] = 14
    elif 200 <= word_count <  225 or 1001 <= word_count <= 1050: scores["length"] = 13
    elif 175 <= word_count <  200 or 1051 <= word_count <= 1100: scores["length"] = 12
    elif 150 <= word_count <  175 or 1101 <= word_count <= 1200: scores["length"] = 11
    elif 125 <= word_count <  150 or 1201 <= word_count <= 1300: scores["length"] = 10
    elif 100 <= word_count <  125 or 1301 <= word_count <= 1400: scores["length"] = 9
    elif  80 <= word_count <  100 or 1401 <= word_count <= 1500: scores["length"] = 8
    elif  60 <= word_count <   80 or 1501 <= word_count <= 1700: scores["length"] = 7
    elif  50 <= word_count <   60 or 1701 <= word_count <= 1900: scores["length"] = 6
    elif  40 <= word_count <   50 or 1901 <= word_count <= 2100: scores["length"] = 5
    elif  30 <= word_count <   40 or 2101 <= word_count <= 2400: scores["length"] = 4
    elif  20 <= word_count <   30 or 2401 <= word_count <= 2800: scores["length"] = 3
    elif  10 <= word_count <   20 or 2801 <= word_count <= 3500: scores["length"] = 2
    else:                                                         scores["length"] = 1

    # ── 2. CONTACT INFO (max 15) — unchanged ─────────────────────────────────
    contact = 0
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        contact += 6
    if re.search(r"(\+?\d[\d\s\-().]{7,15}\d)", text):
        contact += 5
    if any(x in text_lower for x in ["linkedin", "github", "portfolio", "website"]):
        contact += 4
    scores["contact"] = min(contact, 15)

    # ── 3. EDUCATION (max 15) — unchanged ────────────────────────────────────
    edu_map = {
        "phd": 15, "doctorate": 15, "ph.d": 15,
        "master": 12, "mba": 12, "msc": 12,
        "bachelor": 9, "b.s": 9, "b.a": 9, "bsc": 9,
        "university": 6, "college": 6,
        "diploma": 4, "certification": 3,
    }
    edu = max((v for k, v in edu_map.items() if k in text_lower), default=0)
    scores["education"] = min(edu, 15)

    # ── 4. EXPERIENCE (max 20) ────────────────────────────────────────────────
    # Years-of-experience phrase detection (unchanged logic, same max)
    exp = 0
    m = re.search(r"(\d+)\+?\s*years?\s*(of)?\s*experience", text_lower)
    if m:
        yrs = int(m.group(1))
        exp += 20 if yrs >= 10 else 15 if yrs >= 5 else 10 if yrs >= 2 else 6

    # Expanded action-word list covering all 24 job fields in the dataset.
    # Each matched word is worth 1 pt; capped at 10 pts total so years + words ≤ 20.
    action_words = [
        # ── Leadership & management (all fields) ──
        "managed", "led", "supervised", "directed", "coordinated", "oversaw",
        "administered", "spearheaded", "championed", "facilitated", "delegated",
        "mentored", "coached", "guided", "headed",
        # ── Creation & development ──
        "developed", "built", "created", "designed", "implemented", "established",
        "launched", "founded", "initiated", "introduced", "produced", "engineered",
        "constructed", "fabricated", "assembled",
        # ── Achievement & impact ──
        "achieved", "delivered", "improved", "increased", "reduced", "optimized",
        "enhanced", "streamlined", "maximized", "minimized", "accelerated",
        "transformed", "generated", "secured", "saved",
        # ── Analysis & research ──
        "analyzed", "researched", "evaluated", "assessed", "investigated",
        "examined", "identified", "diagnosed", "audited", "reviewed",
        "forecasted", "modeled", "calculated", "reported",
        # ── Communication & collaboration ──
        "presented", "negotiated", "communicated", "collaborated", "consulted",
        "advised", "trained", "educated", "pitched", "published",
        # ── IT / engineering specific ──
        "programmed", "coded", "configured", "integrated", "deployed",
        "automated", "tested", "debugged", "maintained", "upgraded",
        "migrated", "refactored", "architected",
        # ── Sales / business development ──
        "sold", "acquired", "converted", "retained", "expanded", "grew",
        "negotiated", "closed", "prospected", "upsold",
        # ── Healthcare specific ──
        "treated", "administered", "rehabilitated", "monitored", "cared",
        # ── Finance / accounting specific ──
        "budgeted", "reconciled", "allocated", "invested", "filed",
        # ── Teaching / HR specific ──
        "recruited", "onboarded", "facilitated", "lectured", "assessed",
        # ── Construction / operations ──
        "operated", "installed", "repaired", "inspected", "surveyed",
    ]
    # Deduplicate in case of overlap
    unique_action_words = list(dict.fromkeys(action_words))
    exp += min(sum(1 for w in unique_action_words if w in text_lower), 10)
    scores["experience"] = min(exp, 20)

    # ── 5. DOMAIN SKILLS (max 15) ─────────────────────────────────────────────
    # Covers all 24 job categories in the Kaggle dataset.
    # Each matched keyword = 1 pt; score = matched_count / total_in_list * 15,
    # so the metric scales with how comprehensive the list is.
    domain_skills = [
        # ── Information Technology ──
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "swift", "kotlin", "php", "ruby", "r", "scala", "perl", "bash", "shell",
        "react", "angular", "vue", "node", "django", "flask", "spring", "fastapi",
        "express", "html", "css", "tailwind", "bootstrap",
        "tensorflow", "pytorch", "scikit-learn", "sklearn", "pandas", "numpy",
        "keras", "hugging face", "opencv", "nltk",
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "unix",
        "terraform", "ansible", "jenkins", "ci/cd", "devops", "mlops",
        "sql", "postgresql", "mysql", "mongodb", "redis", "oracle", "sqlite",
        "elasticsearch", "kafka", "rabbitmq", "spark", "hadoop", "airflow",
        "tableau", "power bi", "looker", "qlik", "jupyter", "databricks",
        "rest api", "graphql", "microservices", "agile", "scrum", "jira",
        # ── Accounting / Finance ──
        "quickbooks", "sap", "xero", "sage", "tally", "peachtree", "myob",
        "gaap", "ifrs", "erp", "accounts payable", "accounts receivable",
        "financial modeling", "dcf", "valuation", "bloomberg", "vba",
        "budgeting", "forecasting", "tax", "audit", "reconciliation", "cfa",
        "ifrs", "sox", "internal controls", "financial reporting",
        # ── Legal / Advocate ──
        "westlaw", "lexisnexis", "legal research", "litigation", "contract drafting",
        "compliance", "arbitration", "due diligence", "intellectual property",
        "corporate law", "criminal law", "civil litigation", "legal writing",
        # ── Agriculture ──
        "crop management", "irrigation", "agronomy", "precision agriculture",
        "soil science", "hydroponics", "gis", "pesticides", "fertilization",
        "livestock", "horticulture", "aquaculture", "farm management",
        # ── Apparel / Fashion ──
        "fashion design", "textile", "pattern making", "merchandising",
        "garment construction", "fabric", "fashion illustration", "trend analysis",
        "retail buying", "visual merchandising", "cad design",
        # ── Arts ──
        "photoshop", "illustrator", "indesign", "after effects", "premiere pro",
        "lightroom", "blender", "maya", "3ds max", "cinema 4d", "zbrush",
        "procreate", "corel draw", "final cut pro", "davinci resolve",
        # ── Automobile ──
        "solidworks", "catia", "ansys", "creo", "vehicle dynamics", "obd",
        "can bus", "automotive", "powertrain", "chassis", "adas",
        "embedded systems", "matlab simulink", "autocad",
        # ── Aviation ──
        "faa", "easa", "icao", "atpl", "cpl", "flight operations", "amos",
        "aviation safety", "airworthiness", "navigation", "atc", "mcc",
        # ── Banking ──
        "bloomberg terminal", "risk management", "credit analysis", "kyc",
        "aml", "anti-money laundering", "swift", "fintech", "basel",
        "loan origination", "trade finance", "treasury", "derivatives",
        # ── BPO / Customer Service ──
        "crm", "salesforce", "zendesk", "freshdesk", "avaya", "genesys",
        "call center", "ticketing", "service desk", "sla", "kpi",
        "customer satisfaction", "csat", "nps",
        # ── Business Development ──
        "hubspot", "pipedrive", "market research", "pitch deck", "b2b", "b2c",
        "lead generation", "account management", "partnership", "rfp",
        "go-to-market", "competitive analysis",
        # ── Chef / Culinary ──
        "haccp", "food safety", "culinary arts", "menu planning", "food costing",
        "sous vide", "pastry", "catering", "hospitality", "servsafe",
        "kitchen management", "food hygiene",
        # ── Construction ──
        "revit", "bim", "primavera", "ms project", "estimating", "quantity surveying",
        "autocad", "structural", "civil engineering", "project management",
        "site supervision", "cad", "tendering",
        # ── Consulting ──
        "strategy", "management consulting", "business analysis", "change management",
        "stakeholder management", "six sigma", "lean", "process improvement",
        "powerpoint", "data analysis", "kpi",
        # ── Design / UI-UX ──
        "figma", "sketch", "adobe xd", "invision", "zeplin", "wireframing",
        "prototyping", "user research", "usability testing", "design thinking",
        "interaction design", "information architecture",
        # ── Digital Media / Marketing ──
        "seo", "sem", "google analytics", "google ads", "facebook ads",
        "social media", "content management", "wordpress", "hootsuite",
        "mailchimp", "email marketing", "copywriting", "a/b testing",
        "conversion rate", "affiliate marketing",
        # ── Engineering (general) ──
        "labview", "plc", "scada", "pid", "autocad", "solidworks",
        "mechanical engineering", "electrical engineering", "civil engineering",
        "chemical engineering", "quality assurance", "iso", "six sigma",
        # ── Fitness / Sports ──
        "personal training", "nutrition", "exercise physiology", "cpr",
        "nasm", "acsm", "strength conditioning", "crossfit", "yoga",
        "physical therapy", "sports medicine", "group fitness",
        # ── Healthcare ──
        "ehr", "emr", "hipaa", "icd-10", "cpt", "medical coding",
        "clinical", "nursing", "patient care", "pharmacology", "anatomy",
        "radiology", "surgery", "diagnostics", "telemedicine",
        # ── Human Resources ──
        "hris", "workday", "successfactors", "bamboohr", "ats",
        "recruiting", "onboarding", "payroll", "performance management",
        "employee relations", "compensation", "benefits", "talent acquisition",
        # ── Public Relations ──
        "media relations", "press release", "crisis communication",
        "brand management", "reputation management", "event management",
        "stakeholder engagement", "communications strategy",
        # ── Sales ──
        "pipeline management", "cold calling", "account management",
        "territory management", "quota", "negotiation", "prospecting",
        "salesforce crm", "revenue growth", "customer retention",
        # ── Teaching / Education ──
        "curriculum development", "lms", "moodle", "blackboard", "canvas",
        "lesson planning", "classroom management", "e-learning",
        "instructional design", "assessment", "differentiated instruction",
    ]

    matched = sum(1 for k in domain_skills if k in text_lower)
    # Scale: every matched keyword contributes proportionally up to max 15
    # Formula: min(matched * (15 / 20), 15) — reaching max at ~20 matches
    scores["skills"] = min(round(matched * 0.75), 15)

    # ── 6. STRUCTURE / SECTIONS (max 15) — unchanged ─────────────────────────
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
    


# ─────────────────────────────────────────────────────────
# TRENIRANJE
# ─────────────────────────────────────────────────────────

def train(data_path: str, model_path: str = "models/cv_classifier.pkl"):
    print(f"📂 Učitavam dataset: {data_path}")
    df = pd.read_csv(data_path)

    print(f"🏷️  Labelujem {len(df)} CV-jeva...")
    df["quality_label"] = df["Resume_str"].apply(lambda t: score_cv_quality(t)["label"])

    dist = df["quality_label"].value_counts()
    print(f"   HIGH:   {dist.get('HIGH', 0)}")
    print(f"   MEDIUM: {dist.get('MEDIUM', 0)}")
    print(f"   LOW:    {dist.get('LOW', 0)}")

    X = df["Resume_str"].fillna("")
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n⏳ Treniram model (TF-IDF + Random Forest)...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n📊 REZULTATI NA TEST SETU:")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    size_kb = os.path.getsize(model_path) / 1024
    print(f"✅ Model sačuvan: {model_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Putanja do Resume.csv")
    parser.add_argument("--model", default="models/cv_classifier.pkl")
    args = parser.parse_args()
    train(args.data, args.model)
