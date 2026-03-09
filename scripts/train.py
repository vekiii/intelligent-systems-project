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

def score_cv(text: str) -> dict:
    """Ocenjuje kvalitet CV-ja (0-100) po 6 kriterijuma."""
    if not isinstance(text, str) or not text.strip():
        return {"total": 0, "label": "LOW"}

    text_lower = text.lower()
    word_count = len(text_lower.split())
    scores = {}

    # 1. Dužina teksta (max 20)
    if 300 <= word_count <= 1000:
        scores["length"] = 20
    elif 150 <= word_count < 300 or 1000 < word_count <= 1500:
        scores["length"] = 12
    elif word_count > 1500:
        scores["length"] = 6
    else:
        scores["length"] = 2

    # 2. Kontakt info (max 15)
    contact = 0
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        contact += 6
    if re.search(r"(\+?\d[\d\s\-().]{7,15}\d)", text):
        contact += 5
    if any(x in text_lower for x in ["linkedin", "github", "portfolio", "website"]):
        contact += 4
    scores["contact"] = min(contact, 15)

    # 3. Obrazovanje (max 15)
    edu_map = {
        "phd": 15, "doctorate": 15, "ph.d": 15,
        "master": 12, "mba": 12, "msc": 12,
        "bachelor": 9, "b.s": 9, "b.a": 9, "bsc": 9,
        "university": 6, "college": 6,
        "diploma": 4, "certification": 3,
    }
    edu = max((v for k, v in edu_map.items() if k in text_lower), default=0)
    scores["education"] = min(edu, 15)

    # 4. Iskustvo (max 20)
    exp = 0
    m = re.search(r"(\d+)\+?\s*years?\s*(of)?\s*experience", text_lower)
    if m:
        yrs = int(m.group(1))
        exp += 20 if yrs >= 10 else 15 if yrs >= 5 else 10 if yrs >= 2 else 6
    action_words = ["managed", "developed", "led", "designed", "implemented",
                    "achieved", "delivered", "increased", "reduced", "built"]
    exp += min(sum(1 for w in action_words if w in text_lower) * 2, 10)
    scores["experience"] = min(exp, 20)

    # 5. Tehničke veštine (max 15)
    tech = ["python", "java", "javascript", "c++", "c#", "sql", "r", "scala",
            "react", "angular", "vue", "django", "flask", "spring", "node",
            "tensorflow", "pytorch", "sklearn", "pandas", "numpy",
            "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
            "tableau", "power bi", "excel", "spark", "hadoop", "typescript"]
    scores["skills"] = min(sum(1 for k in tech if k in text_lower) * 3, 15)

    # 6. Struktura (max 15)
    sections = ["summary", "objective", "profile", "education", "experience",
                "work history", "skills", "competencies", "projects",
                "achievements", "certifications", "awards", "publications"]
    scores["structure"] = min(sum(1 for s in sections if s in text_lower) * 3, 15)

    total = sum(scores.values())
    scores["total"] = total
    scores["label"] = "HIGH" if total >= 65 else "MEDIUM" if total >= 35 else "LOW"
    return scores


# ─────────────────────────────────────────────────────────
# TRENIRANJE
# ─────────────────────────────────────────────────────────

def train(data_path: str, model_path: str = "models/cv_classifier.pkl"):
    print(f"📂 Učitavam dataset: {data_path}")
    df = pd.read_csv(data_path)

    print(f"🏷️  Labelujem {len(df)} CV-jeva...")
    df["quality_label"] = df["Resume_str"].apply(lambda t: score_cv(t)["label"])

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
