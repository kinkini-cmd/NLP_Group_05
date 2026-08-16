import sys
from pathlib import Path

import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.preprocess import preprocess_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


MODEL_DIR = Path(__file__).resolve().parent

fake = pd.read_csv("datasets/raw/Fake.csv")
real = pd.read_csv("datasets/raw/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)

try:
    short = pd.read_csv(MODEL_DIR.parents[1] / "data" / "short_text_training.csv")
    if {"label", "text"}.issubset(short.columns):
        short = short[["label", "text"]].copy()
        df = pd.concat([df, short], ignore_index=True)
except FileNotFoundError:
    pass

df["clean_text"] = df["text"].apply(preprocess_text)

X = df["clean_text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LinearSVC()

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(
    model,
    MODEL_DIR / "svm_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)
