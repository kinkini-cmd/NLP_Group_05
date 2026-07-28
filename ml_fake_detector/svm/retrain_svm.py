import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


MODEL_DIR = Path(__file__).resolve().parent

fake = pd.read_csv("datasets/raw/Fake.csv")
real = pd.read_csv("datasets/raw/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], ignore_index=True)

X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
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
