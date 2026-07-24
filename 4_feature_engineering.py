import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data/preprocessed_dataset.csv")
df.dropna(subset=["clean_text"], inplace=True)

X = df["clean_text"]
y = df["label"]

# ── SPLIT ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── TF-IDF (for Naive Bayes) ───────────────────────────
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ensure models directory exists
os.makedirs("models", exist_ok=True)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save splits for reuse
df_split = pd.DataFrame({"text": X_test, "label": y_test})
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",  index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",  index=False)

print("TF-IDF shape:", X_train_tfidf.shape)
print("Feature engineering complete.")
