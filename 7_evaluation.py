import pickle
import pandas as pd  # type: ignore
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Load data
X_test  = pd.read_csv("data/X_test.csv").squeeze().fillna("").tolist()
y_test  = pd.read_csv("data/y_test.csv").squeeze().values

# ── Naive Bayes ────────────────────────────────────────
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models/naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

X_test_tfidf = tfidf.transform(X_test)
nb_pred = nb_model.predict(X_test_tfidf)
nb_prob = nb_model.predict_proba(X_test_tfidf)[:, 1]

# ── CNN ────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer(X_test, max_length=128, padding="max_length",
                    truncation=True, return_tensors="np")
X_test_ids = encoded["input_ids"]

cnn_model = tf.keras.models.load_model("models/cnn_model.h5")
cnn_prob = cnn_model.predict(X_test_ids).flatten()
cnn_pred = (cnn_prob >= 0.5).astype(int)

# ── Comparison Table ───────────────────────────────────
results = {
    "Model":     ["Naive Bayes (TF-IDF)", "CNN (BERT Tokenizer)"],
    "Accuracy":  [accuracy_score(y_test, nb_pred),  accuracy_score(y_test, cnn_pred)],
    "F1-Score":  [f1_score(y_test, nb_pred),        f1_score(y_test, cnn_pred)],
    "ROC-AUC":   [roc_auc_score(y_test, nb_prob),   roc_auc_score(y_test, cnn_prob)],
}

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Bar Chart Comparison
df_results.set_index("Model")[["Accuracy","F1-Score","ROC-AUC"]].plot(
    kind="bar", figsize=(10, 5), colormap="Set2"
)
plt.title("Model Comparison"); plt.xticks(rotation=15); plt.ylim(0.8, 1.0)
plt.savefig("model_comparison.png"); plt.show()