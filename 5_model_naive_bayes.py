import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load TF-IDF data
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

X_train = pd.read_csv("data/X_train.csv").squeeze()
X_test  = pd.read_csv("data/X_test.csv").squeeze()
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Train
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = nb_model.predict(X_test_tfidf)
y_prob = nb_model.predict_proba(X_test_tfidf)[:, 1]

print("=== Naive Bayes Results ===")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake","Real"], yticklabels=["Fake","Real"])
plt.title("Naive Bayes Confusion Matrix")
plt.savefig("nb_confusion_matrix.png")
plt.show()

# Save model
with open("models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

print("Naive Bayes model saved.")