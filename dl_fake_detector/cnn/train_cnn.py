import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# ── Load Data ──────────────────────────────────────────
X_train = pd.read_csv("data/X_train.csv").squeeze().fillna("").tolist()
X_test  = pd.read_csv("data/X_test.csv").squeeze().fillna("").tolist()
y_train = pd.read_csv("data/y_train.csv").squeeze().values
y_test  = pd.read_csv("data/y_test.csv").squeeze().values

# ── BERT Tokenizer ─────────────────────────────────────
MAX_LEN = 128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(texts):
    encoded = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    return encoded["input_ids"]

print("Tokenizing training data...")
X_train_ids = tokenize(X_train)
print("Tokenizing test data...")
X_test_ids  = tokenize(X_test)

# ── CNN Model ──────────────────────────────────────────
VOCAB_SIZE = tokenizer.vocab_size  # 30522

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
    Conv1D(filters=128, kernel_size=5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ── Train ──────────────────────────────────────────────
history = model.fit(
    X_train_ids, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

# ── Evaluate ───────────────────────────────────────────
y_prob = model.predict(X_test_ids).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("=== CNN Results ===")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Fake","Real"], yticklabels=["Fake","Real"])
plt.title("CNN Confusion Matrix")
plt.savefig("cnn_confusion_matrix.png")
plt.show()

# Training curves
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("CNN Training Accuracy"); plt.legend()
plt.savefig("cnn_training_curve.png"); plt.show()

# Save model
model.save("models/cnn_model.h5")
print("CNN model saved.")