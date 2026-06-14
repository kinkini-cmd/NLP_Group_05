"""
Improved SVM training with optimized hyperparameters and better preprocessing.
This is the recommended default model for production use.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)

try:
    from src.preprocess import clean_text
except ModuleNotFoundError:
    from preprocess import clean_text


def train_improved_svm():
    """Train improved SVM with better hyperparameters and preprocessing"""
    
    print("=" * 70)
    print("IMPROVED SVM TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading training data...")
    df = pd.read_csv("data/processed/train.csv")
    text_column = "content" if "content" in df.columns else "text"
    
    X = df[text_column]
    y = df["label"]
    
    print(f"   Training samples: {len(df)}")
    print(f"   Fake: {sum(y == 0)}, Real: {sum(y == 1)}")
    print(f"   Class ratio: {sum(y==0)/sum(y==1):.2f}")
    
    # Improved TF-IDF Vectorization
    print("\n2. TF-IDF Feature Extraction...")
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=True,
        strip_accents='unicode',
    )
    
    X_tfidf = vectorizer.fit_transform(X)
    print(f"   Feature matrix shape: {X_tfidf.shape}")
    print(f"   Sparsity: {1.0 - (X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.2%}")
    
    # Hyperparameter tuning with GridSearchCV
    print("\n3. Hyperparameter Tuning (GridSearchCV)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'class_weight': ['balanced', None],
        'dual': [True, False],
    }
    
    svm = LinearSVC(max_iter=2000, random_state=42, loss='squared_hinge')
    
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("   Training with different parameter combinations...")
    grid_search.fit(X_tfidf, y)
    
    print(f"\n   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV F1 score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # Cross-validation evaluation
    print("\n4. Cross-validation Evaluation...")
    cv_f1 = cross_val_score(best_model, X_tfidf, y, cv=cv, scoring='f1_weighted')
    cv_acc = cross_val_score(best_model, X_tfidf, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(best_model, X_tfidf, y, cv=cv, scoring='roc_auc')
    
    print(f"   CV Accuracy:  {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
    print(f"   CV F1 Score:  {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"   CV AUC-ROC:   {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
    
    # Retrain on full dataset
    print("\n5. Retraining on full training set...")
    best_model.fit(X_tfidf, y)
    
    # Save model and vectorizer
    os.makedirs("models/svm", exist_ok=True)
    joblib.dump(best_model, "models/svm/svm_model_improved.pkl")
    joblib.dump(vectorizer, "models/svm/tfidf_vectorizer_improved.pkl")
    print("   Model saved!")
    
    # Evaluate on test set
    print("\n6. Test Set Evaluation...")
    test_df = pd.read_csv("data/processed/test.csv")
    test_text_column = "content" if "content" in test_df.columns else "text"
    
    X_test = vectorizer.transform(test_df[test_text_column])
    y_test = test_df["label"]
    
    y_pred = best_model.predict(X_test)
    y_pred_scores = best_model.decision_function(X_test)
    
    # Normalize scores to [0, 1]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    y_pred_proba = scaler.fit_transform(y_pred_scores.reshape(-1, 1)).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n   TEST SET RESULTS:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}, TP={cm[1,1]}")
    
    print("\n" + "=" * 70)
    print("IMPROVED SVM TRAINING COMPLETED!")
    print("=" * 70)
    
    return best_model, vectorizer


if __name__ == "__main__":
    train_improved_svm()
