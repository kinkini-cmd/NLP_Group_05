"""
Ensemble model combining multiple classifiers for robust fake news detection.
Uses voting to combine predictions from SVM, Logistic Regression, and Random Forest.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

try:
    from src.advanced_features import extract_features_dataframe, combine_tfidf_and_linguistic
except ModuleNotFoundError:
    from advanced_features import extract_features_dataframe, combine_tfidf_and_linguistic


def train_ensemble():
    """Train ensemble model with multiple classifiers"""
    
    print("=" * 70)
    print("ENSEMBLE MODEL TRAINING WITH ADVANCED FEATURES")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading and preparing training data...")
    df = pd.read_csv("data/processed/train.csv")
    text_column = "content" if "content" in df.columns else "text"
    
    X_text = df[text_column]
    y = df["label"]
    
    print(f"   Training samples: {len(df)}")
    print(f"   Fake: {sum(y == 0)}, Real: {sum(y == 1)}")
    
    # Feature Engineering - TF-IDF
    print("\n2. Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=8000,
        min_df=2,
        max_df=0.9,
        stop_words="english",
        lowercase=True,
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"   TF-IDF shape: {X_tfidf.shape}")
    
    # Feature Engineering - Linguistic Features
    print("\n3. Extracting linguistic features...")
    X_linguistic = extract_features_dataframe(X_text)
    print(f"   Linguistic features shape: {X_linguistic.shape}")
    print(f"   Features: {list(X_linguistic.columns)[:5]}...")
    
    # Combine features
    print("\n4. Combining feature sets...")
    X_combined, scaler, linguistic_names = combine_tfidf_and_linguistic(X_tfidf, X_linguistic)
    print(f"   Combined feature shape: {X_combined.shape}")
    
    # Save artifacts for later use
    os.makedirs("models/ensemble", exist_ok=True)
    joblib.dump(vectorizer, "models/ensemble/tfidf_vectorizer.pkl")
    joblib.dump(scaler, "models/ensemble/linguistic_scaler.pkl")
    joblib.dump(linguistic_names, "models/ensemble/linguistic_names.pkl")
    
    # Build ensemble
    print("\n5. Building ensemble classifier...")
    
    estimators = [
        ('svm', LinearSVC(C=1, max_iter=2000, random_state=42, dual=False, class_weight='balanced')),
        ('lr', LogisticRegression(C=10, max_iter=1000, random_state=42, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, 
                                     class_weight='balanced', n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                         random_state=42, subsample=0.8)),
    ]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use probability estimates
        n_jobs=-1
    )
    
    # Cross-validation
    print("\n6. Cross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_combined, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    cv_f1 = cross_val_score(ensemble, X_combined, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"   CV F1 (weighted): {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    
    # Train on full training set
    print("\n7. Training ensemble on full dataset...")
    ensemble.fit(X_combined, y)
    
    # Save ensemble
    joblib.dump(ensemble, "models/ensemble/ensemble_model.pkl")
    print("   Ensemble model saved!")
    
    # Evaluate on test set
    print("\n8. Evaluating on test set...")
    test_df = pd.read_csv("data/processed/test.csv")
    test_text = test_df[text_column]
    y_test = test_df["label"]
    
    X_test_tfidf = vectorizer.transform(test_text)
    X_test_linguistic = extract_features_dataframe(test_text)
    X_test_combined = combine_tfidf_and_linguistic(X_test_tfidf, X_test_linguistic)[0]
    
    y_pred = ensemble.predict(X_test_combined)
    y_pred_proba = ensemble.predict_proba(X_test_combined)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
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
    print("ENSEMBLE TRAINING COMPLETED!")
    print("=" * 70)
    
    return ensemble, vectorizer, scaler


if __name__ == "__main__":
    train_ensemble()
