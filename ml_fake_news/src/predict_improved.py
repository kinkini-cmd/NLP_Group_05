
"""
Improved prediction module supporting multiple model backends.
Automatically selects the best available model.
"""

import os
import joblib
import math
import numpy as np

try:
    from src.preprocess import clean_text
    from src.advanced_features import extract_linguistic_features
except ModuleNotFoundError:
    from preprocess import clean_text
    from advanced_features import extract_linguistic_features


class ModelSelector:
    """Intelligently select and load the best available model"""
    
    @staticmethod
    def get_best_model():
        """Select model in order of preference"""
        model_paths = [
            ("ensemble", "models/ensemble/ensemble_model.pkl", "Ensemble"),
            ("improved_svm", "models/svm/svm_model_improved.pkl", "Improved SVM"),
            ("transformer", "models/transformer/distilbert_classifier.pkl", "Transformer"),
            ("standard", "models/svm/svm_model.pkl", "Standard SVM"),
        ]
        
        for model_type, path, name in model_paths:
            if os.path.exists(path):
                print(f"Loading {name} model from {path}")
                return model_type, path, name
        
        raise FileNotFoundError("No models found! Please train models first.")
    
    @staticmethod
    def load_ensemble_model():
        """Load ensemble model with all required components"""
        model = joblib.load("models/ensemble/ensemble_model.pkl")
        vectorizer = joblib.load("models/ensemble/tfidf_vectorizer.pkl")
        scaler = joblib.load("models/ensemble/linguistic_scaler.pkl")
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'scaler': scaler,
            'type': 'ensemble'
        }
    
    @staticmethod
    def load_svm_model():
        """Load SVM model with vectorizer"""
        try:
            model = joblib.load("models/svm/svm_model_improved.pkl")
            vectorizer = joblib.load("models/svm/tfidf_vectorizer_improved.pkl")
        except FileNotFoundError:
            model = joblib.load("models/svm/svm_model.pkl")
            vectorizer = joblib.load("models/svm/tfidf_vectorizer.pkl")
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'type': 'svm'
        }


# Initialize model
try:
    model_type, model_path, model_name = ModelSelector.get_best_model()
    print(f"Using {model_name} model")
    
    if model_type == "ensemble":
        loaded_model = ModelSelector.load_ensemble_model()
    elif "svm" in model_type:
        loaded_model = ModelSelector.load_svm_model()
    else:
        # Fallback to SVM
        loaded_model = ModelSelector.load_svm_model()
        
except Exception as e:
    print(f"Warning: {e}")
    loaded_model = None


def label_from_score(score, real_threshold=0.5, fake_threshold=-0.5):
    """Convert decision score to label"""
    if score >= real_threshold:
        return "REAL"
    if score <= fake_threshold:
        return "FAKE"
    return "UNCERTAIN"


def percentages_from_score(score):
    """Convert score to percentages using sigmoid"""
    # Clamp score to reasonable range
    score = max(-5, min(5, score))
    real_percentage = 100 / (1 + math.exp(-score * 2))
    fake_percentage = 100 - real_percentage
    return fake_percentage, real_percentage


def get_prediction_ensemble(text):
    """Get prediction using ensemble model"""
    from src.advanced_features import extract_features_dataframe, combine_tfidf_and_linguistic
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None
    
    vectorizer = loaded_model['vectorizer']
    scaler = loaded_model['scaler']
    model = loaded_model['model']
    
    # Get TF-IDF features
    X_tfidf = vectorizer.transform([cleaned_text])
    
    # Get linguistic features
    X_linguistic = extract_features_dataframe([cleaned_text])
    
    # Combine features
    X_combined = combine_tfidf_and_linguistic(X_tfidf, X_linguistic)[0]
    
    # Get probability
    proba = model.predict_proba(X_combined)[0]
    
    # Convert to decision score (real - fake)
    score = np.log(proba[1] / (proba[0] + 1e-10)) if proba[0] > 0 else 5.0
    
    return score


def get_prediction_svm(text):
    """Get prediction using SVM model"""
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None
    
    vectorizer = loaded_model['vectorizer']
    model = loaded_model['model']
    
    # Get decision function score
    score = model.decision_function(vectorizer.transform([cleaned_text]))[0]
    
    return score


def get_prediction_score(text):
    """Get prediction score from the best available model"""
    if loaded_model is None:
        return None
    
    try:
        if loaded_model['type'] == 'ensemble':
            return get_prediction_ensemble(text)
        else:  # SVM
            return get_prediction_svm(text)
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return None


def predict_news_details(text):
    """
    Make detailed prediction for news text.
    
    Returns:
        dict: {
            'label': 'FAKE'|'REAL'|'UNCERTAIN'|'EMPTY',
            'fake_percentage': float,
            'real_percentage': float,
            'confidence': float (0-100),
            'model_used': str
        }
    """
    score = get_prediction_score(text)
    
    if score is None:
        return {
            "label": "EMPTY",
            "fake_percentage": 0.0,
            "real_percentage": 0.0,
            "confidence": 0.0,
            "model_used": loaded_model['type'] if loaded_model else "unknown"
        }
    
    fake_percentage, real_percentage = percentages_from_score(score)
    confidence = abs(score) / 5.0 * 100 if score != 0 else 50.0  # Normalize to 0-100
    confidence = min(100, max(0, confidence))
    
    return {
        "label": label_from_score(score),
        "fake_percentage": round(fake_percentage, 2),
        "real_percentage": round(real_percentage, 2),
        "confidence": round(confidence, 2),
        "model_used": loaded_model['type'] if loaded_model else "unknown"
    }


def predict_news(text):
    """Simple prediction returning just the label"""
    return predict_news_details(text)["label"]


if __name__ == "__main__":
    news = input("Enter news text: ")
    result = predict_news_details(news)
    print(f"\nPrediction: {result['label']}")
    print(f"Fake: {result['fake_percentage']}%")
    print(f"Real: {result['real_percentage']}%")
    print(f"Confidence: {result['confidence']}%")
    print(f"Model: {result['model_used']}")
