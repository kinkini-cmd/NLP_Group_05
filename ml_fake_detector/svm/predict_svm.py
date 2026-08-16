import joblib
from pathlib import Path

from preprocessing.preprocess import preprocess_text
from ml_fake_detector.svm.heuristics import contains_sinhala, heuristic_score
from ml_fake_detector.svm.source_reputation import source_score


MODEL_DIR = Path(__file__).resolve().parent

# Load trained model
model = joblib.load(MODEL_DIR / "svm_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


def predict_news(text, url=None):
    """
    Predict FAKE / REAL using trained SVM model
    """

    # preprocess text
    cleaned_text = preprocess_text(text)

    # vectorize
    vec = vectorizer.transform([cleaned_text])

    # predict
    prediction = model.predict(vec)[0]

    # confidence using decision_function
    score = model.decision_function(vec)[0]

    text_heuristic_score = heuristic_score(text)
    reputation_score = source_score(url)
    combined_heuristic_score = text_heuristic_score + reputation_score
    is_sinhala = contains_sinhala(text)

    # The current SVM/vectorizer is trained on English. For Sinhala text with
    # no matching TF-IDF vocabulary, fall back to language-specific heuristics.
    if is_sinhala and vec.nnz == 0 and combined_heuristic_score != 0:
        score = combined_heuristic_score
        prediction = 1 if score > 0 else 0

    # Trusted/untrusted source labels are stronger than style heuristics, but
    # only adjust predictions that are not highly confident.
    elif reputation_score and abs(score) < 3:
        score = score + (1.2 * reputation_score)
        prediction = 1 if score > 0 else 0

    # Use heuristics only when the SVM is close/moderate, not highly confident.
    elif abs(score) < 1.5:
        score = score + (0.35 * combined_heuristic_score)
        prediction = 1 if score > 0 else 0

    confidence = float(abs(score))
    confidence = round(min(confidence * 10, 99.99), 2)

    label = "REAL" if prediction == 1 else "FAKE"

    return label, confidence


# quick test
if __name__ == "__main__":
    print(predict_news("Government announces new policy today"))
