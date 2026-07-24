import joblib
from preprocessing.preprocess import preprocess_text
from models.svm.heuristics import heuristic_score
from models.svm.source_reputation import source_score

# Load trained model
model = joblib.load("models/svm/svm_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/svm/tfidf_vectorizer.pkl")


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

    combined_heuristic_score = heuristic_score(text) + source_score(url)

    # Use heuristics only when the SVM is close/moderate, not highly confident.
    if abs(score) < 1.5:
        score = score + (0.35 * combined_heuristic_score)
        prediction = 1 if score > 0 else 0

    confidence = float(abs(score))
    confidence = round(min(confidence * 10, 99.99), 2)

    label = "REAL" if prediction == 1 else "FAKE"

    return label, confidence


# quick test
if __name__ == "__main__":
    print(predict_news("Government announces new policy today"))
