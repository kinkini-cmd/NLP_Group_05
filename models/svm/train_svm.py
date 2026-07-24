import joblib
import numpy as np

from preprocessing.preprocess import preprocess_text


MODEL_PATH = "models/svm/svm_model.pkl"
VECTORIZER_PATH = "models/svm/tfidf_vectorizer.pkl"


model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_news(text):

    cleaned = preprocess_text(text)

    features = vectorizer.transform(
        [cleaned]
    )


    prediction = model.predict(features)[0]


    # better confidence calculation
    decision = model.decision_function(features)[0]


    confidence = (
        1 /
        (1 + np.exp(-abs(decision)))
    ) * 100


    confidence = round(
        confidence,
        2
    )


    if prediction == 1:
        label = "REAL"
    else:
        label = "FAKE"


    return {
        "prediction": label,
        "confidence": confidence
    }



if __name__ == "__main__":

    result = predict_news(
        "Government announces new policy today"
    )

    print(result)