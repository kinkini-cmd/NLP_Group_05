import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dl_model.xlm_roberta.predict_xlmr import (
    has_claim_model,
    predict_claim,
    predict_xlmr as predict_dl_news,
)
from language_detection.language_detector import detect_language
from scraper.url_scraper import extract_article
from translator.translate import translate_to_english


app = Flask(__name__)

MIN_RELIABLE_CHARS = 500
MIN_CLAIM_CONFIDENCE = 60


def build_model_result(label=None, confidence=None, error=None, model="XLM-RoBERTa"):
    result = {
        "type": "DL",
        "model": model,
    }

    if error:
        result["error"] = str(error)
    else:
        result["prediction"] = label
        result["confidence"] = confidence

    return result


def predict_with_deep_learning(text):
    try:
        label, confidence = predict_dl_news(text)
        return build_model_result(label, confidence)
    except Exception as exc:
        return build_model_result(error=exc)


def predict_with_claim_model(text):
    try:
        label, confidence = predict_claim(text)

        if confidence < MIN_CLAIM_CONFIDENCE:
            return {
                "type": "DL",
                "model": "XLM-RoBERTa Claim Fine-tuned",
                "prediction": "UNCERTAIN",
                "confidence": confidence,
                "reason": (
                    "Claim model confidence is too low for a reliable "
                    f"REAL/FAKE decision ({confidence}%)."
                ),
            }

        return build_model_result(
            label,
            confidence,
            model="XLM-RoBERTa Claim Fine-tuned",
        )
    except Exception as exc:
        return build_model_result(
            error=exc,
            model="XLM-RoBERTa Claim Fine-tuned",
        )


def build_uncertain_result(text):
    return {
        "type": "DL",
        "model": "XLM-RoBERTa",
        "prediction": "UNCERTAIN",
        "confidence": None,
        "reason": (
            f"Input is too short for reliable article detection "
            f"({len(text)} characters). Paste the full article text."
        ),
    }


def is_too_short_for_article_detection(text):
    return len(str(text).strip()) < MIN_RELIABLE_CHARS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "message": "Deep Learning Fake News Detector API",
            "model": "XLM-RoBERTa",
            "status": "running",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text required"}), 400

    original = data["text"]
    lang = detect_language(original)
    text = original

    if lang != "en":
        text = translate_to_english(text)

    if is_too_short_for_article_detection(text) and has_claim_model():
        dl_result = predict_with_claim_model(text)
    elif is_too_short_for_article_detection(text):
        dl_result = build_uncertain_result(text)
    else:
        dl_result = predict_with_deep_learning(text)

    if "error" in dl_result:
        return jsonify(dl_result), 500

    return jsonify(
        {
            "original_text": original,
            "language": lang,
            "translated_text": text,
            "prediction": dl_result["prediction"],
            "confidence": dl_result["confidence"],
            "dl_result": dl_result,
            "minimum_reliable_characters": MIN_RELIABLE_CHARS,
        }
    )


@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json()

    if not data or "url" not in data:
        return jsonify({"error": "URL required"}), 400

    url = data["url"]
    article = extract_article(url)

    if "error" in article:
        return jsonify(article), 400

    text = f'{article["title"]} {article["text"]}'
    lang = detect_language(text)

    if lang != "en":
        text = translate_to_english(text)

    if is_too_short_for_article_detection(text) and has_claim_model():
        dl_result = predict_with_claim_model(text)
    elif is_too_short_for_article_detection(text):
        dl_result = build_uncertain_result(text)
    else:
        dl_result = predict_with_deep_learning(text)

    if "error" in dl_result:
        return jsonify(dl_result), 500

    return jsonify(
        {
            "url": url,
            "title": article["title"],
            "language": lang,
            "translated_text": text,
            "prediction": dl_result["prediction"],
            "confidence": dl_result["confidence"],
            "dl_result": dl_result,
            "minimum_reliable_characters": MIN_RELIABLE_CHARS,
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
