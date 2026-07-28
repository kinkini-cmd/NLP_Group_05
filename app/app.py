import sys
from pathlib import Path

from flask import Flask, request, jsonify, render_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml_fake_detector.svm.predict_svm import predict_news as predict_ml_news
from ml_fake_detector.svm.source_reputation import source_label

try:
    from dl_fake_detector.xlm_roberta.predict_xlmr import predict_xlmr as predict_dl_news
    DL_IMPORT_ERROR = None
except Exception as exc:
    predict_dl_news = None
    DL_IMPORT_ERROR = exc


from scraper.url_scraper import extract_article


from language_detection.language_detector import detect_language


from translator.translate import translate_to_english



app = Flask(__name__)


def build_model_result(model_type, model_name, label=None, confidence=None, error=None):
    result = {
        "type": model_type,
        "model": model_name,
    }

    if error:
        result["error"] = str(error)
    else:
        result["prediction"] = label
        result["confidence"] = confidence

    return result


def predict_with_models(text, url=None):
    ml_label, ml_confidence = predict_ml_news(
        text,
        url=url
    )

    ml_result = build_model_result(
        "ML",
        "SVM + TF-IDF",
        ml_label,
        ml_confidence
    )

    if predict_dl_news is None:
        dl_result = build_model_result(
            "DL",
            "XLM-RoBERTa",
            error=f"Model unavailable: {DL_IMPORT_ERROR}"
        )
    else:
        try:
            dl_label, dl_confidence = predict_dl_news(
                text
            )
            dl_result = build_model_result(
                "DL",
                "XLM-RoBERTa",
                dl_label,
                dl_confidence
            )
        except Exception as exc:
            dl_result = build_model_result(
                "DL",
                "XLM-RoBERTa",
                error=exc
            )

    return ml_result, dl_result


def text_metrics(text):
    words = str(text).split()

    return {
        "word_count": len(words),
        "character_count": len(str(text))
    }


def reliability_notes(text, url=None):
    notes = []
    metrics = text_metrics(text)

    if metrics["word_count"] < 40:
        notes.append(
            "Short text: confidence is less stable without full article context."
        )

    if not url:
        notes.append(
            "No source URL: reputation scoring was not used."
        )

    return notes


@app.route("/")
def home():
    return render_template("index.html")




@app.route("/health")
def health():
    return jsonify(
        {
            "message": "Multilingual Fake News Detector API",
            "status": "running"
        }
    )





@app.route(
    "/predict",
    methods=["POST"]
)
def predict():


    data = request.get_json()


    if not data or "text" not in data:

        return jsonify(
            {
                "error":
                "Text required"
            }
        )



    original = data["text"]



    # detect language

    lang = detect_language(
        original
    )



    text = original



    # translate non-English text only when the local model has no language path

    if lang not in {"en", "si"}:

        text = translate_to_english(
            text
        )



    ml_result, dl_result = predict_with_models(
        text
    )



    return jsonify(
        {
            "original_text":
            original,


            "language":
            lang,


            "translated_text":
            text,


            "prediction":
            ml_result["prediction"],


            "confidence":
            ml_result["confidence"],

            "source_reputation":
            None,

            "input_metrics":
            text_metrics(text),

            "reliability_notes":
            reliability_notes(text),


            "ml_result":
            ml_result,


            "dl_result":
            dl_result
        }
    )







@app.route(
    "/predict_url",
    methods=["POST"]
)
def predict_url():


    data = request.get_json()



    if not data or "url" not in data:

        return jsonify(
            {
                "error":
                "URL required"
            }
        )



    url = data["url"]



    article = extract_article(url)



    if "error" in article:

        return jsonify(article)



    text = (

        article["title"]
        +
        " "
        +
        article["text"]

    )



    lang = detect_language(text)



    if lang not in {"en", "si"}:

        text = translate_to_english(text)




    ml_result, dl_result = predict_with_models(
        text,
        url=url
    )



    return jsonify(

        {
            "url":url,

            "title":
            article["title"],

            "language":
            lang,

            "translated_text":
            text,

            "prediction":
            ml_result["prediction"],

            "source_reputation":
            source_label(url),

            "confidence":
            ml_result["confidence"],

            "input_metrics":
            text_metrics(text),

            "reliability_notes":
            reliability_notes(text, url=url),

            "ml_result":
            ml_result,

            "dl_result":
            dl_result
        }

    )




if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
