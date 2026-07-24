import sys
from pathlib import Path

from flask import Flask, request, jsonify, render_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.svm.predict_svm import predict_news as predict_ml_news
from models.xlm_roberta.predict_xlmr import predict_xlmr as predict_dl_news
from models.svm.source_reputation import source_label


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



    # translate non english

    if lang != "en":

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



    if lang != "en":

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
