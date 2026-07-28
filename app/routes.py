@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.json
    url = data.get("url", "")

    article = extract_article(url)

    if "error" in article:
        return jsonify({"error": article["error"]})

    text = article["title"] + " " + article["text"]

    label, confidence = predict_news(text)

    return jsonify({
        "url": url,
        "title": article["title"],
        "prediction": label,
        "confidence": confidence
    })