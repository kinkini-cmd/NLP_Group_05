from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent / "saved_model"
CLAIM_MODEL_PATH = Path(__file__).resolve().parent / "claim_model"
LABELS = {
    0: "FAKE",
    1: "REAL",
}

_cache = {}
_torch = None


def has_claim_model():
    return CLAIM_MODEL_PATH.exists()


def _load_model(model_path=MODEL_PATH):
    global _torch

    model_path = Path(model_path)
    cache_key = str(model_path.resolve())

    if cache_key in _cache:
        tokenizer, model = _cache[cache_key]
        return tokenizer, model, _torch

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "DL dependencies are missing. Install torch and transformers to use the XLM-R model."
        ) from exc

    if not model_path.exists():
        raise RuntimeError(f"DL model folder not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    model.eval()

    _cache[cache_key] = (tokenizer, model)
    _torch = torch

    return tokenizer, model, _torch


def predict_xlmr(text, model_path=MODEL_PATH):
    """
    Predict FAKE / REAL using the saved deep-learning classifier.
    """

    tokenizer, model, torch = _load_model(model_path)

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    prediction = int(torch.argmax(probabilities).item())
    confidence = round(float(probabilities[prediction].item()) * 100, 2)
    label = LABELS.get(prediction, str(prediction))

    return label, confidence


def predict_news(text):
    return predict_xlmr(text)


def predict_claim(text):
    return predict_xlmr(text, CLAIM_MODEL_PATH)


if __name__ == "__main__":
    print(predict_xlmr("Government announces new policy today"))
