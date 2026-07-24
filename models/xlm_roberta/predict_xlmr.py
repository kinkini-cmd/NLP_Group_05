from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "models" / "xlmr"
LABELS = {
    0: "FAKE",
    1: "REAL",
}

_tokenizer = None
_model = None
_torch = None


def _load_model():
    global _tokenizer, _model, _torch

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model, _torch

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "DL dependencies are missing. Install torch and transformers to use the XLM-R model."
        ) from exc

    if not MODEL_PATH.exists():
        raise RuntimeError(f"DL model folder not found: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )
    model.eval()

    _tokenizer = tokenizer
    _model = model
    _torch = torch

    return _tokenizer, _model, _torch


def predict_xlmr(text):
    """
    Predict FAKE / REAL using the saved deep-learning classifier.
    """

    tokenizer, model, torch = _load_model()

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


if __name__ == "__main__":
    print(predict_xlmr("Government announces new policy today"))
