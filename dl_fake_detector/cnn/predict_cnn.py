import os
from pathlib import Path

try:
    import tensorflow as tf
    from transformers import BertTokenizer
    HAS_DL_DEPS = True
except Exception:
    HAS_DL_DEPS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_model.h5"
MAX_LEN = 128

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if not HAS_DL_DEPS:
        raise RuntimeError("TensorFlow / transformers not installed")
    if _model is None and MODEL_PATH.exists():
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model is not None


def predict_news(text):
    if not _load():
        raise RuntimeError(f"DL model not found at {MODEL_PATH}")
    tokens = _tokenizer(
        [text],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    prob = float(_model.predict(tokens["input_ids"], verbose=0).flatten()[0])
    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = round(prob * 100, 2) if prob >= 0.5 else round((1 - prob) * 100, 2)
    return label, confidence
