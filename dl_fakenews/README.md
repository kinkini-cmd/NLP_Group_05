# Deep Learning Fake News Detector

This app detects fake or real news with the saved XLM-RoBERTa deep-learning model.

Runtime detection flow:

1. `run.py` starts the Flask app in `app/app.py`.
2. The API receives article text or a news URL.
3. Language detection and translation run first.
4. `dl_model/xlm_roberta/predict_xlmr.py` returns the XLM-RoBERTa prediction and confidence.
5. The API response includes `prediction`, `confidence`, and `dl_result`.

See [MODEL_FILE_SPLIT.md](MODEL_FILE_SPLIT.md) for the current file layout.

## Step-by-Step Accuracy Workflow

1. Use full article text for normal detection. The model was trained on article bodies, not short claims or headlines.

2. Short inputs are marked `UNCERTAIN`. Inputs under 500 characters are rejected for reliable article detection because they caused real claims to be classified as fake.

3. Run the included short-claim check:

```bash
venv312/bin/python tests/evaluate_dl_samples.py tests/external_claim_samples.csv
```

4. To reproduce the old short-claim behavior, force evaluation with:

```bash
venv312/bin/python tests/evaluate_dl_samples.py tests/external_claim_samples.csv --include-short
```

5. For a proper external accuracy test, create a CSV with full article bodies:

```csv
label,text,source
REAL,"Full real article text here","https://source-url"
FAKE,"Full fake article text here","https://source-url"
```

Then run:

```bash
venv312/bin/python tests/evaluate_dl_samples.py path/to/full_article_samples.csv
```

## Claim Model Fine-Tuning

The original article model performs poorly on short claims. To train a separate short-text model:

```bash
venv312/bin/python dl_model/xlm_roberta/build_short_text_dataset.py \
  --output data/short_text_training.csv \
  --max-per-label 100
```

```bash
venv312/bin/python dl_model/xlm_roberta/train_claim_xlmr.py \
  --claims-csv data/short_text_training.csv \
  --output-dir dl_model/xlm_roberta/claim_model \
  --epochs 5 \
  --batch-size 32 \
  --freeze-base \
  --learning-rate 0.001 \
  --max-length 64
```

Evaluate short internet claims with:

```bash
venv312/bin/python tests/evaluate_dl_samples.py \
  tests/external_claim_samples.csv \
  --claim-model
```

Current CPU-trained claim model result:

```text
validation accuracy on local short-title split: 72.50%
accuracy on 10 external short claims: 50.00%
```

Because external claim confidence is near 50%, the app returns `UNCERTAIN` for low-confidence claim-model decisions.
