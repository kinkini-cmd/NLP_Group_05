# Deep Learning Model File Layout

Runtime detection now uses only the XLM-RoBERTa deep-learning model.

## XLM-RoBERTa DL Model Files

| File or folder | Purpose |
| --- | --- |
| `dl_model/xlm_roberta/predict_xlmr.py` | Loads the saved XLM-RoBERTa model and predicts `FAKE` or `REAL`. |
| `dl_model/xlm_roberta/train_xlmr.py` | Training script location for the XLM-RoBERTa model. |
| `dl_model/xlm_roberta/saved_model/` | Saved tokenizer, config, weights, and training args. |
| `notebooks/04_xlmr_training.ipynb` | Notebook for XLM-RoBERTa training. |
| `test_model.py` | Quick XLM-RoBERTa prediction test. |
| `test_multiple.py` | Multiple-sample XLM-RoBERTa prediction test. |

## Application Files

| File or folder | Purpose |
| --- | --- |
| `app/app.py` | Flask app. Calls only the XLM-RoBERTa prediction path. |
| `run.py` | Starts the Flask app. |
| `app/templates/index.html` | Web UI. |
| `app/static/app.js` | Frontend JavaScript. |
| `app/static/styles.css` | Frontend styles. |
| `language_detection/language_detector.py` | Detects whether input text is English or Sinhala. |
| `translator/translate.py` | Translates non-English text to English before prediction. |
| `scraper/url_scraper.py` | Extracts article text from URLs. |
| `datasets/raw/Fake.csv` | Raw fake-news training data. |
| `datasets/raw/True.csv` | Raw real-news training data. |
| `data/processed_news.csv` | Processed dataset output. |
| `requirements.txt` | Dependencies for the DL model and web app. |
| `Dockerfile` | Container setup. |
| `docker-compose.yml` | Compose setup. |

## Historical Comparison Artifacts

The notebooks and CSV files that compare SVM and XLM-RoBERTa are kept as historical experiment artifacts:

| File or folder | Purpose |
| --- | --- |
| `notebooks/03_svm_training.ipynb` | Earlier SVM training notebook. |
| `notebooks/05_model_comparison.ipynb` | Earlier model comparison notebook. |
| `notebooks/model_comparison.csv` | Earlier model comparison data. |
| `notebooks/results/model_comparison.csv` | Earlier model comparison result output. |

## Runtime Flow

1. `run.py` starts `app/app.py`.
2. The API receives text or a URL.
3. Language detection and translation run first.
4. `dl_model/xlm_roberta/predict_xlmr.py` returns the XLM-RoBERTa DL prediction.
5. The API response includes `prediction`, `confidence`, and `dl_result`.
