# Model File Split

This project has two model paths:

- SVM machine-learning model: TF-IDF features plus a Linear SVM classifier.
- XLM-RoBERTa deep-learning model: transformer sequence classification.

## SVM ML Model Files

These files belong to the SVM/TF-IDF machine-learning pipeline.

| File or folder | Purpose |
| --- | --- |
| `ml_fake_detector/svm/train_svm.py` | Trains the Linear SVM model and saves the model/vectorizer. |
| `ml_fake_detector/svm/retrain_svm.py` | Retrains the SVM model from the raw fake/true datasets. |
| `ml_fake_detector/svm/predict_svm.py` | Loads the saved SVM model and predicts `FAKE` or `REAL`. |
| `ml_fake_detector/svm/svm_model.pkl` | Saved trained SVM model. |
| `ml_fake_detector/svm/tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer used by the SVM model. |
| `ml_fake_detector/svm/heuristics.py` | Extra rule-based scoring used with the SVM prediction. |
| `ml_fake_detector/svm/source_reputation.py` | Domain reputation scoring used by URL predictions. |
| `feature_extraction/vectorizer.py` | Feature extraction support for the ML pipeline. |
| `preprocessing/preprocess.py` | Text cleaning, stopword removal, and lemmatization used before SVM vectorization. |
| `notebooks/03_svm_training.ipynb` | Notebook for SVM model training. |
| `test_model.py` | Quick SVM prediction test. |
| `test_multiple.py` | Multiple-sample SVM prediction test. |

## XLM-RoBERTa DL Model Files

These files belong to the XLM-RoBERTa deep-learning pipeline.

| File or folder | Purpose |
| --- | --- |
| `dl_fake_detector/xlm_roberta/predict_xlmr.py` | Loads the saved XLM-RoBERTa model and predicts `FAKE` or `REAL`. |
| `dl_fake_detector/xlm_roberta/train_xlmr.py` | Training script location for the XLM-RoBERTa model. |
| `notebooks/04_xlmr_training.ipynb` | Notebook for XLM-RoBERTa training. |
| `dl_fake_detector/xlm_roberta/saved_model/` | Saved XLM-RoBERTa tokenizer, config, weights, and training args. |

## Shared Application Files

These files are used by both model paths or by the web/API layer.

| File or folder | Purpose |
| --- | --- |
| `app/app.py` | Flask app. Calls both SVM and XLM-RoBERTa predictions. |
| `app/routes.py` | Older route file that only uses SVM prediction. |
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
| `data/source_reputation.csv` | Domain reputation data for URL predictions. |
| `notebooks/01_data_exploration.ipynb` | Dataset exploration. |
| `notebooks/02_preprocessing.ipynb` | Preprocessing workflow. |
| `notebooks/05_model_comparison.ipynb` | Compares SVM and XLM-RoBERTa results. |
| `notebooks/model_comparison.csv` | Model comparison data. |
| `notebooks/results/model_comparison.csv` | Model comparison result output. |
| `requirements.txt` | Dependencies for both model paths and the web app. |
| `Dockerfile` | Container setup. |
| `docker-compose.yml` | Compose setup. |

## Runtime Flow

1. `run.py` starts `app/app.py`.
2. The API receives text or a URL.
3. Language detection and translation run first.
4. `ml_fake_detector/svm/predict_svm.py` returns the SVM ML prediction.
5. `dl_fake_detector/xlm_roberta/predict_xlmr.py` returns the XLM-RoBERTa DL prediction.
6. The API response includes both `ml_result` and `dl_result`.
