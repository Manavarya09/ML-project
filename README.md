# AI-Powered Phishing Email Detection with Explainable Machine Learning

Detect phishing emails using classic ML (TF-IDF + metadata) and a fine-tuned DistilBERT model, with SHAP and LIME explanations for interpretability.

## Project Structure
```
phishing_xai_project/
├── data/
│   └── phishing_emails.csv          # place dataset here (text, label columns)
├── models/                          # saved artifacts after training
│   ├── classic_tfidf_svm.joblib
│   ├── classic_tfidf_random_forest.joblib
│   ├── classic_best_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── metadata_scaler.joblib
│   └── bert_model/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── classic_models.py
│   ├── bert_model.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── inference.py
│   └── cli_demo.py
├── requirements.txt
└── main_train_all.py
```

## Setup
1) Create and activate a virtual environment (Python 3.10+):
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
2) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
3) Place the dataset CSV at `data/phishing_dataset_1900.csv` with columns `email_text` (or `text`) and `label` (0 = ham/safe, 1 = phishing).

> **Note**: Model files are not included in the repository due to size constraints. You must train the models first by running `python main_train_all.py`.

## Training Pipeline
Run the full pipeline (preprocess → classic models → BERT → evaluation):
```bash
python main_train_all.py
```
Artifacts are saved in `models/` and metrics are printed for both classic and BERT models.

## CLI Demo
After training:
```bash
python -m src.cli_demo
```
Paste an email, choose classic or BERT, and view the predicted label, probability, and top contributing terms.

## Components
- Classic models: Logistic Regression, Linear SVM, and Random Forest trained on TF-IDF (uni/bi-grams) plus metadata (URL count, digits, special chars, text length, avg word length, suspicious keywords).
- BERT: DistilBERT (`distilbert-base-uncased`) fine-tuned for binary classification with AdamW and a linear LR scheduler.
- Explainability: SHAP for classic models (TF-IDF + metadata) and LIME text explanations; lightweight token-occlusion importance for BERT.
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix/ROC helpers in `src/evaluation.py`.

## Limitations & Future Work
- Training BERT can be compute-heavy; reduce `bert_max_length`, `bert_batch_size`, or epochs in `src/config.py` if needed.
- Current BERT explanations use a lightweight occlusion heuristic; integrating SHAP's deep/text explainers or Captum would yield richer attributions.
- Add hyperparameter tuning, cross-validation, and more metadata features (HTML tags, sender domain) for improved robustness.
