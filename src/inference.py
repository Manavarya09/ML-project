"""Inference helpers for classic and BERT models."""

import joblib
import numpy as np
import torch

from src import preprocessing
from src.bert_model import load_finetuned_model
from src.config import config
from src.explainability import explain_bert_tokens, explain_single_email_shap_classic
from src.features import combine_features, transform_metadata, transform_tfidf
from src.utils import LOGGER


def _load_classic_artifacts():
    vectorizer = joblib.load(config.tfidf_vectorizer_path)
    scaler = joblib.load(config.metadata_scaler_path)
    try:
        model = joblib.load(config.classic_best_model_path)
        LOGGER.info("Loaded best classic model.")
        return model, vectorizer, scaler
    except FileNotFoundError:
        pass
    try:
        model = joblib.load(config.classic_tfidf_svm_path)
        LOGGER.info("Loaded classic SVM model.")
    except FileNotFoundError:
        model = joblib.load(config.classic_tfidf_rf_path)
        LOGGER.info("Loaded classic Random Forest model.")
    return model, vectorizer, scaler


def predict_email_classic(text: str):
    """
    Predict phishing probability using the classic model and return explanation.

    Returns:
        label (int), probability (float), explanation (list)
    """
    model, vectorizer, scaler = _load_classic_artifacts()
    cleaned = preprocessing.clean_text(text)
    tfidf = transform_tfidf([cleaned], vectorizer)
    meta = transform_metadata([cleaned], scaler)
    features = combine_features(tfidf, meta)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0, 1]
    else:
        decision = model.decision_function(features)
        prob_pos = 1 / (1 + np.exp(-decision))
        proba = float(np.ravel(prob_pos)[0])
    label = int(proba >= 0.5)

    try:
        explanation = explain_single_email_shap_classic(text, model, vectorizer, scaler)
    except Exception as exc:  # pragma: no cover - best-effort
        LOGGER.warning("SHAP explanation failed: %s", exc)
        explanation = []

    return label, float(proba), explanation


def predict_email_bert(text: str):
    """
    Predict phishing probability using the fine-tuned BERT model and return explanation.

    Returns:
        label (int), probability (float), explanation (List[Dict])
    """
    model, tokenizer, device = load_finetuned_model()
    encodings = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=config.bert_max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    proba = float(probs[1])
    label = int(np.argmax(probs))

    try:
        explanation = explain_bert_tokens(text)
    except Exception as exc:  # pragma: no cover - best-effort
        LOGGER.warning("BERT explanation failed: %s", exc)
        explanation = []

    return label, proba, explanation
