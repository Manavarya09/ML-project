"""Explainability utilities using SHAP and LIME."""

from typing import Iterable, List, Tuple

import numpy as np
import numpy as np
import torch

from src import preprocessing
from src.bert_model import load_finetuned_model
from src.config import config
from src.features import combine_features, transform_metadata, transform_tfidf
from src.utils import LOGGER


def _wrap_predict_proba(model, vectorizer, scaler):
    """Create a prediction function compatible with SHAP/LIME."""

    def predict_fn(texts: List[str]) -> np.ndarray:
        cleaned = [preprocessing.clean_text(t) for t in texts]
        tfidf = transform_tfidf(cleaned, vectorizer)
        meta = transform_metadata(cleaned, scaler)
        features = combine_features(tfidf, meta)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)
        # Fall back to decision function turned into probabilities
        decision = model.decision_function(features)
        prob_pos = 1 / (1 + np.exp(-decision))
        prob_pos = np.clip(prob_pos, 1e-6, 1 - 1e-6)
        return np.vstack([1 - prob_pos, prob_pos]).T

    return predict_fn


def explain_single_email_shap_classic(
    text: str, model, vectorizer, scaler, top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Compute SHAP values for a single email using the classic model stack.

    Returns:
        List of (feature, shap_value) sorted by absolute importance.
    """
    cleaned = preprocessing.clean_text(text)
    tfidf = transform_tfidf([cleaned], vectorizer)
    meta = transform_metadata([cleaned], scaler)
    combined_sparse = combine_features(tfidf, meta)
    combined = combined_sparse.toarray()

    feature_names = list(vectorizer.get_feature_names_out()) + [
        "num_urls",
        "num_digits",
        "num_special_chars",
        "text_length",
        "avg_word_length",
        "suspicious_keyword",
    ]

    import shap
    explainer = shap.Explainer(model, combined, feature_names=feature_names)
    shap_values = explainer(combined)
    values = np.array(shap_values.values)[0]
    ranked_idx = np.argsort(np.abs(values))[::-1][:top_k]
    return [(feature_names[i], float(values[i])) for i in ranked_idx]


def explain_single_email_lime_classic(
    text: str, model, vectorizer, scaler, top_k: int = 10
) -> List[Tuple[str, float]]:
    """Generate a LIME explanation for a single email."""
    predict_fn = _wrap_predict_proba(model, vectorizer, scaler)
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["ham", "phishing"])
    explanation = explainer.explain_instance(text, predict_fn, num_features=top_k)
    return explanation.as_list()


def explain_bert_tokens(text: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Lightweight token-level importance for the fine-tuned BERT model.

    Uses an occlusion-based approach: remove one token at a time and
    observe probability drop for the phishing class.
    """
    model, tokenizer, device = load_finetuned_model()
    model.eval()

    # Tokenize and get baseline probability
    inputs = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=config.bert_max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        base_probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
    base_phish_prob = float(base_probs[1])

    encoded = tokenizer.encode_plus(
        text,
        truncation=True,
        max_length=config.bert_max_length,
        add_special_tokens=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    filtered_tokens = [t for t in tokens if t not in tokenizer.all_special_tokens]

    importance = []
    for idx, tok in enumerate(filtered_tokens):
        occluded_tokens = filtered_tokens[:idx] + filtered_tokens[idx + 1 :]
        occluded_text = tokenizer.convert_tokens_to_string(occluded_tokens)
        occluded_inputs = tokenizer(
            [occluded_text],
            truncation=True,
            padding="max_length",
            max_length=config.bert_max_length,
            return_tensors="pt",
        )
        occluded_inputs = {k: v.to(device) for k, v in occluded_inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**occluded_inputs).logits, dim=1).cpu().numpy()[0]
        drop = base_phish_prob - float(probs[1])
        importance.append((tok, drop))

    importance_sorted = sorted(importance, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return importance_sorted
