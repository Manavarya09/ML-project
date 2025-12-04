"""Explainability utilities using SHAP and LIME."""

from typing import Iterable, List, Tuple, Dict, Any

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


def explain_bert_tokens(text: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Compute token importance for highlighting.
    Returns list of dicts: {'word': str, 'score': float, 'start': int, 'end': int}
    """
    model, tokenizer, device = load_finetuned_model()
    model.eval()

    # Tokenize with offsets
    inputs = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=config.bert_max_length,
        return_tensors="pt",
    )
    
    input_ids = inputs["input_ids"][0]
    offsets = inputs["offset_mapping"][0]
    attention_mask = inputs["attention_mask"][0]
    
    # Baseline probability
    model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}
    with torch.no_grad():
        base_probs = torch.softmax(model(**model_inputs).logits, dim=1).cpu().numpy()[0]
    base_phish_prob = float(base_probs[1])

    # Get special token IDs to skip
    special_ids = set(tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id

    results = []
    
    # Iterate over tokens (skip padding and special tokens)
    # We only check the first N tokens to save time if text is long, or all non-pad tokens
    seq_len = int(attention_mask.sum())
    
    for i in range(seq_len):
        tok_id = int(input_ids[i])
        if tok_id in special_ids:
            continue
            
        # Occlusion via Masking (better for BERT than removal)
        masked_ids = input_ids.clone()
        masked_ids[i] = mask_id
        
        masked_inputs = {
            "input_ids": masked_ids.unsqueeze(0).to(device),
            "attention_mask": attention_mask.unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            probs = torch.softmax(model(**masked_inputs).logits, dim=1).cpu().numpy()[0]
        
        drop = base_phish_prob - float(probs[1])
        
        # Only keep positive contributions (evidence FOR phishing)
        if drop > 0.001:
            start, end = offsets[i].numpy()
            # Handle subwords: if start==end (some special cases) skip
            if start == end: continue
            
            results.append({
                "word": text[start:end],
                "score": float(drop),
                "start": int(start),
                "end": int(end)
            })

    # Sort by importance
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results
