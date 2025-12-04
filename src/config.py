"""Project-wide configuration values and paths."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Configuration dataclass holding file paths and hyperparameters."""

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_path: Path = project_root / "data" / "phishing_dataset_1900.csv"
    models_dir: Path = project_root / "models"
    classic_tfidf_svm_path: Path = models_dir / "classic_tfidf_svm.joblib"
    classic_tfidf_rf_path: Path = models_dir / "classic_tfidf_random_forest.joblib"
    classic_best_model_path: Path = models_dir / "classic_best_model.joblib"
    tfidf_vectorizer_path: Path = models_dir / "tfidf_vectorizer.joblib"
    metadata_scaler_path: Path = models_dir / "metadata_scaler.joblib"
    bert_model_dir: Path = models_dir / "bert_model"

    # General
    random_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15

    # TF-IDF
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.9

    # Classic models
    svm_c: float = 1.5
    svm_max_iter: int = 2000
    logistic_reg_c: float = 1.0
    rf_estimators: int = 200
    rf_max_depth: int | None = None

    # BERT
    pretrained_model_name: str = "bert-base-uncased"
    bert_max_length: int = 256
    bert_batch_size: int = 8
    bert_learning_rate: float = 2e-5
    bert_epochs: int = 3
    bert_weight_decay: float = 0.01

    # Explainability
    suspicious_keywords: List[str] = field(
        default_factory=lambda: ["verify", "update", "urgent", "click", "password", "login", "account"]
    )


config = Config()

# Ensure model directory exists at import time
config.models_dir.mkdir(parents=True, exist_ok=True)
