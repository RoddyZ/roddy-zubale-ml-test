# TODO: Compute ROC-AUC, PR-AUC, accuracy
# src/metrics.py
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import numpy as np

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute ROC-AUC, PR-AUC, and accuracy."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
