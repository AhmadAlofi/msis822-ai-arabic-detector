import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)

def _to_1d_array(x):
    """Convert input to a 1D numpy array."""
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1,)
    if x.ndim > 1:
        # common cases: (N,1) or (1,N) -> flatten safely
        x = x.reshape(-1)
    return x


def compute_metrics(y_true, y_pred, scores=None):
    """
    Compute standard binary classification metrics.

    Args:
        y_true: array-like of shape (N,)
        y_pred: array-like of shape (N,) with predicted labels {0,1}
        scores: optional array-like of shape (N,) with probability or decision scores
                (used only for ROC-AUC)

    Returns:
        dict with keys:
        - accuracy
        - balanced_accuracy
        - precision
        - recall
        - f1
        - roc_auc (None if cannot be computed)
    """
    y_true = _to_1d_array(y_true).astype(int)
    y_pred = _to_1d_array(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    auc = None
    if scores is not None:
        scores = _to_1d_array(scores)
        # ROC-AUC fails if scores are constant or y_true has a single class
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = None

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": None if auc is None else float(auc),
    }


def compute_confusion(y_true, y_pred):
    """
    Convenience helper for consistent confusion matrices.
    Returns a 2x2 numpy array in the order:
        [[TN, FP],
         [FN, TP]]
    """
    y_true = _to_1d_array(y_true).astype(int)
    y_pred = _to_1d_array(y_pred).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def pick_best_by_val(results_dict):
    """
    Picks best model based on validation ROC-AUC if present, else F1.

    Args:
        results_dict: dict like:
          {
            "ModelA": {"val": {"roc_auc": ..., "f1": ...}, ...},
            "ModelB": {"val": {...}, ...},
          }

    Returns:
        (best_name, best_score)
    """
    best_name = None
    best_score = -1e18

    for name, res in results_dict.items():
        val = res.get("val", {}) or {}
        if val.get("roc_auc") is not None:
            score = val.get("roc_auc")
        else:
            score = val.get("f1")

        if score is None:
            score = -1e18

        if score > best_score:
            best_score = score
            best_name = name

    return best_name, float(best_score)
