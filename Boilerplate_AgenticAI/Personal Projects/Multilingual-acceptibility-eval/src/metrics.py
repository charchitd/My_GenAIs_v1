from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


@dataclass
class Metrics:
    accuracy: float
    mcc: float
    auroc: float | None


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Metrics:
    """
    Training-free baseline: classify using the median score as a threshold.
    """
    thr = float(np.median(scores))
    y_pred = (scores >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    auroc = None
    try:
        if len(np.unique(y_true)) == 2:
            auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = None

    return Metrics(accuracy=acc, mcc=mcc, auroc=auroc)
