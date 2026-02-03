"""
Metrics helpers for binary classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    f1: float
    precision: float
    recall: float
    report: Dict[str, Any]
    confusion_matrix: List[List[int]]


def evaluate_binary(y_true: List[int], y_pred: List[int]) -> EvalResult:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return EvalResult(
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        report=rep,
        confusion_matrix=cm,
    )


def save_confusion_matrix_png(cm: List[List[int]], out_path: str) -> None:
    arr = np.array(cm, dtype=int)
    plt.figure()
    plt.imshow(arr)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["neg", "pos"])
    plt.yticks([0, 1], ["neg", "pos"])
    for (i, j), v in np.ndenumerate(arr):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
