"""Model evaluation helpers for phishing URL classifiers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_class_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return phishing-class scores from a fitted classifier."""

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            classes = getattr(model, "classes_", None)
            if classes is not None and 1 in list(classes):
                positive_index = list(classes).index(1)
            else:
                positive_index = 1
            return probabilities[:, positive_index]
        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(features)
        raw_scores = np.asarray(raw_scores, dtype=float)
        return 1.0 / (1.0 + np.exp(-raw_scores))

    predictions = model.predict(features)
    return np.asarray(predictions, dtype=float)


def evaluate_classifier(model: Any, features: pd.DataFrame, labels: Iterable[int]) -> Dict[str, Any]:
    """Calculate standard binary classification metrics."""

    y_true = np.asarray(list(labels), dtype=int)
    y_pred = np.asarray(model.predict(features), dtype=int)
    y_score = positive_class_scores(model, features)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 6)
    else:
        metrics["roc_auc"] = None

    return metrics


def extract_feature_importance(
    fitted_model: Any,
    feature_names: List[str],
    top_n: int = 20,
) -> List[Dict[str, float]]:
    """Return feature importance from tree models or linear coefficients."""

    estimator = fitted_model
    if hasattr(fitted_model, "named_steps"):
        estimator = fitted_model.named_steps.get("classifier", fitted_model)

    values: Optional[np.ndarray] = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef.ravel())

    if values is None or len(values) != len(feature_names):
        return []

    ranking = []
    for name, value in zip(feature_names, values):
        ranking.append({"feature": name, "importance": round(float(value), 8)})
    ranking.sort(key=lambda item: item["importance"], reverse=True)
    return ranking[:top_n]
