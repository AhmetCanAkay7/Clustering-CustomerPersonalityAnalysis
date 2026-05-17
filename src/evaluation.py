"""Evaluation helpers for clustering models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Evaluate clustering labels with metrics that are valid for the labels.

    DBSCAN noise points labelled as -1 are excluded from metric calculations,
    but the reported number of clusters also excludes the noise label.
    """
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1}) if -1 in unique_labels else len(unique_labels)
    noise_ratio = float(np.mean(labels == -1)) if -1 in unique_labels else 0.0

    empty_metrics = {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }

    if n_clusters < 2:
        return empty_metrics

    mask = labels != -1
    X_eval = X[mask]
    labels_eval = labels[mask]

    if X_eval.shape[0] < 2 or len(set(labels_eval.tolist())) < 2:
        return empty_metrics

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": float(silhouette_score(X_eval, labels_eval)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(X_eval, labels_eval)),
    }


def format_metric(value: Any, digits: int = 4) -> str:
    """Return a report-friendly metric string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)

