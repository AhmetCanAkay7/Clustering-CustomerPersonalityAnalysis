"""Model tuning routines for K-Means, AGNES, and DBSCAN."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from evaluation import evaluate_clustering


def _cluster_size_metrics(labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    non_noise = labels[labels != -1]
    if non_noise.size == 0:
        return {
            "min_cluster_size": 0,
            "max_cluster_size": 0,
            "min_cluster_ratio": 0.0,
        }

    sizes = pd.Series(non_noise).value_counts()
    return {
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "min_cluster_ratio": float(sizes.min() / non_noise.size),
    }


def _best_by_metrics(results: pd.DataFrame, min_cluster_ratio: float = 0.05) -> pd.Series:
    valid = results.dropna(subset=["silhouette"]).copy()
    if valid.empty:
        return results.iloc[0]

    interpretable = valid[
        (valid["min_cluster_ratio"] >= min_cluster_ratio)
        & (valid["min_cluster_size"] >= 30)
    ]
    if not interpretable.empty:
        valid = interpretable

    valid = valid.sort_values(
        by=["silhouette", "davies_bouldin", "calinski_harabasz"],
        ascending=[False, True, False],
    )
    return valid.iloc[0]


def tune_kmeans(
    X: np.ndarray,
    k_values: range = range(2, 11),
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, np.ndarray, dict[str, Any]]:
    """Tune K-Means by comparing standard clustering metrics."""
    rows: list[dict[str, Any]] = []
    fitted: dict[int, tuple[KMeans, np.ndarray]] = {}

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
            max_iter=300,
        )
        start = perf_counter()
        labels = model.fit_predict(X)
        runtime = perf_counter() - start
        metrics = evaluate_clustering(X, labels)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "runtime_seconds": runtime,
                **_cluster_size_metrics(labels),
                **metrics,
            }
        )
        fitted[k] = (model, labels)

    results = pd.DataFrame(rows)
    best = _best_by_metrics(results)
    best_k = int(best["k"])
    best_model, best_labels = fitted[best_k]
    best_params = {"n_clusters": best_k, "n_init": 20, "max_iter": 300}
    return results, best_model, best_labels, best_params


def tune_agnes(
    X: np.ndarray,
    n_clusters_values: range = range(2, 11),
    linkage_methods: tuple[str, ...] = ("ward", "complete", "average"),
) -> tuple[pd.DataFrame, AgglomerativeClustering, np.ndarray, dict[str, Any]]:
    """Tune AGNES/Agglomerative Clustering over cluster counts and linkages."""
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[int, str], tuple[AgglomerativeClustering, np.ndarray]] = {}

    for n_clusters in n_clusters_values:
        for linkage in linkage_methods:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
            )
            start = perf_counter()
            labels = model.fit_predict(X)
            runtime = perf_counter() - start
            metrics = evaluate_clustering(X, labels)
            rows.append(
                {
                    "n_clusters_param": n_clusters,
                    "linkage": linkage,
                    "runtime_seconds": runtime,
                    **_cluster_size_metrics(labels),
                    **metrics,
                }
            )
            fitted[(n_clusters, linkage)] = (model, labels)

    results = pd.DataFrame(rows)
    best = _best_by_metrics(results)
    key = (int(best["n_clusters_param"]), str(best["linkage"]))
    best_model, best_labels = fitted[key]
    best_params = {"n_clusters": key[0], "linkage": key[1]}
    return results, best_model, best_labels, best_params


def k_distance_values(X: np.ndarray, min_samples: int = 10) -> np.ndarray:
    """Return sorted k-nearest-neighbor distances for DBSCAN eps inspection."""
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    distances, _ = neighbors.fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def tune_dbscan(
    X_scaled: np.ndarray,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DBSCAN, np.ndarray, dict[str, Any], np.ndarray]:
    """Tune DBSCAN on both full scaled data and five-component PCA data."""
    pca = PCA(n_components=5, random_state=random_state)
    X_pca5 = pca.fit_transform(X_scaled)

    spaces = {
        "scaled_full": X_scaled,
        "pca_5": X_pca5,
    }
    eps_grid = {
        "scaled_full": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0],
        "pca_5": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5],
    }
    min_samples_values = [5, 10, 15]

    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, float, int], tuple[DBSCAN, np.ndarray, np.ndarray]] = {}

    for space_name, X_space in spaces.items():
        for eps in eps_grid[space_name]:
            for min_samples in min_samples_values:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                start = perf_counter()
                labels = model.fit_predict(X_space)
                runtime = perf_counter() - start
                metrics = evaluate_clustering(X_space, labels)
                rows.append(
                    {
                        "space": space_name,
                        "eps": eps,
                        "min_samples": min_samples,
                        "runtime_seconds": runtime,
                        **_cluster_size_metrics(labels),
                        **metrics,
                    }
                )
                fitted[(space_name, eps, min_samples)] = (model, labels, X_space)

    results = pd.DataFrame(rows)
    valid = results.dropna(subset=["silhouette"]).copy()
    if not valid.empty:
        acceptable = valid[
            (valid["noise_ratio"] <= 0.50)
            & (valid["min_cluster_ratio"] >= 0.02)
            & (valid["min_cluster_size"] >= 30)
        ]
        if acceptable.empty:
            acceptable = valid[
                (valid["noise_ratio"] <= 0.80)
                & (valid["min_cluster_size"] >= 10)
            ]
        if acceptable.empty:
            acceptable = valid
        best = acceptable.sort_values(
            by=["silhouette", "noise_ratio", "davies_bouldin"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        best = results.sort_values(["n_clusters", "noise_ratio"], ascending=[False, True]).iloc[0]

    key = (str(best["space"]), float(best["eps"]), int(best["min_samples"]))
    best_model, best_labels, best_X = fitted[key]
    best_params = {
        "space": key[0],
        "eps": key[1],
        "min_samples": key[2],
        "pca_5_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    return results, best_model, best_labels, best_params, best_X
