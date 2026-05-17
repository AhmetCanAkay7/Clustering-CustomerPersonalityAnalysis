"""Plotting helpers for EDA and cluster interpretation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


FIG_DPI = 160


def _save_current(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_missing_values(missing_values: pd.Series, path: str | Path) -> None:
    values = missing_values[missing_values > 0].sort_values(ascending=False)
    if values.empty:
        values = missing_values.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(x=values.values, y=values.index, color="#4C78A8")
    plt.title("Missing Values by Column")
    plt.xlabel("Missing value count")
    plt.ylabel("Column")
    _save_current(path)


def plot_distribution(df: pd.DataFrame, column: str, path: str | Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df[column], kde=True, bins=35, color="#4C78A8")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Customer count")
    _save_current(path)


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], path: str | Path) -> None:
    selected = [col for col in columns if col in df.columns]
    corr = df[selected].corr(numeric_only=True)
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.3)
    plt.title("Correlation Heatmap for Important Numerical Features")
    _save_current(path)


def plot_spending_boxplots(df: pd.DataFrame, spending_cols: list[str], path: str | Path) -> None:
    spending = df[[col for col in spending_cols if col in df.columns]].melt(
        var_name="Product Category", value_name="Spending"
    )
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=spending, x="Product Category", y="Spending", color="#72B7B2")
    plt.xticks(rotation=35, ha="right")
    plt.title("Distribution of Product Spending Categories")
    _save_current(path)


def plot_metric_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    path: str | Path,
    title: str,
    hue: str | None = None,
) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o")
    plt.title(title)
    _save_current(path)


def plot_pca_clusters(
    pca_df: pd.DataFrame,
    labels: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    plot_df = pca_df.copy()
    plot_df["Cluster"] = labels.astype(str)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        s=32,
        linewidth=0,
        alpha=0.85,
    )
    plt.title(title)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_current(path)


def plot_k_distance(distances: np.ndarray, path: str | Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(np.sort(distances), color="#4C78A8")
    plt.title(title)
    plt.xlabel("Points sorted by distance")
    plt.ylabel("k-nearest neighbor distance")
    _save_current(path)


def plot_dendrogram_sample(
    X: np.ndarray,
    path: str | Path,
    sample_size: int = 300,
    random_state: int = 42,
) -> None:
    rng = np.random.default_rng(random_state)
    n_rows = X.shape[0]
    if n_rows > sample_size:
        sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
    linked = linkage(X_sample, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(linked, truncate_mode="level", p=5, no_labels=True)
    plt.title("AGNES Dendrogram Sample (Ward Linkage)")
    plt.xlabel("Sampled customers")
    plt.ylabel("Distance")
    _save_current(path)


def plot_cluster_profile_bars(
    profile: pd.DataFrame,
    path: str | Path,
    value_columns: list[str],
    title: str,
) -> None:
    available = [col for col in value_columns if col in profile.columns]
    plot_df = profile[available].copy()
    plot_df.index = profile.index.astype(str)
    normalized = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min()).replace(0, 1)
    normalized = normalized.reset_index().melt(
        id_vars=profile.index.name or "Cluster",
        var_name="Feature",
        value_name="Normalized value",
    )
    cluster_col = normalized.columns[0]
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=normalized, x=cluster_col, y="Normalized value", hue="Feature")
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Profile value (min-max normalized)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_current(path)

