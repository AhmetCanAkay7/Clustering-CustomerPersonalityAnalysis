"""End-to-end clustering analysis pipeline for the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from clustering_models import k_distance_values, tune_agnes, tune_dbscan, tune_kmeans
from data_preprocessing import (
    dataset_understanding,
    ensure_raw_data,
    load_raw_data,
    preprocess_data,
)
from evaluation import evaluate_clustering, format_metric
from feature_engineering import (
    MNT_COLS,
    add_customer_features,
    encode_and_scale,
    select_clustering_features,
)
from visualization import (
    plot_cluster_profile_bars,
    plot_correlation_heatmap,
    plot_dendrogram_sample,
    plot_distribution,
    plot_k_distance,
    plot_metric_line,
    plot_missing_values,
    plot_pca_clusters,
    plot_spending_boxplots,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "marketing_campaign.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report_assets"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


PROFILE_AGG = {
    "Age": ["mean", "median"],
    "Income": ["mean", "median"],
    "Total_Spending": ["mean", "median"],
    "Total_Purchases": ["mean", "median"],
    "Total_Children": ["mean", "median"],
    "Recency": ["mean", "median"],
    "Customer_Tenure_Days": ["mean", "median"],
    "Total_Accepted_Campaigns": ["mean", "median"],
    "MntWines": ["mean", "median"],
    "MntFruits": ["mean", "median"],
    "MntMeatProducts": ["mean", "median"],
    "MntFishProducts": ["mean", "median"],
    "MntSweetProducts": ["mean", "median"],
    "MntGoldProds": ["mean", "median"],
    "NumDealsPurchases": ["mean", "median"],
    "NumWebPurchases": ["mean", "median"],
    "NumCatalogPurchases": ["mean", "median"],
    "NumStorePurchases": ["mean", "median"],
    "NumWebVisitsMonth": ["mean", "median"],
    "Average_Spending_Per_Purchase": ["mean", "median"],
    "Web_Purchase_Ratio": ["mean", "median"],
    "Store_Purchase_Ratio": ["mean", "median"],
    "Catalog_Purchase_Ratio": ["mean", "median"],
    "Deal_Purchase_Ratio": ["mean", "median"],
}


def ensure_directories() -> None:
    for path in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, REPORT_DIR, NOTEBOOK_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Create a simple GitHub-flavored Markdown table without extra dependencies."""
    display_df = df.copy()
    display_df = display_df.fillna("N/A")
    headers = [str(col) for col in display_df.columns]
    rows = [[str(value) for value in row] for row in display_df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def save_dataset_understanding(understanding: dict[str, Any]) -> None:
    pd.DataFrame({"Column": understanding["column_names"]}).to_csv(
        TABLES_DIR / "dataset_columns.csv", index=False
    )
    understanding["dtypes"].rename("dtype").to_csv(TABLES_DIR / "data_types.csv")
    understanding["missing_values"].rename("missing_count").to_csv(
        TABLES_DIR / "missing_values.csv"
    )
    understanding["descriptive_statistics"].to_csv(
        TABLES_DIR / "descriptive_statistics.csv"
    )
    overview = [
        "# Dataset Understanding",
        "",
        "- Dataset: Customer Personality Analysis",
        "- Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data",
        f"- Rows before preprocessing: {understanding['rows']}",
        f"- Columns before preprocessing: {understanding['columns']}",
        f"- Duplicate rows: {understanding['duplicate_rows']}",
        "",
        "## Feature Groups",
        "",
        "- Demographic: ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome, Dt_Customer, Recency, Complain",
        "- Product spending: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds",
        "- Campaign response: AcceptedCmp1-5 and Response",
        "- Purchase channel: NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth",
    ]
    (REPORT_DIR / "dataset_understanding.md").write_text("\n".join(overview), encoding="utf-8")


def make_eda_figures(raw_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
    missing = raw_df.isna().sum()
    plot_missing_values(missing, FIGURES_DIR / "missing_values.png")
    plot_distribution(features_df, "Income", FIGURES_DIR / "income_distribution.png", "Income Distribution")
    plot_distribution(features_df, "Age", FIGURES_DIR / "age_distribution.png", "Age Distribution")
    plot_distribution(
        features_df,
        "Total_Spending",
        FIGURES_DIR / "total_spending_distribution.png",
        "Total Spending Distribution",
    )
    corr_cols = [
        "Age",
        "Income",
        "Total_Spending",
        "Total_Purchases",
        "Total_Children",
        "Recency",
        "Customer_Tenure_Days",
        "Total_Accepted_Campaigns",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
    ]
    plot_correlation_heatmap(features_df, corr_cols, FIGURES_DIR / "correlation_heatmap.png")
    plot_spending_boxplots(features_df, MNT_COLS, FIGURES_DIR / "spending_boxplots.png")


def create_pca_frame(X_scaled: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    variance = pd.DataFrame(
        {
            "Component": ["PC1", "PC2"],
            "Explained Variance Ratio": pca.explained_variance_ratio_,
        }
    )
    return pca_df, variance


def segment_name(row: pd.Series, features_df: pd.DataFrame) -> str:
    """Assign a business-friendly segment name from profile statistics."""
    spending_q75 = features_df["Total_Spending"].quantile(0.75)
    spending_q55 = features_df["Total_Spending"].quantile(0.55)
    spending_q40 = features_df["Total_Spending"].quantile(0.40)
    income_q70 = features_df["Income"].quantile(0.70)
    income_q40 = features_df["Income"].quantile(0.40)
    children_median = features_df["Total_Children"].median()
    deals_q70 = features_df["NumDealsPurchases"].quantile(0.70)
    web_ratio_q70 = features_df["Web_Purchase_Ratio"].quantile(0.70)
    recency_q70 = features_df["Recency"].quantile(0.70)
    catalog_q70 = features_df["NumCatalogPurchases"].quantile(0.70)
    store_q70 = features_df["NumStorePurchases"].quantile(0.70)
    campaign_q75 = features_df["Total_Accepted_Campaigns"].quantile(0.75)
    wine_q60 = features_df["MntWines"].quantile(0.60)

    spending = row.get("Total_Spending_median", 0)
    income = row.get("Income_median", 0)
    children = row.get("Total_Children_mean", 0)
    deals = row.get("NumDealsPurchases_median", 0)
    web_ratio = row.get("Web_Purchase_Ratio_mean", 0)
    recency = row.get("Recency_median", 0)
    catalog = row.get("NumCatalogPurchases_median", 0)
    store = row.get("NumStorePurchases_median", 0)
    campaigns = row.get("Total_Accepted_Campaigns_mean", 0)
    wine = row.get("MntWines_median", 0)

    if spending >= spending_q75 and income >= income_q70:
        return "High Value Affluent Buyers"
    if spending >= spending_q75 and (catalog >= catalog_q70 or store >= store_q70):
        return "Premium Store/Catalog Buyers"
    if campaigns >= campaign_q75 and spending >= spending_q40:
        return "Campaign Responsive Customers"
    if children > children_median and spending <= spending_q40:
        return "Budget-Conscious Family Customers"
    if web_ratio >= web_ratio_q70:
        return "Digitally Active Web Customers"
    if deals >= deals_q70 and income <= income_q40:
        return "Deal-Oriented Value Seekers"
    if recency >= recency_q70 and spending <= spending_q40:
        return "Inactive Low-Engagement Customers"
    # Mid-tier differentiation: separate wine-heavy from balanced spenders
    if spending >= spending_q55 and wine >= wine_q60:
        return "Wine-Enthusiast Mid-Tier Customers"
    if spending >= spending_q40:
        return "Balanced Mid-Tier Customers"
    return "Moderate Regular Customers"


def profile_clusters(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    algorithm_name: str,
) -> pd.DataFrame:
    profile_df = features_df.copy()
    profile_df["Cluster"] = labels
    available_agg = {
        col: aggs for col, aggs in PROFILE_AGG.items() if col in profile_df.columns
    }
    grouped = profile_df.groupby("Cluster")
    profile = grouped.agg(available_agg)
    profile.columns = [f"{col}_{agg}" for col, agg in profile.columns]
    profile.insert(0, "Customer_Count", grouped.size())
    profile.insert(0, "Algorithm", algorithm_name)
    profile["Segment_Name"] = [
        "Noise / Outlier Customers" if idx == -1 else segment_name(row, features_df)
        for idx, row in profile.iterrows()
    ]
    return profile.reset_index()


def best_model_name(comparison: pd.DataFrame) -> str:
    valid = comparison.dropna(subset=["Silhouette Score"]).copy()
    if valid.empty:
        return str(comparison.iloc[0]["Algorithm"])
    acceptable = valid[
        (valid["Noise Ratio"].fillna(0) <= 0.50)
        & (valid["Smallest Cluster Ratio"].fillna(0) >= 0.05)
        & (valid["Smallest Cluster Size"].fillna(0) >= 30)
    ]
    if acceptable.empty:
        acceptable = valid
    best = acceptable.sort_values(
        ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
        ascending=[False, True, False],
    ).iloc[0]
    return str(best["Algorithm"])


def build_comparison(
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    params: dict[str, dict[str, Any]],
    tuning_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for algorithm in ["K-Means", "AGNES", "DBSCAN"]:
        metrics = evaluate_clustering(X, labels[algorithm])
        table = tuning_tables[algorithm]
        if algorithm == "K-Means":
            row = table.loc[table["k"] == params[algorithm]["n_clusters"]].iloc[0]
            runtime = float(row["runtime_seconds"])
            best_parameters = f"k={params[algorithm]['n_clusters']}, n_init=20"
            short = "Centroid-based segments with compact, easy-to-profile customer groups."
        elif algorithm == "AGNES":
            row = table[
                (table["n_clusters_param"] == params[algorithm]["n_clusters"])
                & (table["linkage"] == params[algorithm]["linkage"])
            ].iloc[0]
            runtime = float(row["runtime_seconds"])
            best_parameters = (
                f"n_clusters={params[algorithm]['n_clusters']}, "
                f"linkage={params[algorithm]['linkage']}"
            )
            short = "Hierarchical segments useful for comparing nested customer structure."
        else:
            row = table[
                (table["eps"] == params[algorithm]["eps"])
                & (table["min_samples"] == params[algorithm]["min_samples"])
            ].iloc[0]
            runtime = float(row["runtime_seconds"])
            best_parameters = (
                f"eps={params[algorithm]['eps']}, "
                f"min_samples={params[algorithm]['min_samples']}"
            )
            short = "Density-based grouping that also marks sparse customers as noise."

        rows.append(
            {
                "Algorithm": algorithm,
                "Best Parameters": best_parameters,
                "Number of Clusters": metrics["n_clusters"],
                "Noise Ratio": metrics["noise_ratio"],
                "Smallest Cluster Size": int(row["min_cluster_size"]),
                "Smallest Cluster Ratio": float(row["min_cluster_ratio"]),
                "Silhouette Score": metrics["silhouette"],
                "Davies-Bouldin Index": metrics["davies_bouldin"],
                "Calinski-Harabasz Index": metrics["calinski_harabasz"],
                "Runtime Seconds": runtime,
                "Short Interpretation": short,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    raw_df: pd.DataFrame,
    preprocessing_summary: dict[str, Any],
    comparison: pd.DataFrame,
    best_algorithm: str,
    best_profile: pd.DataFrame,
) -> None:
    report_comparison = comparison.copy()
    for col in [
        "Noise Ratio",
        "Smallest Cluster Ratio",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Runtime Seconds",
    ]:
        report_comparison[col] = report_comparison[col].map(lambda value: format_metric(value, 4))

    profile_preview_cols = [
        "Cluster",
        "Segment_Name",
        "Customer_Count",
        "Age_mean",
        "Income_median",
        "Total_Spending_median",
        "Total_Purchases_median",
        "Total_Children_mean",
        "Total_Accepted_Campaigns_mean",
    ]
    profile_preview = best_profile[
        [col for col in profile_preview_cols if col in best_profile.columns]
    ].copy()
    for col in profile_preview.select_dtypes(include=[float]).columns:
        profile_preview[col] = profile_preview[col].round(2)

    segment_notes = []
    for _, row in profile_preview.iterrows():
        segment_notes.append(
            f"- Cluster {row['Cluster']} - {row['Segment_Name']}: "
            f"median income {format_metric(row.get('Income_median'), 2)}, "
            f"median spending {format_metric(row.get('Total_Spending_median'), 2)}, "
            f"median purchases {format_metric(row.get('Total_Purchases_median'), 2)}."
        )

    content = f"""# Clustering Analysis Report Section

## Problem Definition

The objective of the clustering part is to segment customers based on demographic, spending, purchase-channel, and campaign-response behavior. Since there is no target label, unsupervised clustering methods are used to discover natural customer groups. The discovered segments can help a company design more targeted marketing campaigns and understand customer behavior patterns.

## Dataset

- Dataset name: Customer Personality Analysis
- Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data
- Rows and columns before preprocessing: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns
- Duplicate rows before preprocessing: {int(raw_df.duplicated().sum())}
- Main feature groups: demographic features, product spending, campaign response, and purchase-channel behavior.
- Suitability: the dataset contains customer-level attributes without a target label, making it appropriate for customer segmentation by clustering.

## Data Preprocessing and Cleaning

- Missing values were checked for every column. `Income` had {preprocessing_summary['income_missing_imputed']} missing values and was imputed with the median value {format_metric(preprocessing_summary['income_median_used'], 2)}.
- Non-informative columns removed: {', '.join(preprocessing_summary['removed_columns'])}.
- `Dt_Customer` was parsed as a date and converted into `Customer_Tenure_Days` using reference date {preprocessing_summary['reference_date']}.
- `Year_Birth` was converted into `Age`.
- Age outliers above 100 were removed. Rows removed: {preprocessing_summary['rows_removed_age_over_100']}.
- Engineered features include `Total_Spending`, `Total_Children`, `Total_Purchases`, `Total_Accepted_Campaigns`, spending per purchase, and purchase-channel ratios.
- **Feature selection**: To avoid redundancy in distance calculations, only non-redundant features were selected for clustering input. Individual spending columns (`MntWines`, etc.) were excluded because `Total_Spending` captures the same information; individual purchase counts were excluded in favour of `Total_Purchases` and channel-ratio features; `Kidhome`/`Teenhome` were replaced by `Total_Children`; campaign acceptance columns by `Total_Accepted_Campaigns`; and `Complain` was removed due to near-zero variance.
- `Education` and `Marital_Status` were one-hot encoded with the first category dropped.
- All selected and encoded features were standardized with `StandardScaler`.
- **PCA dimensionality reduction**: PCA was applied to the scaled feature matrix, retaining enough components to explain at least 85 % of the total variance. This reduces the curse-of-dimensionality effect and improves distance-based clustering quality.

## Implementation Details

- K-Means was implemented with scikit-learn `KMeans`. Values of k from 2 to 10 were tested using inertia, silhouette score, Davies-Bouldin index, Calinski-Harabasz index, and runtime. The best k was selected from k >= 3 to ensure richer segmentation.
- AGNES was implemented with scikit-learn `AgglomerativeClustering`. Cluster counts from 2 to 10 and `ward`, `complete`, and `average` linkage methods were tested. Selection also required at least 3 clusters.
- DBSCAN was implemented with scikit-learn `DBSCAN`. A k-distance plot was created and multiple `eps` and `min_samples` values were tested on the PCA-reduced feature matrix.
- PCA with two components was used for visualization of the final cluster assignments.

## Model Evaluation and Performance Results

{dataframe_to_markdown(report_comparison)}

The selected best model for interpretation is **{best_algorithm}**. The selection uses valid clustering metrics, prioritizing higher silhouette score, lower Davies-Bouldin index, higher Calinski-Harabasz index, and avoiding DBSCAN settings with excessive noise when possible.

## Best Model Segment Profiles

{dataframe_to_markdown(profile_preview)}

## Business Interpretation and Recommendations

{chr(10).join(segment_notes)}

Recommended marketing actions:

- High-value or premium customers should receive loyalty rewards, premium bundles, and retention-oriented campaigns.
- Budget-conscious or family-oriented customers should receive discount bundles and practical offers instead of expensive premium campaigns.
- Digitally active customers should be targeted through web campaigns and personalized online recommendations.
- Inactive or low-engagement customers should receive reactivation offers with simple, low-friction messaging.

## Figures and Outputs

Important figures are saved under `outputs/figures/`, including missing values, income/age/spending distributions, correlation heatmap, K-Means elbow and silhouette plots, PCA cluster plots, the DBSCAN k-distance plot, and best-model cluster profile bars.

## Conclusion

The clustering pipeline successfully preprocesses the customer data, selects non-redundant features, applies PCA for dimensionality reduction, and compares K-Means, AGNES, and DBSCAN on the reduced representation. In this run, **{best_algorithm}** provides the primary segment interpretation. K-Means and AGNES are generally easier to interpret for this dataset because they force every customer into a segment, while DBSCAN is useful for detecting sparse or unusual customers but is more sensitive to `eps` and `min_samples`.

Limitations include sensitivity to preprocessing choices and the lack of external labels for ground-truth validation. Possible improvements include trying RobustScaler, UMAP/t-SNE visualization, and validating segments with domain experts.
"""
    (REPORT_DIR / "clustering_report_section.md").write_text(content, encoding="utf-8")


def write_demo_notes(best_algorithm: str, comparison: pd.DataFrame) -> None:
    best_row = comparison[comparison["Algorithm"] == best_algorithm].iloc[0]
    content = f"""# Demo Notes - Clustering Analysis

1. Dataset selection: Customer Personality Analysis is suitable because it has customer demographics, spending, campaign response, and purchase-channel behavior.
2. Customer segmentation means grouping similar customers without using a target label.
3. Scaling is necessary because distance-based algorithms would otherwise be dominated by large-scale variables such as income and spending.
4. `ID`, `Z_CostContact`, and `Z_Revenue` were removed because `ID` is an identifier and the Z columns are constant/non-informative in this dataset.
5. Missing `Income` values were filled with the median because income is skewed and the median is robust to extremes.
6. Engineered features: `Age`, `Customer_Tenure_Days`, `Total_Spending`, `Total_Purchases`, `Total_Children`, `Total_Accepted_Campaigns`, and purchase-channel ratios.
7. K-Means groups customers around centroids. k was selected by testing k=2..10 and comparing inertia plus clustering metrics.
8. AGNES means Agglomerative Nesting. It starts with each point as its own cluster and merges clusters step by step according to linkage distance.
9. DBSCAN uses `eps` as the neighborhood radius and `min_samples` as the density threshold. It can label sparse points as noise.
10. Silhouette is higher when clusters are compact and separated. Davies-Bouldin is better when lower. Calinski-Harabasz is better when higher.
11. Best model in this run: {best_algorithm}. Key metrics: silhouette={format_metric(best_row['Silhouette Score'])}, Davies-Bouldin={format_metric(best_row['Davies-Bouldin Index'])}, Calinski-Harabasz={format_metric(best_row['Calinski-Harabasz Index'])}.
12. Business meaning: the final clusters are interpreted using unscaled original features such as income, spending, purchases, children, recency, campaign acceptance, and channel behavior.
"""
    (REPORT_DIR / "demo_notes.md").write_text(content, encoding="utf-8")


def write_we_swear() -> None:
    text = (
        "We hereby swear that the work done on this project is totally our own; "
        "and on our honor, we have neither given nor received any unauthorized "
        "and/or inappropriate assistance for this project. We understand that by "
        "the school code, violation of these principles will lead to a zero grade "
        "and is subject to harsh discipline issues."
    )
    (PROJECT_ROOT / "we_swear.txt").write_text(text + "\n", encoding="utf-8")


def write_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Customer Personality Analysis - Clustering\n\n"
            "This notebook is a reproducible walkthrough of the clustering pipeline. "
            "The implementation lives in `src/` so the same analysis can be run from "
            "the notebook or from `python src/run_clustering_pipeline.py`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n\n"
            "from run_clustering_pipeline import main\n"
            "print(PROJECT_ROOT)"
        ),
        nbf.v4.new_markdown_cell(
            "## Run the Full Pipeline\n\n"
            "This single call loads the data, preprocesses it, tunes K-Means, AGNES, "
            "and DBSCAN, saves figures/tables, and writes the report assets."
        ),
        nbf.v4.new_code_cell("main()"),
        nbf.v4.new_markdown_cell(
            "## Inspect Final Outputs\n\n"
            "The most important deliverables are written to `outputs/tables`, "
            "`outputs/figures`, and `report_assets`."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n\n"
            "comparison = pd.read_csv(PROJECT_ROOT / 'outputs' / 'tables' / 'clustering_comparison.csv')\n"
            "comparison"
        ),
        nbf.v4.new_code_cell(
            "profiles = pd.read_csv(PROJECT_ROOT / 'outputs' / 'tables' / 'cluster_profiles_best_model.csv')\n"
            "profiles.head()"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NOTEBOOK_DIR / "clustering_analysis.ipynb")


def main() -> None:
    ensure_directories()
    ensure_raw_data(RAW_PATH)

    raw_df = load_raw_data(RAW_PATH)
    understanding = dataset_understanding(raw_df)
    save_dataset_understanding(understanding)

    preprocessing = preprocess_data(raw_df)
    features_df = add_customer_features(preprocessing.data)

    # ---- full feature set (kept for profiling & processed CSVs) ----
    feature_set = encode_and_scale(features_df)
    feature_set.original_features.to_csv(
        PROCESSED_DIR / "customer_personality_features_unscaled.csv", index=False
    )
    feature_set.encoded_features.to_csv(
        PROCESSED_DIR / "customer_personality_encoded.csv", index=False
    )
    feature_set.scaled_features.to_csv(
        PROCESSED_DIR / "customer_personality_processed.csv", index=False
    )

    make_eda_figures(raw_df, features_df)

    # ---- feature selection: keep only non-redundant columns ----
    clustering_df = select_clustering_features(features_df)
    clustering_set = encode_and_scale(clustering_df)
    X_selected_scaled = clustering_set.scaled_features.to_numpy()

    # ---- PCA dimensionality reduction (retain >= 85% variance) ----
    pca_full = PCA(random_state=42)
    pca_full.fit(X_selected_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= 0.85)) + 1
    n_components = max(n_components, 2)

    pca_clustering = PCA(n_components=n_components, random_state=42)
    X_pca = pca_clustering.fit_transform(X_selected_scaled)

    pca_variance = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Explained Variance Ratio": pca_clustering.explained_variance_ratio_,
            "Cumulative Variance": np.cumsum(pca_clustering.explained_variance_ratio_),
        }
    )
    pca_variance.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)

    # 2-D frame for scatter-plot visualisation (first 2 PCA components)
    pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])

    # ---- K-Means (min_clusters=3) ----
    kmeans_results, _, labels_kmeans, params_kmeans = tune_kmeans(
        X_pca, min_clusters=3
    )
    kmeans_results.to_csv(TABLES_DIR / "kmeans_tuning_results.csv", index=False)
    plot_metric_line(
        kmeans_results,
        "k",
        "inertia",
        FIGURES_DIR / "kmeans_elbow.png",
        "K-Means Elbow Plot",
    )
    plot_metric_line(
        kmeans_results,
        "k",
        "silhouette",
        FIGURES_DIR / "kmeans_silhouette.png",
        "K-Means Silhouette by k",
    )
    plot_pca_clusters(
        pca_df,
        labels_kmeans,
        FIGURES_DIR / "kmeans_pca_clusters.png",
        "K-Means Clusters on PCA Projection",
    )

    # ---- AGNES (min_clusters=3) ----
    agnes_results, _, labels_agnes, params_agnes = tune_agnes(
        X_pca, min_clusters=3
    )
    agnes_results.to_csv(TABLES_DIR / "agnes_tuning_results.csv", index=False)
    plot_pca_clusters(
        pca_df,
        labels_agnes,
        FIGURES_DIR / "agnes_pca_clusters.png",
        "AGNES Clusters on PCA Projection",
    )
    plot_dendrogram_sample(X_pca, FIGURES_DIR / "agnes_dendrogram_sample.png")

    # ---- DBSCAN ----
    k_distances = k_distance_values(X_pca, min_samples=10)
    plot_k_distance(
        k_distances,
        FIGURES_DIR / "dbscan_k_distance.png",
        "DBSCAN k-Distance Plot (min_samples=10)",
    )
    dbscan_results, _, labels_dbscan, params_dbscan = tune_dbscan(X_pca)
    dbscan_results.to_csv(TABLES_DIR / "dbscan_tuning_results.csv", index=False)
    dbscan_noise = dbscan_results[
        ["eps", "min_samples", "n_clusters", "noise_ratio", "silhouette"]
    ].copy()
    dbscan_noise.to_csv(TABLES_DIR / "dbscan_noise_ratio_table.csv", index=False)
    plot_pca_clusters(
        pca_df,
        labels_dbscan,
        FIGURES_DIR / "dbscan_pca_clusters.png",
        "DBSCAN Clusters on PCA Projection",
    )

    # ---- comparison & profiling ----
    labels_by_algorithm = {
        "K-Means": labels_kmeans,
        "AGNES": labels_agnes,
        "DBSCAN": labels_dbscan,
    }
    params_by_algorithm = {
        "K-Means": params_kmeans,
        "AGNES": params_agnes,
        "DBSCAN": params_dbscan,
    }
    tuning_tables = {
        "K-Means": kmeans_results,
        "AGNES": agnes_results,
        "DBSCAN": dbscan_results,
    }

    comparison = build_comparison(
        X_pca, labels_by_algorithm, params_by_algorithm, tuning_tables
    )
    comparison.to_csv(TABLES_DIR / "clustering_comparison.csv", index=False)
    comparison_md = comparison.copy()
    for col in [
        "Noise Ratio",
        "Smallest Cluster Ratio",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Runtime Seconds",
    ]:
        comparison_md[col] = comparison_md[col].map(lambda value: format_metric(value, 4))
    (TABLES_DIR / "clustering_comparison.md").write_text(
        dataframe_to_markdown(comparison_md), encoding="utf-8"
    )

    profiles = {
        name: profile_clusters(features_df, labels, name)
        for name, labels in labels_by_algorithm.items()
    }
    for name, profile in profiles.items():
        safe_name = name.lower().replace("-", "").replace(" ", "_")
        profile.to_csv(TABLES_DIR / f"cluster_profiles_{safe_name}.csv", index=False)

    selected_best = best_model_name(comparison)
    best_profile = profiles[selected_best]
    best_profile.to_csv(TABLES_DIR / "cluster_profiles_best_model.csv", index=False)
    plot_cluster_profile_bars(
        best_profile.set_index("Cluster"),
        FIGURES_DIR / "best_model_cluster_profiles.png",
        [
            "Income_median",
            "Total_Spending_median",
            "Total_Purchases_median",
            "Total_Children_mean",
            "Total_Accepted_Campaigns_mean",
        ],
        f"{selected_best} Cluster Profiles",
    )

    run_summary = {
        "raw_rows": int(raw_df.shape[0]),
        "raw_columns": int(raw_df.shape[1]),
        "selected_features_scaled_columns": int(X_selected_scaled.shape[1]),
        "pca_components": n_components,
        "pca_explained_variance_total": float(cumvar[n_components - 1]),
        "processed_rows": int(X_pca.shape[0]),
        "processed_columns": int(X_pca.shape[1]),
        "best_algorithm": selected_best,
        "best_parameters": params_by_algorithm[selected_best],
        "preprocessing": preprocessing.summary,
    }
    (TABLES_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8"
    )

    write_report(raw_df, preprocessing.summary, comparison, selected_best, best_profile)
    write_demo_notes(selected_best, comparison)
    write_we_swear()
    write_notebook()

    print("Clustering pipeline completed.")
    print(f"Best algorithm: {selected_best}")
    print(f"PCA components: {n_components} (variance explained: {cumvar[n_components - 1]:.2%})")
    print(f"Outputs written under: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
