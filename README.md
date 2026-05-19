# CSE4063 Data Mining Project #2 - Clustering Analysis

This repository contains the Clustering Analysis part of CSE4063 Fundamentals of Data Mining Project #2.

The dataset is Customer Personality Analysis from Kaggle:
https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data

The goal is to segment customers using:

- K-Means
- AGNES / Agglomerative Hierarchical Clustering
- DBSCAN

The project includes data understanding, preprocessing, feature selection, PCA dimensionality reduction, model tuning (k >= 3 for richer segmentation), clustering evaluation, visualization, and business interpretation of customer segments.

## Project Structure

- `data/raw/marketing_campaign.csv`: raw tab-separated dataset
- `data/processed/`: cleaned, encoded, and scaled outputs
- `src/`: reusable Python modules and the end-to-end pipeline
- `notebooks/clustering_analysis.ipynb`: reproducible notebook entry point
- `outputs/figures/`: EDA and clustering figures
- `outputs/tables/`: tuning tables, comparison tables, and cluster profiles
- `report_assets/`: report section and demo notes

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full clustering pipeline:

```bash
python src/run_clustering_pipeline.py
```

The script expects `data/raw/marketing_campaign.csv`. If it is missing, it attempts to download the same file from a public research-data mirror of the Kaggle dataset for reproducibility.

## Pipeline Overview

1. Load and understand the raw dataset (2240 rows × 29 columns).
2. Preprocess: handle missing values, remove non-informative columns, engineer features.
3. **Feature selection**: select non-redundant features to avoid multicollinearity in distance calculations.
4. **PCA**: reduce dimensionality while retaining ≥ 85 % of variance (25 scaled features → 15 PCA components).
5. Tune K-Means, AGNES, and DBSCAN on the PCA-reduced data (minimum 3 clusters for K-Means/AGNES).
6. Compare algorithms, profile clusters using original unscaled features, and generate the report.

## Main Outputs

- `outputs/tables/kmeans_tuning_results.csv`
- `outputs/tables/agnes_tuning_results.csv`
- `outputs/tables/dbscan_tuning_results.csv`
- `outputs/tables/clustering_comparison.csv`
- `outputs/tables/cluster_profiles_best_model.csv`
- `report_assets/clustering_report_section.md`
- `report_assets/demo_notes.md`
- `we_swear.txt`

## Notes

This implementation focuses only on the clustering part of the project. It does not implement Apriori, FP-Growth, or ECLAT because those belong to the Frequent Pattern Mining part.
