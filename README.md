# CSE4063 Data Mining Project - Clustering Analysis

This repository contains the Clustering Analysis part of CSE4063 Fundamentals of Data Mining Project #2.

The dataset is Customer Personality Analysis from Kaggle:
https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data

The goal is to segment customers using:

- K-Means
- AGNES / Agglomerative Hierarchical Clustering
- DBSCAN

The project includes data understanding, preprocessing, feature engineering, model tuning, clustering evaluation, visualization, and business interpretation of customer segments.

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

## Main Outputs

- `outputs/tables/kmeans_tuning_results.csv`
- `outputs/tables/agnes_tuning_results.csv`
- `outputs/tables/dbscan_tuning_results.csv`
- `outputs/tables/clustering_comparison.csv`
- `outputs/tables/cluster_profiles_best_model.csv`
- `report_assets/clustering_report_section.md`
- `report_assets/demo_notes.md`
- `we_swear.txt`


