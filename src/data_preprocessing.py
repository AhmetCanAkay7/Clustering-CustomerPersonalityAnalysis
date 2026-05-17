"""Data loading, understanding, and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


RAW_DATA_URL = (
    "https://test.researchdata.tuwien.ac.at/records/32sms-w5z07/files/"
    "marketing_campaign.csv?download=1"
)


@dataclass
class PreprocessingResult:
    """Cleaned data and a summary of preprocessing decisions."""

    data: pd.DataFrame
    summary: dict[str, object]


def ensure_raw_data(raw_path: str | Path, download_if_missing: bool = True) -> Path:
    """Ensure the raw Kaggle CSV exists locally.

    The preferred workflow is to place the Kaggle file at data/raw manually.
    For reproducibility in this repository, the function can also fetch the
    same file from a public research-data mirror when it is missing.
    """
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        return raw_path
    if not download_if_missing:
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    urlretrieve(RAW_DATA_URL, raw_path)
    return raw_path


def load_raw_data(raw_path: str | Path) -> pd.DataFrame:
    """Load the tab-separated Customer Personality Analysis dataset."""
    return pd.read_csv(raw_path, sep="\t")


def dataset_understanding(df: pd.DataFrame) -> dict[str, object]:
    """Return the required dataset-understanding outputs."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str),
        "missing_values": df.isna().sum(),
        "duplicate_rows": int(df.duplicated().sum()),
        "descriptive_statistics": df.describe(include="all").transpose(),
    }


def preprocess_data(df: pd.DataFrame) -> PreprocessingResult:
    """Clean raw data and transform dates/year fields into numeric features."""
    cleaned = df.copy()
    initial_rows = len(cleaned)
    missing_before = cleaned.isna().sum()
    duplicate_rows = int(cleaned.duplicated().sum())

    income_missing = int(cleaned["Income"].isna().sum()) if "Income" in cleaned else 0
    income_median = float(cleaned["Income"].median()) if "Income" in cleaned else None
    if "Income" in cleaned:
        cleaned["Income"] = cleaned["Income"].fillna(cleaned["Income"].median())

    columns_to_drop = ["ID", "Z_CostContact", "Z_Revenue"]
    removed_columns = [col for col in columns_to_drop if col in cleaned.columns]
    cleaned = cleaned.drop(columns=removed_columns)

    if "Dt_Customer" in cleaned.columns:
        cleaned["Dt_Customer"] = pd.to_datetime(
            cleaned["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
        )
        reference_date = cleaned["Dt_Customer"].max() + pd.Timedelta(days=1)
        cleaned["Customer_Tenure_Days"] = (
            reference_date - cleaned["Dt_Customer"]
        ).dt.days
        cleaned = cleaned.drop(columns=["Dt_Customer"])
    else:
        reference_date = pd.Timestamp.today().normalize()

    if "Year_Birth" in cleaned.columns:
        cleaned["Age"] = reference_date.year - cleaned["Year_Birth"]
        cleaned = cleaned.drop(columns=["Year_Birth"])

    rows_before_age_filter = len(cleaned)
    if "Age" in cleaned.columns:
        cleaned = cleaned[cleaned["Age"] <= 100].copy()
    rows_removed_age = rows_before_age_filter - len(cleaned)

    summary = {
        "initial_rows": initial_rows,
        "initial_columns": int(df.shape[1]),
        "missing_before": missing_before.astype(int).to_dict(),
        "duplicate_rows": duplicate_rows,
        "income_missing_imputed": income_missing,
        "income_median_used": income_median,
        "removed_columns": removed_columns,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "rows_removed_age_over_100": rows_removed_age,
        "final_rows_after_cleaning": int(len(cleaned)),
        "final_columns_after_cleaning": int(cleaned.shape[1]),
    }

    return PreprocessingResult(data=cleaned.reset_index(drop=True), summary=summary)
