"""Feature engineering and scaling for the customer personality dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler


MNT_COLS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

PURCHASE_COLS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

CAMPAIGN_COLS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "Response",
]

CATEGORICAL_COLS = ["Education", "Marital_Status"]

# Non-redundant numeric features selected for clustering.
# Individual spending columns (MntWines, ...) are excluded because
# Total_Spending already captures total spend.  Individual purchase-count
# columns are excluded because Total_Purchases and channel-ratio features
# capture the same information.  Kidhome/Teenhome are captured by
# Total_Children.  Campaign columns are captured by Total_Accepted_Campaigns.
# Complain has near-zero variance.
CLUSTERING_NUMERIC_COLS = [
    "Age",
    "Income",
    "Recency",
    "Customer_Tenure_Days",
    "Total_Spending",
    "Total_Purchases",
    "Total_Children",
    "Total_Accepted_Campaigns",
    "Average_Spending_Per_Purchase",
    "Web_Purchase_Ratio",
    "Store_Purchase_Ratio",
    "Catalog_Purchase_Ratio",
    "Deal_Purchase_Ratio",
    "NumWebVisitsMonth",
]


def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select non-redundant features for clustering input."""
    numeric = [c for c in CLUSTERING_NUMERIC_COLS if c in df.columns]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    return df[numeric + categorical].copy()


@dataclass
class FeatureSet:
    """Container for engineered and scaled feature data."""

    original_features: pd.DataFrame
    encoded_features: pd.DataFrame
    scaled_features: pd.DataFrame
    scaler: StandardScaler


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level features used for segmentation."""
    engineered = df.copy()

    engineered["Total_Spending"] = engineered[MNT_COLS].sum(axis=1)
    engineered["Total_Children"] = engineered["Kidhome"] + engineered["Teenhome"]
    engineered["Total_Purchases"] = engineered[PURCHASE_COLS].sum(axis=1)
    engineered["Total_Accepted_Campaigns"] = engineered[CAMPAIGN_COLS].sum(axis=1)

    purchases_no_zero = engineered["Total_Purchases"].replace(0, 1)
    engineered["Average_Spending_Per_Purchase"] = (
        engineered["Total_Spending"] / purchases_no_zero
    )
    engineered["Web_Purchase_Ratio"] = engineered["NumWebPurchases"] / purchases_no_zero
    engineered["Store_Purchase_Ratio"] = (
        engineered["NumStorePurchases"] / purchases_no_zero
    )
    engineered["Catalog_Purchase_Ratio"] = (
        engineered["NumCatalogPurchases"] / purchases_no_zero
    )
    engineered["Deal_Purchase_Ratio"] = (
        engineered["NumDealsPurchases"] / purchases_no_zero
    )

    return engineered


def encode_and_scale(df: pd.DataFrame) -> FeatureSet:
    """One-hot encode categorical columns and standardize all features."""
    encoded = pd.get_dummies(
        df,
        columns=[col for col in CATEGORICAL_COLS if col in df.columns],
        drop_first=True,
    )
    encoded = encoded.astype(float)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(encoded)
    scaled = pd.DataFrame(scaled_array, columns=encoded.columns, index=encoded.index)

    return FeatureSet(
        original_features=df.copy(),
        encoded_features=encoded,
        scaled_features=scaled,
        scaler=scaler,
    )

