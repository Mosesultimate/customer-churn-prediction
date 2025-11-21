"""
Feature engineering module for the Telco customer churn project.

This script consumes the cleaned dataset produced by ``data_preparation.py``,
creates domain-specific features, encodes categorical variables, scales
numerical values, and persists the transformed feature matrix together with
the trained preprocessing pipeline for downstream modeling tasks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CLEAN_DATA_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\cleaned_customer_churn.csv"
DEFAULT_FEATURE_MATRIX_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\features\feature_matrix.parquet"
DEFAULT_TARGET_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\features\target.csv"
DEFAULT_PIPELINE_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\feature_pipeline.joblib"


class FeatureEngineeringError(Exception):
    """Custom exception raised for feature engineering issues."""


@dataclass
class FeatureEngineeringConfig:
    """
    Configuration dataclass for the feature engineering pipeline.

    Attributes
    ----------
    clean_data_path: str
        Location of the cleaned CSV produced by ``data_preparation``.
    feature_matrix_path: str
        Output path for the transformed feature matrix.
    target_path: str
        Output path for the encoded target vector.
    pipeline_path: str
        Location to persist the fitted preprocessing pipeline.
    target_column: str
        Column to predict (defaults to ``Churn``).
    drop_columns: List[str]
        Columns removed before modeling (IDs, leakage sources, etc.).
    service_columns: List[str]
        Service-related categorical columns used to derive custom features.
    binary_mappings: Dict[str, Dict[str, int]]
        Mapping tables used to convert Yes/No style columns to numeric flags.
    ordinal_mappings: Dict[str, Dict[str, int]]
        Ordinal encoding dictionaries for columns with intrinsic order.
    tenure_bins: Tuple[List[int], List[str]]
        Bin edges and labels used to bucket tenure into interpretable groups.
    """

    clean_data_path: str = DEFAULT_CLEAN_DATA_PATH
    feature_matrix_path: str = DEFAULT_FEATURE_MATRIX_PATH
    target_path: str = DEFAULT_TARGET_PATH
    pipeline_path: str = DEFAULT_PIPELINE_PATH
    target_column: str = "Churn"
    drop_columns: List[str] = field(default_factory=lambda: ["customerID"])
    service_columns: List[str] = field(
        default_factory=lambda: [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
    )
    binary_mappings: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "gender": {"Female": 0, "Male": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "MultipleLines": {"No": 0, "No phone service": 0, "Yes": 1},
            "OnlineSecurity": {"No": 0, "No internet service": 0, "Yes": 1},
            "OnlineBackup": {"No": 0, "No internet service": 0, "Yes": 1},
            "DeviceProtection": {"No": 0, "No internet service": 0, "Yes": 1},
            "TechSupport": {"No": 0, "No internet service": 0, "Yes": 1},
            "StreamingTV": {"No": 0, "No internet service": 0, "Yes": 1},
            "StreamingMovies": {"No": 0, "No internet service": 0, "Yes": 1},
            "PaperlessBilling": {"No": 0, "Yes": 1},
        }
    )
    ordinal_mappings: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        }
    )
    tenure_bins: Tuple[List[float], List[str]] = field(
        default_factory=lambda: (
            [0, 6, 12, 24, 48, 72, np.inf],
            ["0-5", "6-11", "12-23", "24-47", "48-71", "72+"],
        )
    )


def load_clean_data(file_path: str) -> pd.DataFrame:
    """Load the cleaned dataset from disk."""
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FeatureEngineeringError(f"Cleaned data not found at {file_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise FeatureEngineeringError("Cleaned dataset is empty. Aborting feature engineering.")
    return df


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a path if it does not exist."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """Encapsulates the feature engineering logic."""

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        self.config = config or FeatureEngineeringConfig()
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[np.ndarray] = None
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessing pipeline and transform the dataset.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Transformed feature matrix and encoded target vector.
        """
        processed_df = self._create_domain_features(df.copy())
        X, y = self._separate_target(processed_df)
        self.numeric_features_, self.categorical_features_ = self._infer_feature_types(X)
        self.pipeline = self._build_pipeline()
        X_array = self.pipeline.fit_transform(X)
        self.feature_names_ = self._get_feature_names()
        X_transformed = pd.DataFrame(X_array, columns=self.feature_names_, index=X.index)
        return X_transformed, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply the fitted preprocessing pipeline to new data.

        Raises
        ------
        FeatureEngineeringError
            If ``fit_transform`` has not been called yet.
        """
        if self.pipeline is None:
            raise FeatureEngineeringError("Pipeline has not been fitted. Call `fit_transform` first.")

        processed_df = self._create_domain_features(df.copy())
        X, y = self._separate_target(processed_df)
        X_array = self.pipeline.transform(X)
        feature_names = self.feature_names_ if self.feature_names_ is not None else np.arange(X_array.shape[1])
        X_transformed = pd.DataFrame(X_array, columns=feature_names, index=X.index)
        return X_transformed, y

    def save_pipeline(self, path: str) -> None:
        """Persist the fitted preprocessing pipeline."""
        if self.pipeline is None:
            raise FeatureEngineeringError("Cannot save pipeline before fitting.")
        ensure_parent_dir(path)
        dump(self.pipeline, path)
        print(f"✓ Feature pipeline saved to {path}")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create custom/domain-specific engineered features."""
        config = self.config

        # Convert binary-style columns using the provided mappings
        for col, mapping in config.binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        for col, mapping in config.ordinal_mappings.items():
            if col in df.columns:
                df[f"{col}_ordinal"] = df[col].map(mapping)

        # Service intensity features
        services_present = []
        service_columns = [col for col in config.service_columns if col in df.columns]
        positive_values = {"Yes", "DSL", "Fiber optic"}
        if service_columns:
            services_present = (
                df[service_columns]
                .apply(lambda row: sum(val in positive_values for val in row), axis=1)
                .rename("services_count")
            )
            df["services_count"] = services_present
            df["has_multiple_services"] = (df["services_count"] >= 3).astype(int)

        if {"MonthlyCharges", "services_count"}.issubset(df.columns):
            df["avg_monthly_charge_per_service"] = df["MonthlyCharges"] / df["services_count"].replace(0, np.nan)
            df["avg_monthly_charge_per_service"] = df["avg_monthly_charge_per_service"].fillna(df["MonthlyCharges"])

        if {"TotalCharges", "tenure"}.issubset(df.columns):
            safe_tenure = df["tenure"].replace(0, np.nan)
            df["total_charges_per_month"] = df["TotalCharges"] / safe_tenure
            df["total_charges_per_month"] = df["total_charges_per_month"].fillna(df["MonthlyCharges"])

        if "tenure" in df.columns:
            bins, labels = config.tenure_bins
            df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False).astype(str)
            df["tenure_years"] = df["tenure"] / 12
            df["long_tenure_flag"] = (df["tenure"] >= 24).astype(int)

        if {"MonthlyCharges", "TotalCharges"}.issubset(df.columns):
            df["charges_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, np.nan)
            df["charges_ratio"] = df["charges_ratio"].replace([np.inf, -np.inf], np.nan)
            df["charges_ratio"] = df["charges_ratio"].fillna(0)

        if "PaymentMethod" in df.columns:
            df["is_electronic_payment"] = (
                df["PaymentMethod"].str.contains("electronic", case=False, na=False).astype(int)
            )

        if "InternetService" in df.columns:
            df["has_fiber"] = df["InternetService"].eq("Fiber optic").astype(int)

        boolean_cols = df.select_dtypes(include=["bool"]).columns
        if len(boolean_cols) > 0:
            df[boolean_cols] = df[boolean_cols].astype(int)

        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _separate_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into feature set (X) and encoded target (y)."""
        target_col = self.config.target_column
        if target_col not in df.columns:
            raise FeatureEngineeringError(f"Target column '{target_col}' missing from dataset.")

        target = df[target_col]
        if target.dtype == "O":
            target = target.map({"No": 0, "Yes": 1})

        if target.isnull().any():
            raise FeatureEngineeringError(f"Target column '{target_col}' contains null or unmapped values.")

        X = df.drop(columns=[target_col])
        for col in self.config.drop_columns:
            if col in X.columns:
                X = X.drop(columns=col)

        return X, target.astype(int)

    def _infer_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Determine numeric and categorical feature lists."""
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Ensure engineered binary columns are treated as numeric
        for col in X.columns.difference(numeric_cols + categorical_cols):
            numeric_cols.append(col)

        if not numeric_cols and not categorical_cols:
            raise FeatureEngineeringError("No features available after preprocessing.")

        return numeric_cols, categorical_cols

    def _build_pipeline(self) -> ColumnTransformer:
        """Construct the sklearn preprocessing pipeline."""
        transformers = []

        if self.numeric_features_:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_transformer, self.numeric_features_))

        if self.categorical_features_:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", categorical_transformer, self.categorical_features_))

        if not transformers:
            raise FeatureEngineeringError("No transformers configured; check feature types.")

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _get_feature_names(self) -> np.ndarray:
        """Retrieve output feature names after fitting the ColumnTransformer."""
        if self.pipeline is None:
            raise FeatureEngineeringError("Pipeline must be fitted before retrieving feature names.")

        try:
            names = self.pipeline.get_feature_names_out()
        except AttributeError:
            # Fallback for older scikit-learn: build names manually
            names = []
            for name, transformer, columns in self.pipeline.transformers_:
                if name == "numeric":
                    names.extend(columns)
                elif name == "categorical":
                    encoder: OneHotEncoder = transformer.named_steps["encoder"]
                    encoded_names = encoder.get_feature_names_out(columns)
                    names.extend(encoded_names)
            names = np.array(names)
        return names


def save_feature_outputs(
    X: pd.DataFrame,
    y: pd.Series,
    feature_matrix_path: str,
    target_path: str,
) -> None:
    """Persist the engineered feature matrix and target vector."""
    ensure_parent_dir(feature_matrix_path)
    ensure_parent_dir(target_path)

    try:
        X.to_parquet(feature_matrix_path, index=False)
        print(f"✓ Feature matrix saved to {feature_matrix_path}")
    except (ImportError, ValueError) as exc:
        fallback_path = str(Path(feature_matrix_path).with_suffix(".csv"))
        X.to_csv(fallback_path, index=False)
        print(
            f"⚠️ Parquet engine unavailable ({exc}). "
            f"Feature matrix saved as CSV fallback to {fallback_path}"
        )

    target_df = y.to_frame(name=self_or_default("target", y.name))
    target_df.to_csv(target_path, index=False)
    print(f"✓ Target vector saved to {target_path}")


def self_or_default(default_name: str, provided_name: Optional[str]) -> str:
    """Return provided name if available; otherwise, return default."""
    return provided_name if provided_name else default_name


def engineer_features(config: Optional[FeatureEngineeringConfig] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Orchestrate the full feature engineering workflow.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Transformed feature matrix and encoded target vector.
    """
    cfg = config or FeatureEngineeringConfig()

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    print(f"Loading cleaned data from: {cfg.clean_data_path}")

    df = load_clean_data(cfg.clean_data_path)
    print(f"✓ Loaded dataset with shape {df.shape}")

    engineer = FeatureEngineer(cfg)
    X, y = engineer.fit_transform(df)

    print(f"✓ Engineered features shape: {X.shape}")
    print(f"✓ Target vector length: {len(y)}")

    save_feature_outputs(X, y, cfg.feature_matrix_path, cfg.target_path)
    engineer.save_pipeline(cfg.pipeline_path)

    print("\n" + "-" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("-" * 60)
    print(f"Numeric features: {len(engineer.numeric_features_)}")
    print(f"Categorical features: {len(engineer.categorical_features_)}")
    print(f"Total transformed features: {X.shape[1]}")
    print(f"Pipeline saved to: {cfg.pipeline_path}")
    print(f"Feature matrix saved to: {cfg.feature_matrix_path}")
    print(f"Target vector saved to: {cfg.target_path}")
    print("-" * 60 + "\n")

    return X, y


if __name__ == "__main__":
    engineer_features()

