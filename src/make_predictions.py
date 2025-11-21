"""
Prediction Module for Customer Churn

This script loads the trained model and feature pipeline to make predictions
on new customer data. It handles data preprocessing and feature engineering
automatically to ensure consistency with training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

# Import feature engineering module
import sys
sys.path.append(str(Path(__file__).parent))
from feature_engineering import FeatureEngineer, FeatureEngineeringConfig


# Default paths
DEFAULT_MODEL_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\churn_model.joblib"
DEFAULT_PIPELINE_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\feature_pipeline.joblib"
DEFAULT_OUTPUT_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\predictions\predictions.csv"


class PredictionError(Exception):
    """Custom exception raised for prediction issues."""


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a path if it does not exist."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def load_model_and_pipeline(
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
) -> Tuple[object, object]:
    """
    Load trained model and feature pipeline.

    Parameters
    ----------
    model_path: str
        Path to saved model (.joblib).
    pipeline_path: str
        Path to saved feature pipeline (.joblib).

    Returns
    -------
    Tuple[object, object]
        Loaded model and feature pipeline.
    """
    model_file = Path(model_path)
    pipeline_file = Path(pipeline_path)

    if not model_file.exists():
        raise PredictionError(f"Model not found at {model_path}")

    if not pipeline_file.exists():
        raise PredictionError(f"Feature pipeline not found at {pipeline_path}")

    print(f"Loading model from {model_path}...")
    model = load(model_path)

    print(f"Loading feature pipeline from {pipeline_path}...")
    pipeline = load(pipeline_path)

    print("✓ Model and pipeline loaded successfully")
    return model, pipeline


def prepare_data_for_prediction(
    df: pd.DataFrame,
    pipeline: object,
    feature_engineer: Optional[FeatureEngineer] = None,
) -> pd.DataFrame:
    """
    Prepare and transform data for prediction using the feature pipeline.

    Parameters
    ----------
    df: pd.DataFrame
        Raw customer data.
    pipeline: object
        Fitted feature pipeline (ColumnTransformer).
    feature_engineer: Optional[FeatureEngineer]
        Feature engineer instance for domain feature creation.

    Returns
    -------
    pd.DataFrame
        Transformed feature matrix ready for prediction.
    """
    if feature_engineer is None:
        # Create feature engineer with default config to use its methods
        config = FeatureEngineeringConfig()
        feature_engineer = FeatureEngineer(config)

    # Create domain features (same as training)
    processed_df = feature_engineer._create_domain_features(df.copy())

    # Drop target column if present (for prediction)
    target_col = feature_engineer.config.target_column
    if target_col in processed_df.columns:
        processed_df = processed_df.drop(columns=[target_col])

    # Drop ID columns
    for col in feature_engineer.config.drop_columns:
        if col in processed_df.columns:
            processed_df = processed_df.drop(columns=[col])

    # Infer feature types to match training
    numeric_features, categorical_features = feature_engineer._infer_feature_types(processed_df)

    # Reorder columns to match training order (numeric first, then categorical)
    all_features = numeric_features + categorical_features
    missing_features = [f for f in all_features if f not in processed_df.columns]
    if missing_features:
        # Add missing columns with default values
        for feat in missing_features:
            if feat in numeric_features:
                processed_df[feat] = 0
            else:
                processed_df[feat] = processed_df[categorical_features[0]].mode()[0] if categorical_features else 'Unknown'
    
    # Reorder to match training order
    processed_df = processed_df[all_features]

    # Transform using pipeline
    X_array = pipeline.transform(processed_df)

    # Get feature names
    try:
        feature_names = pipeline.get_feature_names_out()
    except AttributeError:
        # Fallback: try to get from transformer attributes
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    X_transformed = pd.DataFrame(
        X_array, columns=feature_names, index=processed_df.index
    )

    return X_transformed


def predict_churn(
    data: pd.DataFrame,
    model: object,
    pipeline: object,
    feature_engineer: Optional[FeatureEngineer] = None,
    return_probabilities: bool = True,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Make churn predictions on customer data.

    Parameters
    ----------
    data: pd.DataFrame
        Raw customer data to predict on.
    model: object
        Trained classification model.
    pipeline: object
        Fitted feature pipeline.
    feature_engineer: Optional[FeatureEngineer]
        Feature engineer instance.
    return_probabilities: bool
        Whether to return prediction probabilities.
    threshold: float
        Probability threshold for binary classification (default: 0.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and (optionally) probabilities.
    """
    print(f"\nPreparing {len(data)} customer(s) for prediction...")

    # Prepare features
    X = prepare_data_for_prediction(data, pipeline, feature_engineer)

    print(f"✓ Features prepared: {X.shape}")

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X)
    probabilities = None

    if return_probabilities:
        probabilities = model.predict_proba(X)[:, 1]  # Probability of churn (class 1)

    # Create results DataFrame
    results = pd.DataFrame(index=data.index)

    # Add original data identifiers if available
    if "customerID" in data.columns:
        results["customerID"] = data["customerID"].values

    results["churn_prediction"] = predictions
    results["churn_prediction_label"] = results["churn_prediction"].map({0: "No", 1: "Yes"})

    if probabilities is not None:
        results["churn_probability"] = probabilities
        results["churn_risk"] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["Low", "Medium", "High", "Very High"],
            include_lowest=True,
        )

    print(f"✓ Predictions completed")
    print(f"  - Predicted churn: {predictions.sum()} customer(s)")
    print(f"  - Predicted no churn: {(predictions == 0).sum()} customer(s)")

    return results


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> None:
    """
    Save predictions to CSV file.

    Parameters
    ----------
    predictions: pd.DataFrame
        Predictions DataFrame to save.
    output_path: str
        Path to save predictions CSV.
    """
    ensure_parent_dir(output_path)
    predictions.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to {output_path}")


def predict_from_csv(
    input_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
    output_path: Optional[str] = None,
    return_probabilities: bool = True,
) -> pd.DataFrame:
    """
    Load data from CSV, make predictions, and save results.

    Parameters
    ----------
    input_path: str
        Path to input CSV with customer data.
    model_path: str
        Path to trained model.
    pipeline_path: str
        Path to feature pipeline.
    output_path: Optional[str]
        Path to save predictions. Uses default if None.
    return_probabilities: bool
        Whether to include prediction probabilities.

    Returns
    -------
    pd.DataFrame
        Predictions DataFrame.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH

    print("=" * 80)
    print("CHURN PREDICTION")
    print("=" * 80)

    # Load data
    print(f"\n[1/4] Loading data from {input_path}...")
    if not Path(input_path).exists():
        raise PredictionError(f"Input file not found at {input_path}")

    data = pd.read_csv(input_path)
    print(f"✓ Loaded {len(data)} customer(s)")

    # Load model and pipeline
    print(f"\n[2/4] Loading model and pipeline...")
    model, pipeline = load_model_and_pipeline(model_path, pipeline_path)

    # Make predictions
    print(f"\n[3/4] Making predictions...")
    predictions = predict_churn(
        data, model, pipeline, return_probabilities=return_probabilities
    )

    # Save predictions
    print(f"\n[4/4] Saving predictions...")
    save_predictions(predictions, output_path)

    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)

    return predictions


def predict_single_customer(
    customer_data: Dict,
    model: Optional[object] = None,
    pipeline: Optional[object] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline_path: str = DEFAULT_PIPELINE_PATH,
) -> Dict:
    """
    Make prediction for a single customer (useful for API/web app).

    Parameters
    ----------
    customer_data: Dict
        Dictionary with customer features.
    model: Optional[object]
        Pre-loaded model. Loads from file if None.
    pipeline: Optional[object]
        Pre-loaded pipeline. Loads from file if None.
    model_path: str
        Path to model if not pre-loaded.
    pipeline_path: str
        Path to pipeline if not pre-loaded.

    Returns
    -------
    Dict
        Dictionary with prediction results.
    """
    # Load model/pipeline if not provided
    if model is None or pipeline is None:
        model, pipeline = load_model_and_pipeline(model_path, pipeline_path)

    # Convert dict to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Make prediction
    results = predict_churn(customer_df, model, pipeline, return_probabilities=True)

    # Return as dictionary
    result_dict = {
        "churn_prediction": int(results["churn_prediction"].iloc[0]),
        "churn_prediction_label": results["churn_prediction_label"].iloc[0],
        "churn_probability": float(results["churn_probability"].iloc[0]),
        "churn_risk": str(results["churn_risk"].iloc[0]),
    }

    return result_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make churn predictions on customer data")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV file with customer data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model (.joblib)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=DEFAULT_PIPELINE_PATH,
        help="Path to feature pipeline (.joblib)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save predictions CSV",
    )

    args = parser.parse_args()

    if args.input is None:
        print("Error: --input argument required")
        print("\nUsage:")
        print("  python make_predictions.py --input path/to/customers.csv")
        print("\nOptional arguments:")
        print("  --model path/to/model.joblib")
        print("  --pipeline path/to/pipeline.joblib")
        print("  --output path/to/predictions.csv")
    else:
        predict_from_csv(args.input, args.model, args.pipeline, args.output)

