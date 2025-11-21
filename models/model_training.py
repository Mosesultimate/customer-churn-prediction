"""
Model Training Module for Customer Churn Prediction

This script loads engineered features, trains multiple classification models,
evaluates them using cross-validation, compares performance, and selects
the best model for churn prediction.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Try importing advanced models (optional dependencies)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.neural_network import MLPClassifier
    HAS_MLP = True
except ImportError:
    HAS_MLP = False


# Default paths
DEFAULT_FEATURE_MATRIX_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\features\feature_matrix.parquet"
DEFAULT_TARGET_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\features\target.csv"
DEFAULT_MODEL_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\churn_model.joblib"
DEFAULT_RESULTS_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\training_results.json"
DEFAULT_REPORT_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\training_report.txt"


class ModelTrainingError(Exception):
    """Custom exception raised for model training issues."""


@dataclass
class ModelTrainingConfig:
    """
    Configuration dataclass for model training pipeline.

    Attributes
    ----------
    feature_matrix_path: str
        Path to the engineered feature matrix (parquet or CSV).
    target_path: str
        Path to the target vector (CSV).
    model_path: str
        Path to save the best trained model.
    results_path: str
        Path to save training results as JSON.
    report_path: str
        Path to save detailed training report.
    test_size: float
        Proportion of dataset to use for testing (default: 0.2).
    cv_folds: int
        Number of folds for cross-validation (default: 5).
    random_state: int
        Random seed for reproducibility (default: 42).
    scoring_metric: str
        Primary metric for model selection (default: 'roc_auc').
    use_class_weight: bool
        Whether to use class weights for imbalanced data (default: True).
    """

    feature_matrix_path: str = DEFAULT_FEATURE_MATRIX_PATH
    target_path: str = DEFAULT_TARGET_PATH
    model_path: str = DEFAULT_MODEL_PATH
    results_path: str = DEFAULT_RESULTS_PATH
    report_path: str = DEFAULT_REPORT_PATH
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    scoring_metric: str = "roc_auc"
    use_class_weight: bool = True


@dataclass
class ModelResult:
    """Container for model evaluation results."""

    model_name: str
    model: object
    train_accuracy: float
    test_accuracy: float
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    cv_mean: Dict[str, float] = field(default_factory=dict)
    cv_std: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    prediction_time: float = 0.0


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a path if it does not exist."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def load_features_and_target(
    feature_path: str, target_path: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load feature matrix and target vector from disk.

    Parameters
    ----------
    feature_path: str
        Path to feature matrix (parquet or CSV).
    target_path: str
        Path to target vector (CSV).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix and target vector.
    """
    # Load features
    feature_file = Path(feature_path)
    if not feature_file.exists():
        # Try CSV fallback
        csv_path = str(feature_file.with_suffix(".csv"))
        if Path(csv_path).exists():
            print(f"⚠️ Parquet not found, loading CSV: {csv_path}")
            X = pd.read_csv(csv_path)
        else:
            raise ModelTrainingError(
                f"Feature matrix not found at {feature_path} or {csv_path}"
            )
    else:
        try:
            X = pd.read_parquet(feature_path)
        except Exception:
            # Fallback to CSV if parquet fails
            csv_path = str(feature_file.with_suffix(".csv"))
            print(f"⚠️ Parquet load failed, loading CSV: {csv_path}")
            X = pd.read_csv(csv_path)

    # Load target
    target_file = Path(target_path)
    if not target_file.exists():
        raise ModelTrainingError(f"Target vector not found at {target_path}")

    y_df = pd.read_csv(target_path)
    if "target" in y_df.columns:
        y = y_df["target"]
    elif y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        raise ModelTrainingError(f"Could not identify target column in {target_path}")

    print(f"✓ Loaded features: {X.shape}")
    print(f"✓ Loaded target: {y.shape}")
    print(f"✓ Class distribution: {y.value_counts().to_dict()}")

    return X, y


def get_class_weights(y: pd.Series) -> Optional[Dict[int, float]]:
    """
    Calculate class weights for imbalanced data.

    Parameters
    ----------
    y: pd.Series
        Target vector.

    Returns
    -------
    Optional[Dict[int, float]]
        Class weights dictionary or None.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def create_models(use_class_weight: bool = True, random_state: int = 42) -> Dict[str, object]:
    """
    Create a dictionary of classification models to evaluate.

    Parameters
    ----------
    use_class_weight: bool
        Whether to use class weights for imbalanced data.
    random_state: int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, object]
        Dictionary of model name -> model instance.
    """
    models = {}

    # Logistic Regression
    models["Logistic Regression"] = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="liblinear",
        class_weight="balanced" if use_class_weight else None,
    )

    # Random Forest
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced" if use_class_weight else None,
        n_jobs=-1,
    )

    # Gradient Boosting
    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
    )

    # Decision Tree
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced" if use_class_weight else None,
    )

    # SVM (using a subset for efficiency with large datasets)
    models["SVM (RBF)"] = SVC(
        kernel="rbf",
        probability=True,
        random_state=random_state,
        class_weight="balanced" if use_class_weight else None,
    )

    # XGBoost (if available)
    if HAS_XGBOOST:
        scale_pos_weight = None
        if use_class_weight:
            # Calculate scale_pos_weight for XGBoost
            # This is approximate; ideally compute from actual class distribution
            scale_pos_weight = 1.0  # Will be set properly during training

        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )

    # LightGBM (if available)
    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
            class_weight="balanced" if use_class_weight else None,
            verbose=-1,
        )

    # MLP (if available)
    if HAS_MLP:
        models["Neural Network (MLP)"] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

    return models


def evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
    model_name: str = "Model",
) -> ModelResult:
    """
    Evaluate a model using cross-validation and test set.

    Parameters
    ----------
    model: object
        Scikit-learn compatible classifier.
    X_train: pd.DataFrame
        Training features.
    y_train: pd.Series
        Training target.
    X_test: pd.DataFrame
        Test features.
    y_test: pd.Series
        Test target.
    cv_folds: int
        Number of cross-validation folds.
    random_state: int
        Random seed for reproducibility.
    model_name: str
        Name of the model for reporting.

    Returns
    -------
    ModelResult
        Evaluation results container.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Set scale_pos_weight for XGBoost if needed
    if HAS_XGBOOST and isinstance(model, xgb.XGBClassifier):
        if model.get_params().get("scale_pos_weight") is not None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                model.set_params(scale_pos_weight=neg_count / pos_count)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_scores = {}
    cv_mean = {}
    cv_std = {}

    print("Running cross-validation...")
    for metric_name, scoring in metrics.items():
        try:
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )
            cv_scores[metric_name] = scores.tolist()
            cv_mean[metric_name] = scores.mean()
            cv_std[metric_name] = scores.std()
            print(f"  CV {metric_name}: {cv_mean[metric_name]:.4f} (+/- {cv_std[metric_name]:.4f})")
        except Exception as e:
            print(f"  ⚠️ Could not compute {metric_name}: {e}")
            cv_scores[metric_name] = []
            cv_mean[metric_name] = 0.0
            cv_std[metric_name] = 0.0

    # Train model
    print("Training model...")
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Predictions
    print("Making predictions...")
    pred_start = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - pred_start

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    test_metrics = {
        "accuracy": test_accuracy,
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
    }

    try:
        test_metrics["roc_auc"] = roc_auc_score(y_test, y_test_pred_proba)
    except Exception:
        test_metrics["roc_auc"] = 0.0

    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Prediction Time: {pred_time:.2f}s")

    return ModelResult(
        model_name=model_name,
        model=model,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        cv_scores=cv_scores,
        cv_mean=cv_mean,
        cv_std=cv_std,
        test_metrics=test_metrics,
        training_time=train_time,
        prediction_time=pred_time,
    )


def select_best_model(
    results: List[ModelResult], scoring_metric: str = "roc_auc"
) -> ModelResult:
    """
    Select the best model based on cross-validation mean score.

    Parameters
    ----------
    results: List[ModelResult]
        List of model evaluation results.
    scoring_metric: str
        Metric to use for selection (default: 'roc_auc').

    Returns
    -------
    ModelResult
        Best model result.
    """
    if not results:
        raise ModelTrainingError("No model results to select from")

    best_result = max(
        results, key=lambda r: r.cv_mean.get(scoring_metric, 0.0)
    )
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_result.model_name}")
    print(f"  CV {scoring_metric}: {best_result.cv_mean.get(scoring_metric, 0.0):.4f}")
    print(f"  Test {scoring_metric}: {best_result.test_metrics.get(scoring_metric, 0.0):.4f}")
    print(f"{'='*60}")

    return best_result


def save_model_and_results(
    best_model: ModelResult,
    all_results: List[ModelResult],
    config: ModelTrainingConfig,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Save the best model, results JSON, and detailed report.

    Parameters
    ----------
    best_model: ModelResult
        Best model result to save.
    all_results: List[ModelResult]
        All model evaluation results.
    config: ModelTrainingConfig
        Training configuration.
    X_test: pd.DataFrame
        Test features for detailed report.
    y_test: pd.Series
        Test target for detailed report.
    """
    # Save model
    ensure_parent_dir(config.model_path)
    dump(best_model.model, config.model_path)
    print(f"✓ Best model saved to {config.model_path}")

    # Save results as JSON
    ensure_parent_dir(config.results_path)
    results_dict = {
        "best_model": best_model.model_name,
        "best_model_score": best_model.cv_mean.get(config.scoring_metric, 0.0),
        "models": [
            {
                "name": r.model_name,
                "cv_mean": r.cv_mean,
                "cv_std": r.cv_std,
                "test_metrics": r.test_metrics,
                "train_accuracy": r.train_accuracy,
                "test_accuracy": r.test_accuracy,
                "training_time": r.training_time,
                "prediction_time": r.prediction_time,
            }
            for r in all_results
        ],
    }

    with open(config.results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"✓ Results saved to {config.results_path}")

    # Generate detailed report
    ensure_parent_dir(config.report_path)
    y_test_pred = best_model.model.predict(X_test)
    y_test_pred_proba = best_model.model.predict_proba(X_test)[:, 1]

    with open(config.report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Best Model: {best_model.model_name}\n")
        f.write(f"Selection Metric: {config.scoring_metric}\n")
        f.write(f"Best CV Score: {best_model.cv_mean.get(config.scoring_metric, 0.0):.4f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("MODEL COMPARISON\n")
        f.write("-" * 80 + "\n\n")

        # Create comparison table
        f.write(f"{'Model':<25} {'CV ROC-AUC':<15} {'Test ROC-AUC':<15} {'Test F1':<15} {'Train Time':<15}\n")
        f.write("-" * 80 + "\n")
        for result in sorted(all_results, key=lambda r: r.cv_mean.get("roc_auc", 0.0), reverse=True):
            f.write(
                f"{result.model_name:<25} "
                f"{result.cv_mean.get('roc_auc', 0.0):<15.4f} "
                f"{result.test_metrics.get('roc_auc', 0.0):<15.4f} "
                f"{result.test_metrics.get('f1', 0.0):<15.4f} "
                f"{result.training_time:<15.2f}\n"
            )

        f.write("\n" + "-" * 80 + "\n")
        f.write("BEST MODEL DETAILED METRICS\n")
        f.write("-" * 80 + "\n\n")

        f.write("Cross-Validation Scores:\n")
        for metric, mean in best_model.cv_mean.items():
            std = best_model.cv_std.get(metric, 0.0)
            f.write(f"  {metric.upper()}: {mean:.4f} (+/- {std:.4f})\n")

        f.write("\nTest Set Metrics:\n")
        for metric, value in best_model.test_metrics.items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")

        f.write(f"\nTraining Time: {best_model.training_time:.2f}s\n")
        f.write(f"Prediction Time: {best_model.prediction_time:.2f}s\n\n")

        f.write("-" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n\n")
        f.write(classification_report(y_test, y_test_pred))

        f.write("\n" + "-" * 80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n\n")
        cm = confusion_matrix(y_test, y_test_pred)
        f.write(f"                Predicted\n")
        f.write(f"                No    Yes\n")
        f.write(f"Actual No    {cm[0][0]:5d} {cm[0][1]:5d}\n")
        f.write(f"       Yes   {cm[1][0]:5d} {cm[1][1]:5d}\n")

    print(f"✓ Detailed report saved to {config.report_path}")


def train_models(config: Optional[ModelTrainingConfig] = None) -> Tuple[object, List[ModelResult]]:
    """
    Main function to train and evaluate multiple models.

    Parameters
    ----------
    config: Optional[ModelTrainingConfig]
        Training configuration. Uses default if None.

    Returns
    -------
    Tuple[object, List[ModelResult]]
        Best trained model and all evaluation results.
    """
    cfg = config or ModelTrainingConfig()

    print("\n" + "=" * 80)
    print("MODEL TRAINING PIPELINE")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading data...")
    X, y = load_features_and_target(cfg.feature_matrix_path, cfg.target_path)

    # Split data
    print(f"\n[2/5] Splitting data (test_size={cfg.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Create models
    print(f"\n[3/5] Creating models...")
    models = create_models(
        use_class_weight=cfg.use_class_weight, random_state=cfg.random_state
    )
    print(f"  Created {len(models)} models: {', '.join(models.keys())}")

    # Evaluate models
    print(f"\n[4/5] Evaluating models (CV folds={cfg.cv_folds})...")
    results = []
    for name, model in models.items():
        try:
            result = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                cv_folds=cfg.cv_folds,
                random_state=cfg.random_state,
                model_name=name,
            )
            results.append(result)
        except Exception as e:
            print(f"  ⚠️ Failed to evaluate {name}: {e}")
            continue

    if not results:
        raise ModelTrainingError("No models were successfully evaluated")

    # Select best model
    print(f"\n[5/5] Selecting best model (metric={cfg.scoring_metric})...")
    best_result = select_best_model(results, cfg.scoring_metric)

    # Save results
    print(f"\n[6/6] Saving model and results...")
    save_model_and_results(best_result, results, cfg, X_test, y_test)

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)

    return best_result.model, results


if __name__ == "__main__":
    # Run training pipeline
    best_model, all_results = train_models()

    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ Best model: {all_results[0].model_name if all_results else 'N/A'}")
    print("\nNext steps:")
    print("  1. Review the training report for detailed metrics")
    print("  2. Use the saved model for predictions")
    print("  3. Optionally fine-tune hyperparameters for better performance")

