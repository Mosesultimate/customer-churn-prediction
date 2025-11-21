"""
Model Evaluation Module for Customer Churn Prediction

This script provides comprehensive model evaluation including:
- Feature importance analysis
- SHAP values for model explainability
- ROC curves and AUC scores
- Precision-Recall curves
- Confusion matrix analysis
- Threshold optimization
- Model calibration plots
- Detailed performance metrics
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    brier_score_loss,
)

# Handle calibration_curve import for different sklearn versions
# In older versions (< 0.22), it's in sklearn.calibration
# In newer versions (>= 0.22), it's in sklearn.metrics
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        # If neither works, define a simple fallback or disable calibration
        calibration_curve = None
        print("⚠️ calibration_curve not available. Calibration plots will be skipped.")
from sklearn.model_selection import train_test_split

# Try importing SHAP for explainability (optional dependency)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not available. Install with: pip install shap")


# Default paths
DEFAULT_FEATURE_MATRIX_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\features\feature_matrix.parquet"
DEFAULT_TARGET_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\data\processed\features\target.csv"
DEFAULT_MODEL_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\churn_model.joblib"
DEFAULT_RESULTS_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\models\training_results.json"
DEFAULT_EVALUATION_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\reports\model_evaluation.json"
DEFAULT_REPORT_PATH = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\reports\evaluation_report.txt"
DEFAULT_PLOTS_DIR = r"C:\Users\ADMIN\Desktop\DataAnalytics\customer-churn-prediction\reports\plots"


class ModelEvaluationError(Exception):
    """Custom exception raised for model evaluation issues."""


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a path if it does not exist."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def load_model_and_data(
    model_path: str = DEFAULT_MODEL_PATH,
    feature_path: str = DEFAULT_FEATURE_MATRIX_PATH,
    target_path: str = DEFAULT_TARGET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[object, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load trained model and feature/target data, split into train/test.

    Parameters
    ----------
    model_path: str
        Path to trained model.
    feature_path: str
        Path to feature matrix.
    target_path: str
        Path to target vector.
    test_size: float
        Proportion of data for testing.
    random_state: int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[object, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        Model, X_train, y_train, X_test, y_test
    """
    print("=" * 80)
    print("LOADING MODEL AND DATA")
    print("=" * 80)

    # Load model
    print(f"\n[1/3] Loading model from {model_path}...")
    if not Path(model_path).exists():
        raise ModelEvaluationError(f"Model not found at {model_path}")
    model = load(model_path)
    print(f"✓ Model loaded: {type(model).__name__}")

    # Load features
    print(f"\n[2/3] Loading features from {feature_path}...")
    feature_file = Path(feature_path)
    if feature_file.exists():
        try:
            X = pd.read_parquet(feature_path)
        except Exception:
            csv_path = str(feature_file.with_suffix(".csv"))
            if Path(csv_path).exists():
                X = pd.read_csv(csv_path)
            else:
                raise ModelEvaluationError(f"Features not found at {feature_path} or {csv_path}")
    else:
        csv_path = str(feature_file.with_suffix(".csv"))
        if Path(csv_path).exists():
            X = pd.read_csv(csv_path)
        else:
            raise ModelEvaluationError(f"Features not found at {feature_path} or {csv_path}")

    print(f"✓ Features loaded: {X.shape}")

    # Load target
    print(f"\n[3/3] Loading target from {target_path}...")
    if not Path(target_path).exists():
        raise ModelEvaluationError(f"Target not found at {target_path}")

    y_df = pd.read_csv(target_path)
    if "target" in y_df.columns:
        y = y_df["target"]
    elif y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        raise ModelEvaluationError(f"Could not identify target column in {target_path}")

    print(f"✓ Target loaded: {y.shape}")

    # Ensure indices align
    if len(X) != len(y):
        raise ModelEvaluationError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")

    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    return model, X_train, y_train, X_test, y_test


def calculate_feature_importance(
    model: object, feature_names: List[str]
) -> Optional[pd.DataFrame]:
    """
    Calculate feature importance for tree-based models.

    Parameters
    ----------
    model: object
        Trained model.
    feature_names: List[str]
        List of feature names.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with feature importances, or None if not available.
    """
    importance_dict = {}

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importance_dict = dict(zip(feature_names, model.feature_importances_))
    # Linear models (coefficients)
    elif hasattr(model, "coef_"):
        # Take absolute value and average across classes if multi-class
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef[0])
        importance_dict = dict(zip(feature_names, importance))
    # SVM with linear kernel
    elif hasattr(model, "coef_") and hasattr(model, "kernel"):
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef[0])
        importance_dict = dict(zip(feature_names, importance))
    else:
        return None

    # Create DataFrame and sort
    importance_df = pd.DataFrame(
        {"feature": list(importance_dict.keys()), "importance": list(importance_dict.values())}
    ).sort_values("importance", ascending=False)

    # Normalize to percentages
    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    )

    return importance_df


def calculate_shap_values(
    model: object, X_test: pd.DataFrame, max_samples: int = 100
) -> Optional[Tuple[np.ndarray, shap.Explainer]]:
    """
    Calculate SHAP values for model explainability.

    Parameters
    ----------
    model: object
        Trained model.
    X_test: pd.DataFrame
        Test features.
    max_samples: int
        Maximum number of samples to use (for performance).

    Returns
    -------
    Optional[Tuple[np.ndarray, shap.Explainer]]
        SHAP values and explainer, or None if SHAP not available.
    """
    if not HAS_SHAP:
        return None

    print(f"\nCalculating SHAP values (max {max_samples} samples)...")

    try:
        # Limit samples for performance
        if len(X_test) > max_samples:
            X_sample = X_test.sample(n=max_samples, random_state=42)
        else:
            X_sample = X_test

        # Create explainer
        explainer = shap.TreeExplainer(model) if hasattr(model, "tree_") or hasattr(model, "estimators_") else shap.KernelExplainer(model.predict_proba, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        print(f"✓ SHAP values calculated for {len(X_sample)} samples")
        return shap_values, explainer

    except Exception as e:
        print(f"⚠️ Could not calculate SHAP values: {e}")
        return None


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    model_name: str = "Model",
) -> None:
    """
    Plot ROC curve and save.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_prob: np.ndarray
        Predicted probabilities.
    save_path: Optional[str]
        Path to save plot.
    model_name: str
        Model name for plot title.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ ROC curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    model_name: str = "Model",
) -> None:
    """
    Plot Precision-Recall curve and save.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_prob: np.ndarray
        Predicted probabilities.
    save_path: Optional[str]
        Path to save plot.
    model_name: str
        Model name for plot title.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.3f})", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Precision-Recall curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    model_name: str = "Model",
) -> None:
    """
    Plot confusion matrix and save.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_pred: np.ndarray
        Predicted labels.
    save_path: Optional[str]
        Path to save plot.
    model_name: str
        Model name for plot title.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    # Add percentages
    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({cm_percent[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
    model_name: str = "Model",
) -> None:
    """
    Plot feature importance and save.

    Parameters
    ----------
    importance_df: pd.DataFrame
        DataFrame with feature importances.
    top_n: int
        Number of top features to plot.
    save_path: Optional[str]
        Path to save plot.
    model_name: str
        Model name for plot title.
    """
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top_features)), top_features["importance_pct"].values[::-1])
    plt.yticks(range(len(top_features)), top_features["feature"].values[::-1])
    plt.xlabel("Importance (%)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Feature importance plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    model_name: str = "Model",
) -> None:
    """
    Plot calibration curve and save.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_prob: np.ndarray
        Predicted probabilities.
    save_path: Optional[str]
        Path to save plot.
    model_name: str
        Model name for plot title.
    """
    if calibration_curve is None:
        print("⚠️ calibration_curve not available. Skipping calibration plot.")
        return
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10
    )

    brier_score = brier_score_loss(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label=f"{model_name} (Brier = {brier_score:.3f})",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title(f"Calibration Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Calibration curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.

    Parameters
    ----------
    y_true: np.ndarray
        True labels.
    y_prob: np.ndarray
        Predicted probabilities.
    metric: str
        Metric to optimize ('f1', 'f0.5', 'f2', 'youden').

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Optimal threshold and metrics at that threshold.
    """
    from sklearn.metrics import fbeta_score

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0

    if metric == "youden":
        # Use Youden's J statistic (TPR - FPR)
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
        youden_scores = tpr - fpr
        best_idx = np.argmax(youden_scores)
        best_threshold = thresholds_roc[best_idx] if best_idx < len(thresholds_roc) else 0.5
    else:
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "f0.5":
                score = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)  # Favor precision
            elif metric == "f2":
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)  # Favor recall
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

    # Calculate metrics at best threshold
    y_pred_best = (y_prob >= best_threshold).astype(int)
    metrics_at_threshold = {
        "threshold": float(best_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred_best)),
        "precision": float(precision_score(y_true, y_pred_best, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_best, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_best, zero_division=0)),
    }

    return float(best_threshold), metrics_at_threshold


def evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
    plots_dir: Optional[str] = None,
    model_name: str = "Model",
) -> Dict:
    """
    Comprehensive model evaluation.

    Parameters
    ----------
    model: object
        Trained model.
    X_train: pd.DataFrame
        Training features.
    y_train: pd.Series
        Training labels.
    X_test: pd.DataFrame
        Test features.
    y_test: pd.Series
        Test labels.
    feature_names: Optional[List[str]]
        Feature names. Uses X_test columns if None.
    plots_dir: Optional[str]
        Directory to save plots. Creates plots if provided.
    model_name: str
        Model name for reporting.

    Returns
    -------
    Dict
        Dictionary with all evaluation metrics and results.
    """
    if feature_names is None:
        feature_names = X_test.columns.tolist()

    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 80)

    # Make predictions
    print("\n[1/6] Making predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    print("✓ Predictions completed")

    # Calculate metrics
    print("\n[2/6] Calculating metrics...")
    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_train_prob),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_prob),
            "average_precision": average_precision_score(y_test, y_test_prob),
            "brier_score": brier_score_loss(y_test, y_test_prob),
        },
    }

    print(f"✓ Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"✓ Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"✓ Test F1: {metrics['test']['f1']:.4f}")

    # Feature importance
    print("\n[3/6] Calculating feature importance...")
    importance_df = calculate_feature_importance(model, feature_names)
    if importance_df is not None:
        print(f"✓ Top 5 features: {', '.join(importance_df.head(5)['feature'].tolist())}")
    else:
        print("⚠️ Feature importance not available for this model type")

    # SHAP values
    print("\n[4/6] Calculating SHAP values...")
    shap_results = calculate_shap_values(model, X_test)
    shap_values = None
    if shap_results is not None:
        shap_values, _ = shap_results
        print("✓ SHAP values calculated")

    # Threshold optimization
    print("\n[5/6] Optimizing classification threshold...")
    optimal_threshold, threshold_metrics = optimize_threshold(y_test, y_test_prob, metric="f1")
    print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Metrics at optimal threshold:")
    print(f"    Accuracy: {threshold_metrics['accuracy']:.4f}")
    print(f"    F1: {threshold_metrics['f1']:.4f}")

    # Generate plots
    print("\n[6/6] Generating evaluation plots...")
    if plots_dir:
        ensure_parent_dir(plots_dir)

        # ROC curve
        plot_roc_curve(
            y_test, y_test_prob, save_path=f"{plots_dir}/roc_curve.png", model_name=model_name
        )

        # Precision-Recall curve
        plot_precision_recall_curve(
            y_test,
            y_test_prob,
            save_path=f"{plots_dir}/precision_recall_curve.png",
            model_name=model_name,
        )

        # Confusion matrix
        plot_confusion_matrix(
            y_test,
            y_test_pred,
            save_path=f"{plots_dir}/confusion_matrix.png",
            model_name=model_name,
        )

        # Feature importance
        if importance_df is not None:
            plot_feature_importance(
                importance_df,
                top_n=20,
                save_path=f"{plots_dir}/feature_importance.png",
                model_name=model_name,
            )

        # Calibration curve (if available)
        if calibration_curve is not None:
            plot_calibration_curve(
                y_test,
                y_test_prob,
                save_path=f"{plots_dir}/calibration_curve.png",
                model_name=model_name,
            )
        else:
            print("  ⚠️ Skipping calibration curve (not available)")

        print(f"✓ All plots saved to {plots_dir}")

    # Compile results
    results = {
        "model_name": model_name,
        "metrics": metrics,
        "optimal_threshold": optimal_threshold,
        "threshold_metrics": threshold_metrics,
        "feature_importance": importance_df.to_dict(orient="records") if importance_df is not None else None,
        "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
    }

    return results


def generate_evaluation_report(
    results: Dict,
    save_path: str = DEFAULT_REPORT_PATH,
) -> None:
    """
    Generate detailed text evaluation report.

    Parameters
    ----------
    results: Dict
        Evaluation results dictionary.
    save_path: str
        Path to save report.
    """
    ensure_parent_dir(save_path)

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {results['model_name']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n\n")

        f.write("Training Set:\n")
        for metric, value in results["metrics"]["train"].items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")

        f.write("\nTest Set:\n")
        for metric, value in results["metrics"]["test"].items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("THRESHOLD OPTIMIZATION\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"Optimal Threshold: {results['optimal_threshold']:.3f}\n")
        f.write("Metrics at Optimal Threshold:\n")
        for metric, value in results["threshold_metrics"].items():
            if metric != "threshold":
                f.write(f"  {metric.upper()}: {value:.4f}\n")

        if results["feature_importance"]:
            f.write("\n" + "-" * 80 + "\n")
            f.write("TOP 20 FEATURE IMPORTANCES\n")
            f.write("-" * 80 + "\n\n")
            for i, feat in enumerate(results["feature_importance"][:20], 1):
                f.write(
                    f"{i:2d}. {feat['feature']:<40} {feat['importance_pct']:>6.2f}%\n"
                )

        f.write("\n" + "-" * 80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n\n")
        if "classification_report" in results and results["classification_report"]:
            cr_dict = results["classification_report"]
            f.write("Per-Class Metrics:\n")
            for class_key, class_metrics in cr_dict.items():
                if isinstance(class_metrics, dict):
                    f.write(f"\n{class_key}:\n")
                    for metric, val in class_metrics.items():
                        if isinstance(val, (int, float)):
                            f.write(f"  {metric}: {val:.4f}\n")
                elif isinstance(class_metrics, (int, float)):
                    f.write(f"{class_key}: {class_metrics:.4f}\n")

    print(f"✓ Evaluation report saved to {save_path}")


def run_evaluation(
    model_path: str = DEFAULT_MODEL_PATH,
    feature_path: str = DEFAULT_FEATURE_MATRIX_PATH,
    target_path: str = DEFAULT_TARGET_PATH,
    evaluation_path: str = DEFAULT_EVALUATION_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    plots_dir: str = DEFAULT_PLOTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Main function to run complete model evaluation.

    Parameters
    ----------
    model_path: str
        Path to trained model.
    feature_path: str
        Path to feature matrix.
    target_path: str
        Path to target vector.
    evaluation_path: str
        Path to save evaluation JSON.
    report_path: str
        Path to save text report.
    plots_dir: str
        Directory to save plots.
    test_size: float
        Test set proportion.
    random_state: int
        Random seed.

    Returns
    -------
    Dict
        Evaluation results dictionary.
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 80)

    # Load model and data
    model, X_train, y_train, X_test, y_test = load_model_and_data(
        model_path, feature_path, target_path, test_size, random_state
    )

    # Get model name
    model_name = type(model).__name__

    # Run evaluation
    results = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names=X_test.columns.tolist(),
        plots_dir=plots_dir,
        model_name=model_name,
    )

    # Save results
    print(f"\nSaving evaluation results...")
    ensure_parent_dir(evaluation_path)
    with open(evaluation_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Evaluation results saved to {evaluation_path}")

    # Generate report
    generate_evaluation_report(results, report_path)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained churn prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model (.joblib)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=DEFAULT_FEATURE_MATRIX_PATH,
        help="Path to feature matrix",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET_PATH,
        help="Path to target vector",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_EVALUATION_PATH,
        help="Path to save evaluation JSON",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path to save text report",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to save plots",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        feature_path=args.features,
        target_path=args.target,
        evaluation_path=args.output,
        report_path=args.report,
        plots_dir=args.plots,
    )

    print("\n✓ Model evaluation completed successfully!")
    print("\nNext steps:")
    print("  1. Review evaluation report for detailed metrics")
    print("  2. Check plots directory for visualizations")
    print("  3. Use optimal threshold for predictions if needed")

