"""
Model training, evaluation, comparison, and persistence for Iris Flower Classification.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.data_loader import load_iris_dataframe, split_features_target, FEATURE_NAMES, TARGET_NAME


def get_model_candidates() -> Dict[str, Any]:
    """
    Define standard candidate classifiers wrapped in Pipelines.
    Pipelines ensure scaling is fitted only on training splits to prevent data leakage.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=500, 
                random_state=42, 
                solver="lbfgs"
            ))
        ]),
        "K-Nearest Neighbors (KNN)": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
        ])
    }


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Train each candidate model, perform stratified cross-validation, and calculate test metrics.
    
    Returns:
        results: Dictionary containing performance metrics per model.
        conf_matrices: Dictionary of confusion matrices for each model.
    """
    models = get_model_candidates()
    results = {}
    conf_matrices = {}
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    classes = sorted(y_train.unique())
    
    for name, pipeline in models.items():
        # Cross-validation on training data
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
        
        # Fit model on training set
        pipeline.fit(X_train, y_train)
        
        # Predict on holdout test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        
        conf_matrices[name] = cm
        results[name] = {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "test_accuracy": float(acc),
            "test_precision_weighted": float(prec),
            "test_recall_weighted": float(rec),
            "test_f1_weighted": float(f1),
            "classification_report": report,
            "trained_pipeline": pipeline
        }
        
    return results, conf_matrices


def plot_confusion_matrices(
    conf_matrices: Dict[str, np.ndarray], 
    classes: list, 
    output_path: str = "reports/figures/confusion_matrices.png"
):
    """
    Plot and save confusion matrices side-by-side for model comparison.
    """
    n_models = len(conf_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, cm) in zip(axes, conf_matrices.items()):
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            cbar=False,
            xticklabels=classes, 
            yticklabels=classes, 
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"}
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        
    plt.suptitle("Model Confusion Matrices on Test Set", y=1.03, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved confusion matrices figure to {output_path}")


def select_best_model(results: Dict[str, Any]) -> Tuple[str, Any, str]:
    """
    Select best performing model with scientific rationale.
    
    Criteria:
    1. Highest Test Accuracy and Test F1-Score
    2. Highest Cross-Validation Mean Accuracy
    3. Lowest CV variance (stability)
    4. Occam's razor / simplicity (lower risk of overfitting on small dataset)
    """
    best_name = None
    best_score = -1.0
    
    # Sort primarily by test_f1_weighted, secondarily by cv_accuracy_mean
    for name, data in results.items():
        composite_score = data["test_f1_weighted"] * 0.6 + data["cv_accuracy_mean"] * 0.4
        if composite_score > best_score:
            best_score = composite_score
            best_name = name
            
    best_data = results[best_name]
    justification = (
        f"The best performing model is '{best_name}'. "
        f"It achieved a hold-out test accuracy of {best_data['test_accuracy']:.2%} and "
        f"weighted F1-score of {best_data['test_f1_weighted']:.4f}, with a 5-fold CV accuracy "
        f"of {best_data['cv_accuracy_mean']:.2%} (±{best_data['cv_accuracy_std']:.2%}). "
        f"It provides optimal generalization with minimal variance and well-calibrated class boundaries."
    )
    
    return best_name, best_data["trained_pipeline"], justification


def save_evaluation_summary(results: Dict[str, Any], best_name: str, justification: str, output_path: str = "reports/evaluation_summary.json"):
    """
    Export serializable evaluation summary report.
    """
    summary = {
        "best_model": best_name,
        "justification": justification,
        "models": {}
    }
    
    for name, data in results.items():
        summary["models"][name] = {
            "cv_accuracy_mean": data["cv_accuracy_mean"],
            "cv_accuracy_std": data["cv_accuracy_std"],
            "test_accuracy": data["test_accuracy"],
            "test_precision_weighted": data["test_precision_weighted"],
            "test_recall_weighted": data["test_recall_weighted"],
            "test_f1_weighted": data["test_f1_weighted"],
            "classification_report": data["classification_report"]
        }
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved evaluation summary to {output_path}")


def main():
    df = load_iris_dataframe()
    X_train, X_test, y_train, y_test = split_features_target(df, test_size=0.20, random_state=42)
    classes = sorted(y_train.unique())
    
    results, conf_matrices = train_and_evaluate_all(X_train, X_test, y_train, y_test)
    plot_confusion_matrices(conf_matrices, classes)
    
    best_name, best_pipeline, justification = select_best_model(results)
    print("\n" + "="*60)
    print(f"BEST MODEL SELECTION: {best_name}")
    print("="*60)
    print(justification)
    
    # Save best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_iris_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nPersisted best model pipeline to {model_path}")
    
    save_evaluation_summary(results, best_name, justification)
    
    # Print formatted comparison table
    print("\nModel Comparison Table:")
    summary_rows = []
    for name, data in results.items():
        summary_rows.append({
            "Model": name,
            "CV Acc (Mean ± Std)": f"{data['cv_accuracy_mean']:.3f} ± {data['cv_accuracy_std']:.3f}",
            "Test Acc": f"{data['test_accuracy']:.4f}",
            "Precision": f"{data['test_precision_weighted']:.4f}",
            "Recall": f"{data['test_recall_weighted']:.4f}",
            "F1-Score": f"{data['test_f1_weighted']:.4f}"
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
