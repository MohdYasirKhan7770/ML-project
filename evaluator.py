"""
============================================================================
  Evaluator Module — Fake News Detection System
============================================================================
  Generates all evaluation artefacts:
    • Accuracy, Precision, Recall, F1 (per model)
    • Classification report (text)
    • Confusion matrix heatmap (saved as PNG)
    • Model comparison bar chart
============================================================================
"""

import os

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — no GUI windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


# ──────────────────────────────────────────────────────────────────────────
# 1. COMPUTE METRICS FOR A SINGLE MODEL
# ──────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Evaluate a single model on the test set.

    Returns
    -------
    dict  —  accuracy, precision, recall, f1, y_pred
    """
    y_pred = model.predict(X_test)

    metrics = {
        "name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "y_pred": y_pred,
    }

    print(f"\n{'─' * 50}")
    print(f"  Evaluation: {model_name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────
# 2. CLASSIFICATION REPORT (TEXT)
# ──────────────────────────────────────────────────────────────────────────
def print_classification_report(y_test, y_pred, model_name: str = "Model"):
    """Print the sklearn classification report."""
    target_names = ["Fake News", "Real News"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\n📋 Classification Report — {model_name}\n")
    print(report)
    return report


# ──────────────────────────────────────────────────────────────────────────
# 3. CONFUSION MATRIX HEATMAP
# ──────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name: str = "Model",
                          save: bool = True):
    """
    Plot and optionally save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Fake", "Real"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor="grey")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(config.REPORT_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        print(f"  [PLOT] Saved → {path}")

    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 4. EVALUATE ALL TRAINED MODELS
# ──────────────────────────────────────────────────────────────────────────
def evaluate_all(results: list, X_test, y_test):
    """
    Iterate through the list of training results, evaluate each on the
    test set, generate reports and confusion matrices.

    Returns
    -------
    list of metric dicts (one per model)
    """
    all_metrics = []

    for result in results:
        name = result["name"]
        model = result["best_model"]

        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        print_classification_report(y_test, metrics["y_pred"], model_name=name)
        plot_confusion_matrix(y_test, metrics["y_pred"], model_name=name)
        all_metrics.append(metrics)

    return all_metrics


# ──────────────────────────────────────────────────────────────────────────
# 5. MODEL COMPARISON BAR CHART
# ──────────────────────────────────────────────────────────────────────────
def plot_model_comparison(all_metrics: list, save: bool = True):
    """
    Create a grouped bar chart comparing Accuracy, Precision, Recall,
    and F1 across all models.
    """
    names = [m["name"] for m in all_metrics]
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, key in enumerate(metrics_keys):
        values = [m[key] for m in all_metrics]
        bars = ax.bar(x + i * width, values, width, label=key.capitalize(),
                      color=colors[i], edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Fake News Detection", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(config.REPORT_DIR, "model_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"\n  [PLOT] Saved → {path}")

    plt.close(fig)
    return fig
