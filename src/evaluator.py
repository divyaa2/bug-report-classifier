"""
evaluator.py
------------
Computes and displays evaluation metrics, confusion matrices, and
comparison plots for all trained models.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_avg(y_true, y_pred, metric_fn, **kwargs):
    """Return weighted-average metric, handling zero-division gracefully."""
    return metric_fn(y_true, y_pred, average="weighted",
                     zero_division=0, **kwargs)


# ── per-model evaluation ─────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted pipeline and return a metrics dictionary.

    Parameters
    ----------
    model_name : str
    pipeline : fitted sklearn Pipeline
    X_test, y_test : pd.Series

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, f1,
                    report, confusion_matrix, labels
    """
    y_pred = pipeline.predict(X_test)
    labels = sorted(y_test.unique().tolist())

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": _safe_avg(y_test, y_pred, precision_score),
        "recall": _safe_avg(y_test, y_pred, recall_score),
        "f1": _safe_avg(y_test, y_pred, f1_score),
        "report": classification_report(
            y_test, y_pred, zero_division=0, target_names=labels
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    # Pretty-print to console
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}  (weighted)")
    print(f"  Recall   : {metrics['recall']:.4f}  (weighted)")
    print(f"  F1-Score : {metrics['f1']:.4f}  (weighted)")
    print(f"\n  Classification Report:\n{metrics['report']}")

    return metrics


# ── confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    metrics: dict,
    output_dir: str = "results",
) -> str:
    """
    Save a heatmap of the confusion matrix to *output_dir*.

    Returns the file path of the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]
    model_name = metrics["model"]

    # Shorten long label names for readability
    short_labels = [lbl[:20] for lbl in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluator] Confusion matrix saved → {path}")
    return path


# ── comparison bar chart ──────────────────────────────────────────────────────

def plot_model_comparison(
    all_metrics: list[dict],
    output_dir: str = "results",
) -> str:
    """
    Bar chart comparing Accuracy, Precision, Recall, and F1 across models.

    Returns the file path of the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)", "F1\n(weighted)"]
    model_names = [m["model"] for m in all_metrics]

    x = np.arange(len(metric_keys))
    width = 0.8 / len(model_names)
    offsets = np.linspace(-(len(model_names) - 1) / 2, (len(model_names) - 1) / 2, len(model_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for i, (m, offset) in enumerate(zip(all_metrics, offsets)):
        values = [m[k] for k in metric_keys]
        bars = ax.bar(
            x + offset * width,
            values,
            width,
            label=m["model"],
            color=colors[i % len(colors)],
            edgecolor="white",
        )
        # Annotate bar tops
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Bug Report Classification", fontsize=13, pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluator] Comparison chart saved → {path}")
    return path


# ── save metrics to JSON ──────────────────────────────────────────────────────

def save_metrics(
    all_metrics: list[dict],
    output_dir: str = "results",
) -> str:
    """
    Persist all metrics to a JSON file.

    Returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # confusion_matrix is already a list (JSON-serialisable)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"metrics_{timestamp}.json")

    serialisable = []
    for m in all_metrics:
        entry = {k: v for k, v in m.items() if k != "confusion_matrix"}
        entry["confusion_matrix"] = m["confusion_matrix"]
        serialisable.append(entry)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)

    print(f"[evaluator] Metrics saved → {path}")
    return path
