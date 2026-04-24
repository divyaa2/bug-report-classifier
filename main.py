#!/usr/bin/env python3
"""
main.py
-------
Entry point for the Bug Report Classification tool.

Usage
-----
  # Train all models and evaluate:
  python main.py

  # Predict the component for a custom bug report:
  python main.py --predict "Firefox crashes when opening a new tab on Linux"

  # Use a different dataset:
  python main.py --data path/to/other.csv
"""

import argparse
import sys

from src.data_loader import load_dataset, select_features_and_label
from src.preprocessor import preprocess, clean_text
from src.models import MODEL_REGISTRY
from src.trainer import split_data, train_model
from src.evaluator import (
    evaluate_model,
    plot_confusion_matrix,
    plot_model_comparison,
    save_metrics,
)

DEFAULT_DATA_PATH = "sample_mozilla_core.csv"
RESULTS_DIR = "results"


# ── training / evaluation pipeline ───────────────────────────────────────────

def run_pipeline(data_path: str) -> tuple[dict, list]:
    """
    Full train-evaluate pipeline.

    Returns
    -------
    trained_models : dict  {model_name: fitted_pipeline}
    all_metrics    : list  [metrics_dict, …]
    """
    # 1. Load
    raw_df = load_dataset(data_path)

    # 2. Select features + label
    df = select_features_and_label(raw_df)

    # 3. Preprocess
    df = preprocess(df)

    if len(df) < 10:
        print(
            "[main] ERROR: Too few samples after preprocessing "
            f"({len(df)}). Cannot train reliably."
        )
        sys.exit(1)

    # 4. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 5. Train + evaluate each model
    trained_models: dict = {}
    all_metrics: list = []

    for model_name, builder_fn in MODEL_REGISTRY.items():
        print(f"\n[main] Training: {model_name} …")
        pipeline = builder_fn()
        pipeline = train_model(pipeline, X_train, y_train)
        trained_models[model_name] = pipeline

        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        all_metrics.append(metrics)

        # Per-model confusion matrix
        plot_confusion_matrix(metrics, output_dir=RESULTS_DIR)

    # 6. Comparison chart
    plot_model_comparison(all_metrics, output_dir=RESULTS_DIR)

    # 7. Save metrics to file
    save_metrics(all_metrics, output_dir=RESULTS_DIR)

    return trained_models, all_metrics


# ── CLI prediction ────────────────────────────────────────────────────────────

def predict_single(text: str, trained_models: dict) -> None:
    """
    Print the predicted component for a user-supplied bug report.

    Parameters
    ----------
    text : str
        Raw bug report text entered by the user.
    trained_models : dict
        {model_name: fitted_pipeline}
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        print("[predict] The input text is empty after cleaning. "
              "Please provide a more descriptive bug report.")
        return

    print(f"\n{'='*60}")
    print("  Bug Report Prediction")
    print(f"{'='*60}")
    print(f"  Input  : {text[:120]}")
    print(f"  Cleaned: {cleaned[:120]}")
    print()

    for model_name, pipeline in trained_models.items():
        prediction = pipeline.predict([cleaned])[0]
        # Probability / decision score (if available)
        try:
            proba = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_
            top_idx = proba.argsort()[::-1][:3]
            top_preds = [(classes[i], proba[i]) for i in top_idx]
            top_str = ", ".join(f"{c} ({p:.2f})" for c, p in top_preds)
            print(f"  [{model_name}]")
            print(f"    Predicted : {prediction}")
            print(f"    Top-3     : {top_str}")
        except AttributeError:
            # LinearSVC has no predict_proba
            print(f"  [{model_name}]")
            print(f"    Predicted : {prediction}")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bug Report Classification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the bug-report CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--predict",
        metavar="BUG_REPORT",
        default=None,
        help="Classify a single bug report text and exit.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"Directory for output files (default: {RESULTS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global RESULTS_DIR
    RESULTS_DIR = args.results_dir

    # Always run the training pipeline so models are available for prediction
    trained_models, all_metrics = run_pipeline(args.data)

    # If --predict was supplied, also run the CLI prediction
    if args.predict:
        predict_single(args.predict, trained_models)
    else:
        # Interactive mode: keep asking until the user quits
        print("\n" + "="*60)
        print("  Interactive Prediction Mode")
        print("  Type a bug report and press Enter to classify it.")
        print("  Type 'quit' or press Ctrl-C to exit.")
        print("="*60)
        while True:
            try:
                user_input = input("\nEnter bug report (or 'quit'): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[main] Exiting.")
                break
            if user_input.lower() in {"quit", "exit", "q"}:
                print("[main] Exiting.")
                break
            if user_input:
                predict_single(user_input, trained_models)


if __name__ == "__main__":
    main()
