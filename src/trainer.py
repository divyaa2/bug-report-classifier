"""
trainer.py
----------
Splits data and trains each model pipeline.

For small datasets with many classes the minimum viable test size is
  ceil(n_classes / n_samples)  — we compute this automatically.
"""

import math
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratified train-test split that adapts to small / imbalanced datasets.

    Stratification requires at least one sample per class in the test set.
    When the requested test_size is too small for that, we bump it up to the
    minimum that satisfies the constraint.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'text' and 'label' columns.
    test_size : float
        Desired fraction of data for testing (may be increased automatically).
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.Series
    """
    X = df["text"]
    y = df["label"]

    n_samples = len(df)
    n_classes = y.nunique()

    # Minimum test size so every class can appear at least once
    min_test_size = math.ceil(n_classes / n_samples * n_samples) / n_samples
    # Add a small buffer so sklearn's rounding doesn't bite us
    min_test_size = min(min_test_size + 0.05, 0.5)

    effective_test_size = max(test_size, min_test_size)
    if effective_test_size > test_size:
        print(
            f"[trainer] Dataset has {n_samples} samples and {n_classes} classes. "
            f"Increasing test_size from {test_size:.2f} to {effective_test_size:.2f} "
            f"to satisfy stratification constraints."
        )

    # Check whether every class has ≥ 2 samples (needed for stratify)
    label_counts = y.value_counts()
    can_stratify = (label_counts >= 2).all()

    if can_stratify:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=effective_test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError as exc:
            # Last-resort fallback: non-stratified split
            print(f"[trainer] Stratified split failed ({exc}); "
                  "falling back to non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=effective_test_size,
                random_state=random_state,
            )
    else:
        print(
            "[trainer] Some classes have < 2 samples; "
            "using non-stratified split."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=effective_test_size,
            random_state=random_state,
        )

    print(f"[trainer] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(pipeline, X_train: pd.Series, y_train: pd.Series):
    """
    Fit a pipeline on the training data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X_train : pd.Series
    y_train : pd.Series

    Returns
    -------
    Fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline
