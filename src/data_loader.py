"""
data_loader.py
--------------
Handles loading and initial validation of the Mozilla bug-report CSV dataset.
"""

import csv
import sys
import pandas as pd


# Increase CSV field-size limit to handle large embedded JSON cells
csv.field_size_limit(sys.maxsize)

# Columns we care about
TEXT_COLS = ["Summary", "Description"]
LABEL_COL = "Component"          # classification target


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the Mozilla bug-report CSV into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all columns present in the file.
    """
    print(f"[data_loader] Loading dataset from: {filepath}")

    # pandas cannot handle the huge embedded-JSON cells with its default
    # CSV engine, so we read with the stdlib csv module first.
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[data_loader] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def select_features_and_label(
    df: pd.DataFrame,
    text_cols: list[str] | None = None,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Keep only the text feature columns and the label column.
    Combines multiple text columns into a single 'text' column.

    Parameters
    ----------
    df : pd.DataFrame
    text_cols : list of str, optional
        Columns to concatenate as the input text.  Defaults to
        ['Summary', 'Description'].
    label_col : str
        Column to use as the classification label.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['text', 'label'].
    """
    if text_cols is None:
        text_cols = TEXT_COLS

    # Validate columns exist
    missing = [c for c in text_cols + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Combine text columns (fill NaN with empty string before joining)
    df = df.copy()
    df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.strip()
    df["label"] = df[label_col]

    return df[["text", "label"]]
