"""
preprocessor.py
---------------
Text cleaning, tokenisation, and missing-value handling.

Steps applied to each document:
  1. Lowercase
  2. Remove URLs
  3. Remove punctuation / special characters
  4. Tokenise (whitespace split)
  5. Remove NLTK English stop-words
  6. Rejoin tokens into a clean string
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords

# Download NLTK data on first use (silent if already present)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))

# Regex patterns compiled once for speed
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single string.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned, stop-word-free text.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = _URL_RE.sub(" ", text)

    # 3. Remove punctuation / special characters
    text = _PUNCT_RE.sub(" ", text)

    # 4. Tokenise
    tokens = _WHITESPACE_RE.split(text.strip())

    # 5. Remove stop-words and very short tokens (length < 2)
    tokens = [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]

    # 6. Rejoin
    return " ".join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'text' column and drop rows with missing labels or empty text.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'text' and 'label'.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns ['text', 'label'].
    """
    df = df.copy()

    # Handle missing values
    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].fillna("")

    # Drop rows where label is empty / whitespace
    df = df[df["label"].str.strip() != ""]

    # Drop rows where label appears only once (can't stratify-split)
    label_counts = df["label"].value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    dropped = (~df["label"].isin(valid_labels)).sum()
    if dropped:
        print(
            f"[preprocessor] Dropping {dropped} rows with singleton labels "
            f"(cannot stratify-split)."
        )
    df = df[df["label"].isin(valid_labels)].copy()

    # Apply text cleaning
    print("[preprocessor] Cleaning text …")
    df["text"] = df["text"].apply(clean_text)

    # Drop rows that became empty after cleaning
    empty_mask = df["text"].str.strip() == ""
    if empty_mask.sum():
        print(f"[preprocessor] Dropping {empty_mask.sum()} rows with empty text after cleaning.")
    df = df[~empty_mask].copy()

    df.reset_index(drop=True, inplace=True)
    print(f"[preprocessor] {len(df)} rows remain after preprocessing.")
    return df
