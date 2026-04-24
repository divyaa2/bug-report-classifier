"""
features.py
-----------
TF-IDF feature extraction using scikit-learn.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with sensible defaults.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple
        Range of n-gram sizes to include.
    sublinear_tf : bool
        Apply sublinear TF scaling (log(1 + tf)).

    Returns
    -------
    TfidfVectorizer (unfitted)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z][a-z0-9]+\b",
    )
