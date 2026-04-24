"""
models.py
---------
Defines the three classifiers used in this project:
  - Multinomial Naive Bayes  (baseline)
  - Logistic Regression      (improved)
  - Linear SVM               (optional)
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features import build_tfidf_vectorizer


def build_naive_bayes_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Baseline model: TF-IDF + Multinomial Naive Bayes.

    Parameters
    ----------
    alpha : float
        Laplace smoothing parameter.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )


def build_logistic_regression_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Improved model: TF-IDF + Logistic Regression.

    Parameters
    ----------
    C : float
        Inverse regularisation strength.
    max_iter : int
        Maximum number of solver iterations.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_linear_svm_pipeline(C: float = 1.0) -> Pipeline:
    """
    Optional model: TF-IDF + Linear SVM.

    Parameters
    ----------
    C : float
        Regularisation parameter.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            (
                "clf",
                LinearSVC(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# Registry: name → builder function (no-arg callable)
MODEL_REGISTRY: dict[str, callable] = {
    "Naive Bayes (baseline)": build_naive_bayes_pipeline,
    "Logistic Regression": build_logistic_regression_pipeline,
    "Linear SVM": build_linear_svm_pipeline,
}
