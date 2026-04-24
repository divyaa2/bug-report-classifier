# Bug Report Classification Tool

A machine-learning pipeline that classifies Mozilla bug reports into their
**Component** category (e.g. *Audio/Video: Playback*, *Graphics*, *DOM: Events*).

---

## Project Structure

```
.
├── main.py                  # Entry point & CLI
├── requirements.txt         # Python dependencies
├── sample_mozilla_core.csv  # Dataset (Mozilla Core bugs)
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # CSV loading & column selection
│   ├── preprocessor.py      # Text cleaning & stop-word removal
│   ├── features.py          # TF-IDF vectoriser factory
│   ├── models.py            # Model pipeline definitions
│   ├── trainer.py           # Train/test split & model fitting
│   └── evaluator.py         # Metrics, plots, JSON export
└── results/                 # Auto-created; stores plots & metrics JSON
```

---

## Dataset

The project uses **`sample_mozilla_core.csv`** — a snapshot of bug reports
exported from [Bugzilla Mozilla](https://bugzilla.mozilla.org/).

| Column      | Used as          |
|-------------|------------------|
| `Summary`   | Input text (part 1) |
| `Description` | Input text (part 2) |
| `Component` | Classification label |

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK stop-word data is downloaded automatically on first run.

---

## Usage

### Train all models and evaluate

```bash
python main.py
```

This will:
1. Load and preprocess the dataset.
2. Train three models: **Naive Bayes** (baseline), **Logistic Regression**, and **Linear SVM**.
3. Print accuracy, precision, recall, and F1-score for each model.
4. Save confusion-matrix heatmaps and a comparison bar chart to `results/`.
5. Save all metrics to a timestamped JSON file in `results/`.
6. Drop into an **interactive prediction prompt**.

### Classify a single bug report (non-interactive)

```bash
python main.py --predict "Firefox crashes when opening a new tab on Linux"
```

### Use a different dataset

```bash
python main.py --data path/to/other_bugs.csv
```

### Change the output directory

```bash
python main.py --results-dir my_output
```

---

## Models

| Model | Description |
|-------|-------------|
| **Naive Bayes** (baseline) | Multinomial NB with Laplace smoothing |
| **Logistic Regression** | L2-regularised, balanced class weights |
| **Linear SVM** | LinearSVC, balanced class weights |

All models use a **TF-IDF** feature extractor (unigrams + bigrams, top 5 000 features).

---

## Evaluation Metrics

- **Accuracy** — fraction of correctly classified reports
- **Precision / Recall / F1** — weighted averages across all classes
- **Confusion matrix** — per-class breakdown (saved as PNG)
- **Comparison chart** — side-by-side bar chart of all four metrics

---

## Output Files (`results/`)

| File | Description |
|------|-------------|
| `confusion_matrix_naive_bayes_baseline.png` | Confusion matrix for Naive Bayes |
| `confusion_matrix_logistic_regression.png` | Confusion matrix for Logistic Regression |
| `confusion_matrix_linear_svm.png` | Confusion matrix for Linear SVM |
| `model_comparison.png` | Bar chart comparing all models |
| `metrics_<timestamp>.json` | All metrics in machine-readable JSON |

---

## Notes

- The dataset contains ~99 labelled bug reports.  With such a small sample,
  accuracy figures should be interpreted with caution.
- Classes with only one sample are automatically excluded (cannot stratify-split).
- The interactive prompt accepts free-form text; type `quit` to exit.
