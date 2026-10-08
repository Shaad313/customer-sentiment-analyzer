"""Train and save a TF-IDF + logistic regression sentiment model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .preprocessing import clean_text
except ImportError:  # Supports `python src/train.py`
    from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def load_reviews(csv_path: Path) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(csv_path)
    text_col = _find_column(frame, ("text", "review", "content", "reviews", "review_text"))
    label_col = _find_column(frame, ("sentiment", "label", "class", "rating", "score"))
    if text_col is None or label_col is None:
        raise ValueError("CSV needs a review text column (text/review) and a label column (sentiment/label).")

    data = frame[[text_col, label_col]].dropna().copy()
    data[text_col] = data[text_col].map(clean_text)
    data[label_col] = data[label_col].astype(str).str.strip().str.lower()
    data = data[data[text_col].str.len() > 0]

    # Common Amazon star ratings: 1–2 negative, 3 neutral, 4–5 positive.
    rating_map = {"1": "negative", "2": "negative", "3": "neutral", "4": "positive", "5": "positive"}
    data[label_col] = data[label_col].replace(rating_map)
    data = data[data[label_col].isin(("positive", "negative", "neutral"))]
    if data[label_col].nunique() < 2:
        raise ValueError("At least two sentiment classes are required. Labels may be positive/negative/neutral or ratings 1–5.")
    return data[text_col].tolist(), data[label_col].tolist()


def default_data_path() -> Path:
    """Use the local Kaggle sample when present, otherwise the tracked demo CSV."""
    kaggle_sample = ROOT / "data" / "reviews.csv"
    return kaggle_sample if kaggle_sample.exists() else ROOT / "data" / "sample_reviews.csv"


def train(csv_path: Path | None = None, output_dir: Path = ROOT / "models") -> dict:
    csv_path = csv_path or default_data_path()
    texts, labels = load_reviews(csv_path)
    counts = pd.Series(labels).value_counts()
    can_split = len(texts) >= 6 and counts.min() >= 2
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    metrics: dict = {"samples": len(texts), "classes": sorted(set(labels))}
    if can_split:
        x_train, x_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        evaluation_model = Pipeline(model.steps)
        evaluation_model.fit(x_train, y_train)
        predictions = evaluation_model.predict(x_test)
        metrics["accuracy"] = accuracy_score(y_test, predictions)
        metrics["report"] = classification_report(y_test, predictions, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(
            y_test, predictions, labels=sorted(set(labels))
        ).tolist()
    model.fit(texts, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.named_steps["classifier"], output_dir / "sentiment_model.pkl")
    joblib.dump(model.named_steps["tfidf"], output_dir / "tfidf_vectorizer.pkl")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    results = train(args.data, args.output)
    print(f"Trained on {results['samples']} reviews; classes: {', '.join(results['classes'])}")
    if "accuracy" in results:
        print(f"Holdout accuracy: {results['accuracy']:.3f}\n{results['report']}")
        print(f"Confusion matrix (labels {results['classes']}): {results['confusion_matrix']}")
    else:
        print("Dataset is too small for a reliable holdout evaluation; add more labeled reviews.")
    print(f"Saved model artifacts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
