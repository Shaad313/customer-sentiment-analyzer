"""Load saved artifacts and predict sentiment for review text."""

from pathlib import Path

import joblib

try:
    from .preprocessing import clean_text
except ImportError:  # Supports `python src/predict.py`
    from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]


def load_model(model_dir: Path = ROOT / "models"):
    model_path = model_dir / "sentiment_model.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError("Trained model not found. Add labeled reviews to data/reviews.csv, then run: python -m src.train")
    return joblib.load(model_path), joblib.load(vectorizer_path)


def predict_sentiment(text: str, model_dir: Path = ROOT / "models") -> dict:
    model, vectorizer = load_model(model_dir)
    features = vectorizer.transform([clean_text(text)])
    label = str(model.predict(features)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features).max())
    return {"sentiment": label, "confidence": confidence}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Review to classify")
    args = parser.parse_args()
    result = predict_sentiment(args.text)
    print(f"{result['sentiment']}" + (f" ({result['confidence']:.1%})" if result["confidence"] is not None else ""))
