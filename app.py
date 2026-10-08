"""Streamlit interface for the review sentiment classifier."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Customer Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Customer Sentiment Analyzer")
st.write("Classify customer reviews with a TF-IDF and Logistic Regression model.")

try:
    model, vectorizer = load_model(ROOT / "models")
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.stop()

tab_single, tab_batch = st.tabs(["Single review", "Batch CSV"])
with tab_single:
    review = st.text_area("Enter a customer review", placeholder="The product arrived quickly and works great!", height=140)
    if st.button("Analyze sentiment", type="primary", disabled=not review.strip()):
        features = vectorizer.transform([clean_text(review)])
        sentiment = str(model.predict(features)[0])
        confidence = float(model.predict_proba(features).max()) if hasattr(model, "predict_proba") else None
        st.metric("Predicted sentiment", sentiment.title())
        if confidence is not None:
            st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

with tab_batch:
    st.caption("Upload a CSV with a column named `text`, `review`, or `content`.")
    upload = st.file_uploader("Choose a CSV", type="csv")
    if upload is not None:
        frame = pd.read_csv(upload)
        text_col = next((c for c in frame.columns if str(c).strip().lower() in {"text", "review", "content", "reviews", "review_text"}), None)
        if text_col is None:
            st.error("No review text column found. Expected `text`, `review`, or `content`.")
        else:
            cleaned = frame[text_col].fillna("").map(clean_text)
            frame["predicted_sentiment"] = model.predict(vectorizer.transform(cleaned)).astype(str)
            if hasattr(model, "predict_proba"):
                frame["confidence"] = model.predict_proba(vectorizer.transform(cleaned)).max(axis=1).round(4)
            st.dataframe(frame, use_container_width=True)
            st.download_button("Download predictions", frame.to_csv(index=False).encode("utf-8"), "sentiment_predictions.csv", "text/csv")
