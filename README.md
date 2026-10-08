# Customer Sentiment Analyzer

A Python project that classifies customer reviews using TF-IDF features and Logistic Regression. The Streamlit app supports single reviews and batch CSV predictions. The included Kaggle training data has positive and negative labels.

## Open and run in VS Code

1. Open this folder in VS Code and install the recommended Python and Jupyter extensions.
2. Create and activate a virtual environment in the VS Code terminal:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.
3. Select the `.venv` interpreter in VS Code (`Ctrl+Shift+P` → **Python: Select Interpreter**).
4. Train the model. This uses your local 10,000-row Kaggle sample if present; a fresh clone falls back to the small tracked demo CSV:

   ```powershell
   python -m src.train
   ```

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

   Or select **Streamlit: Sentiment Analyzer** in VS Code's Run and Debug panel.

`data/reviews.csv` is a balanced 10,000-review sample drawn from the [Amazon Reviews for Sentiment Analysis Kaggle dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). Its two classes are negative and positive. The local Kaggle CSV and source archive under `data/raw/` are excluded from Git because Kaggle lists the dataset license as unknown. Check the dataset page's current terms before sharing its data. `data/sample_reviews.csv` contains 30 illustrative examples so a fresh clone can still train and run a demo.

The helper `src/prepare_kaggle_data.py` can rebuild the balanced sample from `data/raw/train.ft.txt.bz2`:

```powershell
python -m src.prepare_kaggle_data
```

Preprocessing uses NLTK Treebank tokenization, English stopword removal (while retaining negations), WordNet lemmatization, and Porter stemming. The small `stopwords` and `wordnet` corpora are downloaded into `.nltk_data/` on first use. Training accepts text columns named `text`, `review`, `content`, `reviews`, or `review_text`, and label columns named `sentiment`, `label`, `class`, `rating`, or `score`. Text labels can be positive/negative/neutral; numeric ratings 1–5 map to negative (1–2), neutral (3), and positive (4–5). Other sentiment schemes should be mapped before training.

Training saves `models/sentiment_model.pkl` and `models/tfidf_vectorizer.pkl`. The initial evaluation is a stratified holdout report; for research-quality results, evaluate on a larger, representative, untouched test set. Confidence is the classifier's predicted probability and is not a guarantee of correctness.

## Command line prediction

```powershell
python -m src.predict "The quality is great and delivery was fast"
```

## Project layout

```text
data/reviews.csv                 Local 10,000-row Kaggle sample (ignored by Git)
data/sample_reviews.csv          Tiny illustrative CSV included in Git
data/raw/                        Kaggle source archive and BZ2 file (ignored by Git)
notebooks/sentiment_analysis.ipynb  Exploration notebook
models/                          Saved model artifacts (created by training)
src/preprocessing.py             Text normalization
src/train.py                      Training and evaluation
src/prepare_kaggle_data.py        Build the balanced sample from Kaggle's BZ2 file
src/predict.py                    Reusable prediction function / CLI
app.py                            Streamlit single and batch interface
```
