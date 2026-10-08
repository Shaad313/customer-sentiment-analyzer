"""NLTK-based text normalization shared by training and prediction."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

ROOT = Path(__file__).resolve().parents[1]
NLTK_DATA = ROOT / ".nltk_data"
nltk.data.path.insert(0, str(NLTK_DATA))

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_NEGATIONS = {"no", "not", "nor", "never", "n't"}
_TOKENIZER = TreebankWordTokenizer()
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()


def _ensure_corpora() -> None:
    """Load or download the small English stopword and WordNet resources locally."""
    try:
        stopwords.words("english")
        _LEMMATIZER.lemmatize("reviews", pos="v")
    except LookupError:
        NLTK_DATA.mkdir(parents=True, exist_ok=True)
        for package in ("stopwords", "wordnet"):
            if not nltk.download(package, download_dir=str(NLTK_DATA), quiet=True):
                raise RuntimeError(
                    f"Could not download the NLTK '{package}' corpus. "
                    "Check your internet connection, then run preprocessing again."
                )
        # Ensure the requested corpus resources were actually installed.
        stopwords.words("english")
        _LEMMATIZER.lemmatize("reviews", pos="v")


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    _ensure_corpora()
    return frozenset(stopwords.words("english")) - _NEGATIONS


def clean_text(text: object) -> str:
    """Tokenize, remove non-sentiment stopwords, lemmatize, and stem a review."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _HTML_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = value.lower()

    tokens = _TOKENIZER.tokenize(value)
    stop_words = _english_stopwords()
    processed: list[str] = []
    for token in tokens:
        if not any(character.isalnum() for character in token):
            continue
        if token in stop_words:
            continue
        lemma = _LEMMATIZER.lemmatize(token)
        lemma = _LEMMATIZER.lemmatize(lemma, pos="v")
        processed.append(_STEMMER.stem(lemma))
    return " ".join(processed)
