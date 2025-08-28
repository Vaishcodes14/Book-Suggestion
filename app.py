# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Tuple
from dataclasses import dataclass
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to import transformers (for better emotion detection). If unavailable, fallback.
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# ---- Sidebar / Page config ----
st.set_page_config(page_title="Bibliotherapy — Emotion-driven Book Recs", layout="wide",
                   initial_sidebar_state="expanded")
st.title("📚 Bibliotherapy — Emotion-driven Book Recommendations")
st.markdown(
    "Type a life event, feeling, or situation and get book suggestions matched by emotion and summary similarity."
)

# ---- Utilities: Text preprocessing ----
@nltk.cache
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        nltk.download("wordnet")

ensure_nltk()
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ---- Emotion detection ----
@st.cache_resource
def get_emotion_pipeline():
    if not HAS_TRANSFORMERS:
        return None
    try:
        # Recommended emotion model — HuggingFace will download when run first time.
        # Model choice is flexible; this is just an example that's common for emotion detection.
        pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        return pipe
    except Exception:
        return None

emotion_pipe = get_emotion_pipeline()

def detect_emotion_transformer(text: str) -> Tuple[str, dict]:
    """
    Returns (top_emotion, {emotion:score,...}) using transformer if available.
    """
    if emotion_pipe is None:
        return None, {}
    out = emotion_pipe(text)
    # For pipeline text-classification with no top_k, HuggingFace returns list of dicts
    # However depending on the pipeline configuration, output can differ; normalise defensively.
    scores = {}
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and 'label' in out[0]:
        # single dict
        for item in out:
            scores[item['label'].lower()] = float(item['score'])
    elif isinstance(out, dict):
        # sometimes returns dict of all classes
        for k, v in out.items():
            scores[k.lower()] = float(v)
    # choose top
    if scores:
        top = max(scores.items(), key=lambda x: x[1])[0]
        return top, scores
    return None, {}

# Fallback rule-based emotion mapping (simple)
EMOTION_KEYWORDS = {
    "sadness": ["sad", "heartbroken", "depressed", "lonely", "grief", "loss", "sorrow"],
    "joy": ["happy", "joy", "excited", "delighted", "ecstatic", "elated"],
    "anger": ["angry", "resentful", "furious", "mad", "irate"],
    "fear": ["anxious", "afraid", "scared", "fear", "panic", "worried"],
    "surprise": ["surprised", "shocked", "astonished"],
    "love": ["love", "cherish", "adore", "fond"],
    "trust": ["trust", "secure", "safe"],
    "anticipation": ["hopeful", "anticipate", "eager", "expect"]
}

def detect_emotion_fallback(text: str) -> Tuple[str, dict]:
    text_l = text.lower()
    counts = {}
    for e, kws in EMOTION_KEYWORDS.items():
        counts[e] = sum(text_l.count(k) for k in kws)
    # choose the emotion with highest count (if zero counts, fallback to neutral)
    top = max(counts.items(), key=lambda x: x[1])
    if top[1] == 0:
        return "neutral", counts
    return top[0], counts

def detect_emotion(text: str) -> Tuple[str, dict]:
    # try transformer first
    if emotion_pipe is not None:
        try:
            top, scores = detect_emotion_transformer(text)
            if top:
                return top, scores
        except Exception:
            pass
    # fallback
    return detect_emotion_fallback(text)

# ---- Data loading / sample data ----
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload a books CSV (cols: title, author, summary, genres optional)", type=["csv"])
use_sample = st.sidebar.checkbox("Use included sample dataset (small) if no upload", value=True)

SAMPLE_CSV = """title,author,summary,genres
The Alchemist,Paulo Coelho,"A shepherd's journey toward his dreams teaches him about the treasures of life and love.",Fiction;Fantasy
Tiny Beautiful Things,Cheryl Strayed,"A collection of advice and empathetic letters about loss, healing and love.",Non-fiction;Self-help
Man's Search for Meaning,Viktor Frankl,"A memoir and psychological exploration about finding meaning through suffering.",Psychology;Memoir
The Art of Happiness,Dalai Lama,"Conversations and stories about cultivating inner peace, compassion, and joy.",Self-help;Spirituality
Where the Crawdads Sing,Delia Owens,"A story of isolation, nature and resilience after abandonment and loss.",Fiction;Mystery
"""

@st.cache_data
def load_dataset_from_file(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

@st.cache_data
def load_sample_df() -> pd.DataFrame:
    from io import StringIO
    df = pd.read_csv(StringIO(SAMPLE_CSV))
    return df

if uploaded is not None:
    try:
        df = load_dataset_from_file(uploaded)
        st.sidebar.success(f"Loaded {len(df)} books from uploaded CSV.")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        df = load_sample_df()
        st.sidebar.info("Loaded sample dataset instead.")
else:
    if use_sample:
        df = load_sample_df()
        st.sidebar.info(f"Using sample dataset ({len(df)} rows).")
    else:
        st.sidebar.warning("No dataset selected. Upload a CSV or enable sample dataset.")
        df = pd.DataFrame(columns=["title", "author", "summary", "genres"])

# validate required columns
required_cols = {"title", "summary"}
if not required_cols.issubset(set(df.columns.str.lower())):
    # try to normalize column names (lowercase)
    df.columns = [c.lower() for c in df.columns]
if not required_cols.issubset(set(df.columns)):
    st.error("Dataset must include at least 'title' and 'summary' columns (case-insensitive).")
    st.stop()

# normalize column names
df = df.rename(columns={c: c.lower() for c in df.columns})
if "author" not in df.columns:
    df["author"] = "Unknown"
if "genres" not in df.columns:
    df["genres"] = ""

# ---- Preprocess summaries ----
@st.cache_data
def preprocess_summaries(summaries: List[str]) -> List[str]:
    return [clean_text(str(s)) for s in summaries]

with st.spinner("Preprocessing book summaries..."):
    df["summary_clean"] = preprocess_summaries(df["summary"].fillna(""))

# ---- Vectorizer + TF-IDF on summaries ----
@st.cache_data
def build_vectorizer_and_matrix(docs: List[str]):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    X = vectorizer.fit_transform(docs)
    # normalize rows for cosine to effectively be dot product
    X = normalize(X)
    return vectorizer, X

vectorizer, tfidf_matrix = build_vectorizer_and_matrix(df["summary_clean"].tolist())

# ---- Recommendation logic ----
def make_query_embedding(user_text: str, emotion_label: str, vectorizer: TfidfVectorizer):
    """
    Combine user text with emotion tag to bias results toward emotion-relevant books.
    Then vectorize with the same TF-IDF vectorizer.
    """
    # append explicit emotion phrase to steer matching
    combined = f"{user_text} emotion:{emotion_label}"
    combined_clean = clean_text(combined)
    q_vec = vectorizer.transform([combined_clean])
    q_vec = normalize(q_vec)
    return q_vec

def get_top_n_recommendations(q_vec, tfidf_matrix, df: pd.DataFrame, top_n=10):
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx = np.argsort(-sims)[:top_n]
    results = df.iloc[idx].copy()
    results["score"] = sims[idx]
    return results.reset_index(drop=True)

# ---- UI: Input ----
st.markdown("## Describe your life event / mood")
user_input = st.text_area("Write a short description (example: 'I just broke up and feel lost', 'Anxious before exams', 'Need motivation to start a new career')", height=120)

col1, col2 = st.columns([1,3])
with col1:
    st.write("### Settings")
    top_k = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, step=1)
    show_snippet = st.checkbox("Show summary snippet", value=True)
    include_genre = st.checkbox("Show genres", value=True)
    use_transformer_if_available = st.checkbox("Attempt transformer emotion detection (may download model)", value=True)
with col2:
    st.write("### Quick tips")
    st.markdown("- Be specific about the event and emotion (e.g., 'I'm grieving the loss of my pet' → better matches).")
    st.markdown("- If you uploaded a large dataset, first run sample queries to ensure quality.")

# ---- Process user input and show recommendations ----
if st.button("Recommend books"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter a life event, mood, or situation to continue.")
    else:
        with st.spinner("Detecting emotion & computing recommendations..."):
            # detect emotion
            top_emotion, emotion_scores = detect_emotion(user_input)
            if top_emotion is None:
                top_emotion = "neutral"
            # build query vector
            q_vec = make_query_embedding(user_input, top_emotion, vectorizer)
            recommendations = get_top_n_recommendations(q_vec, tfidf_matrix, df, top_n=top_k)

        st.success(f"Emotion detected: **{top_emotion}**")
        # show emotion breakdown if available
        if emotion_scores:
            st.write("Emotion scores / counts (debug):")
            st.write({k: round(float(v), 3) for k, v in emotion_scores.items()})

        # Display recommendations
        st.markdown("### Recommended books")
        for i, row in recommendations.iterrows():
            st.markdown(f"**{i+1}. {row['title']}** — _{row.get('author', 'Unknown')}_")
            st.write(f"**Similarity score:** {row['score']:.4f}")
            if include_genre and row.get("genres", ""):
                st.write(f"**Genres:** {row.get('genres')}")
            if show_snippet:
                snippet = (row["summary"][:400] + "...") if len(str(row["summary"])) > 400 else row["summary"]
                st.write(snippet)
            # small spacer
            st.markdown("---")

        # Optional: show a few summary terms that matched (top tfidf features)
        st.markdown("### Why these books? (top matching terms)")
        try:
            # get feature names and multiply q_vec * tfidf_matrix to find strong features (approx)
            feature_names = vectorizer.get_feature_names_out()
            # get nonzero indices of q_vec
            q_arr = q_vec.toarray().flatten()
            top_q_idx = np.argsort(-q_arr)[:10]
            top_features = [feature_names[idx] for idx in top_q_idx if q_arr[idx] > 0]
            if top_features:
                st.write(", ".join(top_features))
            else:
                st.write("No highly-weighted query terms (try writing a longer description).")
        except Exception:
            st.write("Couldn't extract matching terms (vectorizer internals unavailable).")

st.markdown("---")
st.caption("Built with TF-IDF + cosine similarity. For stronger semantic matching, switch to transformer / SBERT embeddings and cosine similarity (requires additional packages).")

# ---- Footer: dataset stats ----
st.sidebar.markdown("### Dataset stats")
st.sidebar.write(f"Books loaded: **{len(df):,}**")
vals = df['genres'].value_counts().head(10)
if not vals.empty:
    st.sidebar.markdown("Top genres (sample):")
    st.sidebar.write(vals.to_dict())
