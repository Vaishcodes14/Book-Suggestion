import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

DATA_PATH = "books_50k.csv"
PARQUET_PATH = "books_50k.parquet"
VEC_PATH = "vectorizer.pkl"
TFIDF_PATH = "tfidf_matrix.pkl"

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    # If parquet already exists, load that (faster)
    if os.path.exists(PARQUET_PATH):
        return pd.read_parquet(PARQUET_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        df.to_parquet(PARQUET_PATH, index=False)
        return df

@st.cache_resource
def load_tfidf(df):
    if os.path.exists(VEC_PATH) and os.path.exists(TFIDF_PATH):
        vectorizer = joblib.load(VEC_PATH)
        tfidf_matrix = joblib.load(TFIDF_PATH)
    else:
        df["text"] = df["Title"].astype(str) + " " + df["Subjects"].astype(str)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        tfidf_matrix = vectorizer.fit_transform(df["text"])
        joblib.dump(vectorizer, VEC_PATH)
        joblib.dump(tfidf_matrix, TFIDF_PATH)
    return vectorizer, tfidf_matrix

@st.cache_resource
def load_model():
    return pipeline("text-classification", 
                    model="bhadresh-savani/distilbert-base-uncased-emotion")

# ---------- Main App ----------
st.title("📚 Sentiment-Driven Book Recommendation")
st.write("Bibliotherapy: Book suggestions based on your mood or life events.")

# Load dataset & models
df = load_data()
vectorizer, tfidf_matrix = load_tfidf(df)
emotion_model = load_model()

# User input
user_event = st.text_area("Describe your mood or situation:", 
                          "I feel stressed and need motivation.")

if st.button("Find Books"):
    with st.spinner("Analyzing your emotion..."):
        emotions = emotion_model(user_event, top_k=1)
        emotion = emotions[0]["label"]

    st.subheader(f"Detected Emotion: {emotion}")

    # Compute similarity
    user_vector = vectorizer.transform([user_event])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-5:][::-1]

    recommendations = df.iloc[top_indices][["Title", "Authors", "Subjects", "Bookshelves"]]

    st.subheader("📖 Recommended Books:")
    for _, row in recommendations.iterrows():
        st.markdown(f"""
        - **{row['Title']}**  
          👤 Author: {row['Authors']}  
          🏷️ Subject: {row['Subjects']}  
          📚 Shelf: {row['Bookshelves']}
        """)
