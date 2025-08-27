import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    return df

# -------------------------
# Load embedding model
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Generate embeddings for all book summaries
# -------------------------
@st.cache_data
def embed_books(df, model):
    embeddings = model.encode(df["summary"].tolist(), show_progress_bar=True)
    return embeddings

# -------------------------
# Recommender function
# -------------------------
def recommend_books(user_mood, df, model, embeddings, top_n=5):
    # Convert mood into embedding
    mood_embedding = model.encode([user_mood])
    
    # Compute cosine similarity
    similarities = cosine_similarity(mood_embedding, embeddings)[0]
    
    # Get top matches
    top_idx = np.argsort(similarities)[::-1][:top_n]
    return df.iloc[top_idx][["title", "author", "summary"]]

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 Sentiment-Driven Book Recommender")
st.write("Discover books based on your **current mood** — powered by NLP & Transformers.")

# Mood options
mood = st.selectbox(
    "How are you feeling today?",
    ["Happy", "Sad", "Anxious", "Stressed", "Motivated", "Lonely", "Relaxed", "Curious"]
)

# Load resources
df = load_data()
model = load_model()
embeddings = embed_books(df, model)

if st.button("Recommend Books"):
    results = recommend_books(mood, df, model, embeddings, top_n=5)
    st.subheader(f"📖 Top Book Picks for '{mood}' Mood:")
    for _, row in results.iterrows():
        st.markdown(f"**{row['title']}** by *{row['author']}*")
        st.write(row['summary'])
        st.write("---")
