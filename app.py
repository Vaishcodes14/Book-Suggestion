import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("books_50k.csv")
    return df

books_df = load_data()

# ----------------------------
# Preprocess: Combined text field
# ----------------------------
books_df["combined"] = (
    books_df["Title"].fillna("") + " " +
    books_df["Subjects"].fillna("") + " " +
    books_df["Bookshelves"].fillna("") + " " +
    books_df["Authors"].fillna("")
)

# ----------------------------
# NLP Model Setup
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books_df["combined"])

# Load HuggingFace Emotion Classifier
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", 
                    model="cardiffnlp/twitter-roberta-base-emotion", 
                    return_all_scores=True)

emotion_model = load_emotion_model()

# ----------------------------
# Emotion → Bookshelves Mapping
# ----------------------------
emotion_to_bookshelves = {
    "joy": ["Fiction", "Adventure", "Fantasy", "Poetry"],
    "sadness": ["Philosophy", "Poetry", "Biography"],
    "anger": ["Politics", "History", "Philosophy"],
    "fear": ["Religion", "Philosophy", "Science"],
    "surprise": ["Science", "Adventure", "Fantasy"]
}

# ----------------------------
# Helper Function: Detect Emotion
# ----------------------------
def detect_emotion(text):
    results = emotion_model(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[0]["label"], results[0]["score"]

# ----------------------------
# Helper Function: Get Recommendations
# ----------------------------
def get_recommendations(user_input, top_n=5):
    detected_emotion, confidence = detect_emotion(user_input)
    mapped_bookshelves = emotion_to_bookshelves.get(detected_emotion.lower(), [])
    
    # Filter books by mapped shelves first
    if mapped_bookshelves:
        filtered_books = books_df[
            books_df["Bookshelves"].isin(mapped_bookshelves)
        ]
    else:
        filtered_books = books_df
    
    if filtered_books.empty:
        filtered_books = books_df

    # Vectorize input
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # Get top indices within filtered set
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    return detected_emotion, confidence, books_df.iloc[top_indices][
        ["Title", "Authors", "Language", "Bookshelves", "Subjects", "Rights"]
    ]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📚 Emotion-Aware Bibliotherapy Recommender")
st.write("Books chosen based on your **mood, life events, and emotions** ❤️📖")

# User input
user_input = st.text_input("How are you feeling or what’s happening in your life?")

if st.button("Get Book Suggestions"):
    if user_input.strip():
        detected_emotion, confidence, results = get_recommendations(user_input, top_n=5)
        st.subheader(f"😊 Detected Emotion: **{detected_emotion.capitalize()}** (confidence: {confidence:.2f})")
        st.subheader("📖 Recommended Books for You")
        for i, row in results.iterrows():
            st.markdown(f"""
            **{row['Title']}**  
            👤 *{row['Authors']}*  
            🌍 Language: {row['Language']}  
            📂 Bookshelves: {row['Bookshelves']}  
            🏷 Subjects: {row['Subjects']}  
            ⚖ Rights: {row['Rights']}  
            """)
    else:
        st.warning("Please describe your mood or life event.")

# Optional: Surprise Me
if st.button("🎲 Surprise Me"):
    rand_row = books_df.sample(1).iloc[0]
    st.markdown(f"""
    **{rand_row['Title']}**  
    👤 *{rand_row['Authors']}*  
    🌍 Language: {rand_row['Language']}  
    📂 Bookshelves: {rand_row['Bookshelves']}  
    🏷 Subjects: {rand_row['Subjects']}  
    ⚖ Rights: {rand_row['Rights']}  
    """)
