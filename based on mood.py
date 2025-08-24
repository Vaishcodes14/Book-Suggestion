import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
# Your dataset must have columns: title, authors, Summary
books_df = pd.read_csv("books_dataset.csv")

if "Summary" not in books_df.columns:
    books_df["Summary"] = "No description available."

# -------------------------------
# Step 2: Mood → Emotion Mapping
# -------------------------------
mood_map = {
    "happy": "celebration joy positivity success gratitude",
    "sad": "comfort hope healing motivation overcoming struggles",
    "anxious": "calm mindfulness peace relaxation stress relief",
    "stressed": "balance productivity resilience peace work-life",
    "motivated": "growth achievement inspiration leadership success",
    "lonely": "friendship connection belonging relationships empathy",
    "angry": "patience forgiveness calmness letting go peace",
    "lost": "self-discovery purpose meaning direction life journey"
}

# -------------------------------
# Step 3: Load Transformer Model
# -------------------------------
print("🔄 Loading transformer model... (this may take a minute)")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast, good quality

# -------------------------------
# Step 4: Encode Book Summaries
# -------------------------------
print("🔄 Encoding book summaries...")
book_embeddings = model.encode(books_df["Summary"].fillna(""), show_progress_bar=True)

# -------------------------------
# Step 5: Recommendation Function
# -------------------------------
def recommend_books(user_mood, top_n=5):
    if user_mood not in mood_map:
        return f"❌ Mood '{user_mood}' not recognized. Try: {list(mood_map.keys())}"

    # Convert mood into embedding
    mood_text = mood_map[user_mood]
    mood_embedding = model.encode([mood_text])

    # Compute cosine similarity
    similarities = cosine_similarity(mood_embedding, book_embeddings).flatten()

    # Get top N book indices
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Return recommendations
    recommendations = books_df.iloc[top_indices][["title", "authors", "Summary"]]
    return recommendations

# -------------------------------
# Step 6: Test
# -------------------------------
user_mood = "anxious"  # try "happy", "sad", etc.
print(f"\n📖 Top Book Recommendations for mood: {user_mood}\n")
print(recommend_books(user_mood, top_n=5))
