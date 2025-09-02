import pandas as pd
import numpy as np
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet

# Download NLTK resources if not already
nltk.download('vader_lexicon')
nltk.download('wordnet')

# ---------------------------
# STEP 1: Load Dataset
# ---------------------------
df = pd.read_csv("gutenberg_metadata.csv")

# Preprocess summaries
df['Subjects'] = df['Subjects'].fillna("")
df['Bookshelves'] = df['Bookshelves'].fillna("")
df['summary'] = df['Subjects'] + " " + df['Bookshelves']

# ---------------------------
# STEP 2: TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['summary'])

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# ---------------------------
# STEP 3: Mood Mapping
# ---------------------------
mood_map = {
    "Happy": ["joy", "cheerful", "bright", "smile"],
    "Sad": ["grief", "sorrow", "depression", "tears"],
    "Angry": ["rage", "fury", "violence", "hate"],
    "Frustrated": ["stress", "failure", "struggle"],
    "Lonely": ["isolation", "solitude", "alone"],
    "Motivated": ["success", "ambition", "drive", "focus"],
    "Relaxed": ["calm", "peace", "meditation"],
    "Anxious": ["fear", "uncertainty", "panic"],
    "Excited": ["adventure", "fun", "joy", "surprise"],
    "Stressed": ["pressure", "tension", "workload"],
}

# ---------------------------
# STEP 4: Expand text with synonyms
# ---------------------------
def expand_text(text):
    words = text.split()
    expanded = []
    for w in words:
        syns = wordnet.synsets(w)
        if syns:
            expanded.append(syns[0].lemmas()[0].name())
    return text + " " + " ".join(expanded)

# ---------------------------
# STEP 5: Recommender Function
# ---------------------------
def recommend_books(user_input, mood_choice="None", top_n=5):
    query = user_input
    
    if mood_choice and mood_choice != "None":
        query += " " + " ".join(mood_map[mood_choice])
    
    query = expand_text(query)
    
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    indices = cosine_sim.argsort()[-top_n:][::-1]
    results = df.iloc[indices][["Title", "Authors", "Subjects", "Read online (web)"]]
    return results

# ---------------------------
# STEP 6: Gradio App UI
# ---------------------------
def gradio_recommender(user_input, mood_choice, top_n):
    results = recommend_books(user_input, mood_choice, top_n)
    output = ""
    for i, row in results.iterrows():
        output += f"📖 **{row['Title']}** by {row['Authors']}\n"
        output += f"   📂 {row['Subjects']}\n"
        output += f"   🔗 [Read Online]({row['Read online (web)']})\n\n"
    return output

mood_options = ["None"] + list(mood_map.keys())

with gr.Blocks() as demo:
    gr.Markdown("# 📚 Personalized Book Recommender\n### Mood + Emotion Driven Bibliotherapy")
    
    user_input = gr.Textbox(label="Describe your mood / situation", placeholder="e.g. I feel tired and anxious today")
    mood_choice = gr.Dropdown(mood_options, value="None", label="Or pick a mood directly")
    top_n = gr.Slider(3, 10, value=5, step=1, label="Number of books to recommend")
    
    recommend_btn = gr.Button("Recommend Books")
    output_box = gr.Markdown()
    
    recommend_btn.click(fn=gradio_recommender, inputs=[user_input, mood_choice, top_n], outputs=output_box)

if __name__ == "__main__":
    demo.launch()
