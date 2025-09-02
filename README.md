Project: Personalized Book Suggestion Based on Life Events  

## 🎯 Objective  
Recommend books based on a user’s **mood** or **life situation**. This project uses **NLP + Sentiment Analysis** to match book summaries with emotions, enabling **bibliotherapy** (healing through reading).  

---

## 🛠️ Tech Stack  
- **Python**  
- **Scikit-learn** (TF-IDF Vectorization, Cosine Similarity)  
- **NLTK + VADER** (Sentiment Analysis & Synonym Expansion)  
- **Pandas** (Dataset handling)  
- **Gradio** (Interactive App UI in Colab)  

---

## 🧮 Algorithms & Workflow  
1. **User Input** → User types mood/situation (e.g., *“I feel lonely and stressed”*)  
2. **Emotion Detection** → Sentiment analysis using VADER + optional mood dropdown  
3. **Text Expansion** → Synonym expansion with WordNet  
4. **Cosine Similarity** → Match user’s mood to **book summaries** in the dataset  
5. **Recommendation** → Suggest relevant books with online reading links  

---

## 📊 Dataset  
- **Source**: [Project Gutenberg Metadata (50k books)](https://www.gutenberg.org/)  
- **Fields Used**:  
  - Title  
  - Authors  
  - Subjects  
  - Bookshelves  
  - Online Read Link  

---

## 💡 Use Case: Bibliotherapy  
- Helps readers **cope with emotions** by suggesting books that align with their current **mental state**  
- Unique angle: **Sentiment-driven book discovery** instead of traditional genre-based recommendation  

---

## 🚀 How to Run (Google Colab)  
1. Upload `gutenberg_metadata.csv` dataset  
2. Run the provided notebook cells  
3. Launch Gradio app with:  

```python
demo.launch(share=True)

Features of the App

Free text input: “I am stressed about exams”

Dropdown mood selector: Happy, Sad, Angry, Lonely, Anxious, etc.

Adjustable number of book recommendations

Clickable “Read Online” links
