# 🎬 Movie Recommendation System (NLP & Machine Learning)

## 🚀 Overview
This project is an end-to-end, content-based movie recommendation system designed to suggest relevant films based on user preferences. By leveraging Natural Language Processing (NLP) workflows—including text preprocessing, tokenization, stemming, and vectorization—the system maps movie metadata to a multi-dimensional feature space. It computes cosine similarity scores across thousands of films to deliver real-time, accurate recommendations via a clean, interactive web dashboard.

---

## 💡 Key Features
- **Data Engineering & Preprocessing:** Cleans and merges complex movie metadata (genres, keywords, cast, crew, and taglines) into structured text corpora.
- **Natural Language Processing (NLP):** Implements text normalization (Stemming) and Bag-of-Words vectorization (`CountVectorizer`) to capture text patterns.
- **Similarity Computation:** Utilizes Cosine Similarity to mathematically calculate spatial proximity between movie vectors for recommendations.
- **Interactive UI Deployment:** Features an intuitive Streamlit dashboard allowing users to search for movies, see recommendations, and explore poster details dynamically.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 🐍
- **Machine Learning & NLP:** Scikit-learn (`CountVectorizer`, `cosine_similarity`), NLTK (Stemming)
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Web App Framework:** Streamlit

---

## 📂 Project Structure
```
movie-recommendation-system/
│
├── app.py                 # Streamlit main application and user interface script
├── movie_dict.pkl         # Serialized movie dataset containing preprocessed tokens
├── similarity.pkl         # Precomputed Cosine Similarity matrix for instant retrieval
├── TMDB_5000_movies.csv   # Primary raw dataset containing comprehensive movie details
├── requirements.txt       # Project installation dependencies
└── README.md              # Detailed project documentation
