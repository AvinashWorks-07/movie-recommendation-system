# 🎬 Movie Recommendation System | NLP & Machine Learning

## 🚀 Project Overview
Built a content-based movie recommendation system using Natural Language Processing (NLP) and Machine Learning techniques to recommend similar movies based on their metadata. The system processes movie attributes, converts textual information into numerical feature vectors, and uses cosine similarity to identify relevant recommendations. An interactive Streamlit application provides a user-friendly interface for discovering movies.

## 💡 Key Features

- **Data Preprocessing & Feature Engineering:** Cleaned and combined movie metadata, including genres, keywords, cast, crew, and taglines, into a unified text representation for similarity analysis.
- **Natural Language Processing:** Applied text preprocessing and stemming to normalize movie-related text and reduce variations in word forms.
- **Text Vectorization:** Used Scikit-learn's `CountVectorizer` to transform textual movie metadata into numerical feature vectors using a Bag-of-Words approach.
- **Similarity-Based Recommendations:** Implemented cosine similarity to measure the similarity between movie vectors and retrieve relevant movie recommendations.
- **Interactive Web Application:** Developed a Streamlit interface that allows users to select a movie and discover similar titles, with movie poster details where integrated.
- **Efficient Recommendation Retrieval:** Precomputed and serialized movie data and similarity scores using Pickle for faster recommendation lookup.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-learn, NLTK
- **Algorithms & Techniques:** Bag-of-Words, Stemming, Cosine Similarity
- **Web Application:** Streamlit
- **Dataset:** TMDB 5000 Movies Dataset

## 📂 Project Structure

```text
movie-recommendation-system/
│
├── app.py                   # Streamlit application and recommendation interface
├── movie_dict.pkl           # Serialized movie metadata
├── similarity.pkl           # Precomputed movie similarity scores
├── TMDB_5000_movies.csv     # Raw movie metadata dataset
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation and setup instructions
```

## 🎯 Project Objective
Apply NLP and similarity-based Machine Learning techniques to build a personalized movie discovery experience, demonstrating practical skills in text preprocessing, feature engineering, vectorization, similarity computation, and web application development.

## 🔮 Potential Future Enhancements
- Integrate TMDB API for dynamic movie posters and additional movie information.
- Evaluate recommendation quality using Precision@K, Recall@K, or other suitable ranking metrics.
- Optimize similarity computation and storage for scalability.
- Add genre filters, user ratings, and personalized recommendation options.
- Deploy the application publicly and document its usage with screenshots and a live demo link.
