import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# Recommendation function
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]

        movies_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(movies.iloc[i[0]].title)

        return recommended_movies
    except:
        return ["Movie not found 😢"]

# UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommender System")

st.write("Find similar movies instantly!")

# Dropdown (better than text input)
movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

# Button
if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies 🎯")
    for movie in recommendations:
        st.write(movie)
