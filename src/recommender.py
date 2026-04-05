import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load Data
# ----------------------------
movies = pd.read_csv("data/movies.csv")
credits = pd.read_csv("data/credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# ----------------------------
# Clean Columns
# ----------------------------
movies["movie_id"] = movies["movie_id_x"]
movies["cast"] = movies["cast_y"]
movies["crew"] = movies["crew_y"]

movies = movies[[
    "movie_id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew"
]].copy()

# Remove missing values
movies.dropna(inplace=True)

# ----------------------------
# Feature Engineering
# ----------------------------

# Convert cast (top 3 actors)
def convert_cast(text):
    L = []
    for i in ast.literal_eval(text)[:3]:
        L.append(i["name"])
    return L

movies["cast"] = movies["cast"].apply(convert_cast)

# Extract director from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i["job"] == "Director":
            L.append(i["name"])
    return L

movies["crew"] = movies["crew"].apply(fetch_director)

# Convert overview to list
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# Remove spaces in names
movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags
movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

# Final dataframe
new_df = movies[["movie_id", "title", "tags"]].copy()

# Convert tags to string
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

# ----------------------------
# Vectorization
# ----------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend(movie):
    movie = movie.lower()
    
    if movie not in new_df["title"].str.lower().values:
        return ["Movie not found. Please try another name."]
    
    movie_index = new_df[new_df["title"].str.lower() == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [new_df.iloc[i[0]].title for i in movies_list]


# ----------------------------
# Test Run
# ----------------------------
if __name__ == "__main__":
    print("Recommended Movies:")
    print(recommend("Avatar"))
