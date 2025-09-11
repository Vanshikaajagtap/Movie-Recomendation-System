import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display 

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

def clean_title(title):
    return re.sub("[^a-zA-Z0-9]","",title)
    
movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results

def find_similar_movies(movie_id):
    similar_movies = search(movies[movies["id"] == movie_id]["title"].iloc[0])
    return similar_movies[["title", "genres"]]

movie_input_name = widgets.Text(
    value="",
    description="Movie Title:",
    disabled=False
)

recs_list = widgets.Output()

def on_type(data):
    with recs_list:
        recs_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            if not results.empty:
                movie_id = results.iloc[0]["id"]
                similar_movies = find_similar_movies(movie_id)
                display(similar_movies)
            
movie_input_name.observe(on_type, names="value")

display(movie_input_name, recs_list)