# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)
CORS(app)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize variables
movies = pd.DataFrame()
credits = pd.DataFrame()
vectorizer = None
tfidf = None

def load_data():
    global movies, credits, vectorizer, tfidf
    try:
        # Load from public URL instead of local file
        movies_url = "https://raw.githubusercontent.com/your-username/your-repo/main/tmdb_5000_movies.csv"
        movies = pd.read_csv(movies_url)
        print("CSV file loaded from URL successfully!")
        
        # Check if poster_path column exists, if not create it
        if 'poster_path' not in movies.columns:
            print("Warning: poster_path column not found in CSV")
            movies['poster_path'] = None
        
        # Prepare the data - combine title, overview, and genres for content-based filtering
        movies['clean_title'] = movies['title'].apply(clean_title)
        
        # Fill NaN values
        movies['overview'] = movies['overview'].fillna('')
        movies['genres'] = movies['genres'].fillna('[]')
        movies['poster_path'] = movies['poster_path'].fillna('')
        
        # Create a content column for similarity matching
        movies['content'] = (
            movies['title'] + ' ' + 
            movies['overview'] + ' ' + 
            movies['genres'].apply(extract_genre_names)
        )
        
        # Initialize vectorizer with movie content
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(movies['content'])
        print("TF-IDF vectorizer trained successfully with movie content!")
        return True
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def create_sample_data():
    """Create sample data for testing when CSV files are missing"""
    global movies, vectorizer, tfidf
    
    sample_movies = [
        {'id': 1, 'title': 'The Dark Knight', 'genres': '[{"id": 28, "name": "Action"}]', 'release_date': '2008-07-18', 'vote_average': 9.0},
        {'id': 2, 'title': 'Inception', 'genres': '[{"id": 28, "name": "Action"}]', 'release_date': '2010-07-16', 'vote_average': 8.8},
        {'id': 3, 'title': 'Avatar', 'genres': '[{"id": 12, "name": "Adventure"}]', 'release_date': '2009-12-18', 'vote_average': 7.8},
        {'id': 4, 'title': 'The Matrix', 'genres': '[{"id": 28, "name": "Action"}]', 'release_date': '1999-03-31', 'vote_average': 8.7},
        {'id': 5, 'title': 'Pulp Fiction', 'genres': '[{"id": 80, "name": "Crime"}]', 'release_date': '1994-10-14', 'vote_average': 8.9}
    ]
    
    movies = pd.DataFrame(sample_movies)
    movies["clean_title"] = movies["title"].apply(clean_title)
    
    # Initialize vectorizer with sample data
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])
    print("Sample data created and vectorizer trained!")

def clean_title(title):
    if pd.isna(title):
        return ""
    return re.sub("[^a-zA-Z0-9]", "", str(title))

def search(title):
    if movies.empty:
        return movies
    
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

def find_similar_movies(movie_id):
    try:
        if movies.empty:
            return movies[["id", "title", "genres", "release_date", "vote_average"]]
            
        movie_title = movies[movies["id"] == movie_id]["title"].iloc[0]
        similar_movies = search(movie_title)
        return similar_movies[["id", "title", "genres", "release_date", "vote_average"]]
    except Exception as e:
        print(f"Error finding similar movies: {e}")
        return pd.DataFrame()

def parse_genres(genres_str):
    """Parse genres from string format to list"""
    try:
        if pd.isna(genres_str):
            return []
        if isinstance(genres_str, str):
            genres_list = ast.literal_eval(genres_str)
            return [genre['name'] for genre in genres_list] if genres_list else []
        return []
    except:
        return []

# Load data when the app starts
data_loaded = load_data()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", 
        "movies_loaded": not movies.empty,
        "total_movies": len(movies),
        "using_sample_data": len(movies) > 0 and len(movies) < 10
    })

@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        data = request.json
        title = data.get('title', '')
        
        if len(title) < 3:
            return jsonify({"error": "Title too short"}), 400
        
        print(f"Searching for: {title}")
        results = search(title)
        movies_list = []
        
        for _, row in results.iterrows():
            movies_list.append({
                "id": int(row['id']),
                "title": row['title'],
                "genres": parse_genres(row['genres']),
                "year": pd.to_datetime(row['release_date']).year if pd.notna(row['release_date']) else None,
                "rating": float(row['vote_average']) if pd.notna(row['vote_average']) else None
            })
        
        print(f"Found {len(movies_list)} results")
        return jsonify({"movies": movies_list})
    
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.json
        movie_id = data.get('movie_id')
        
        if not movie_id:
            return jsonify({"error": "Movie ID is required"}), 400
        
        print(f"Finding recommendations for movie ID: {movie_id}")
        similar_movies = find_similar_movies(movie_id)
        movies_list = []
        
        for _, row in similar_movies.iterrows():
            movies_list.append({
                "id": int(row['id']),
                "title": row['title'],
                "genres": parse_genres(row['genres']),
                "year": pd.to_datetime(row['release_date']).year if pd.notna(row['release_date']) else None,
                "rating": float(row['vote_average']) if pd.notna(row['vote_average']) else None
            })
        
        print(f"Found {len(movies_list)} recommendations")
        return jsonify({"movies": movies_list})
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    """Get a list of all movies for debugging"""
    movies_list = []
    for _, row in movies.iterrows():
        movies_list.append({
            "id": int(row['id']),
            "title": row['title'],
            "genres": parse_genres(row['genres'])
        })
    return jsonify({"movies": movies_list})

@app.route('/api/files', methods=['GET'])
def check_files():
    """Check if required files exist"""
    movies_path = os.path.join(current_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(current_dir, "tmdb_5000_credits.csv")
    
    return jsonify({
        "current_directory": current_dir,
        "movies_file_exists": os.path.exists(movies_path),
        "credits_file_exists": os.path.exists(credits_path),
        "movies_file_path": movies_path,
        "credits_file_path": credits_path
    })

if __name__ == '__main__':
    print("Starting Movie Recommendation API...")
    print(f"Current directory: {current_dir}")
    print(f"Movies dataset shape: {movies.shape}")
    print(f"Using {'SAMPLE' if len(movies) < 10 else 'REAL'} data")
    app.run(debug=True, port=5002, host='0.0.0.0')
