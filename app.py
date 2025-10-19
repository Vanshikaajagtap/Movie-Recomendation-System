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

current_dir = os.path.dirname(os.path.abspath(__file__))

movies = pd.DataFrame()
vectorizer = None
tfidf_matrix = None

def load_data():
    global movies, vectorizer, tfidf_matrix
    
    try:
        movies_path = os.path.join(current_dir, "tmdb_5000_movies.csv")
        movies = pd.read_csv(movies_path)
        print("CSV file loaded successfully!")
        print(f"Loaded {len(movies)} movies")
        
        if 'poster_path' not in movies.columns:
            print("Warning: poster_path column not found in CSV")
            movies['poster_path'] = None
        
        movies['clean_title'] = movies['title'].apply(clean_title)
        
        movies['overview'] = movies['overview'].fillna('')
        movies['genres'] = movies['genres'].fillna('[]')
        movies['poster_path'] = movies['poster_path'].fillna('')
        
        movies['content'] = (
            movies['title'] + ' ' + 
            movies['overview'] + ' ' + 
            movies['genres'].apply(extract_genre_names)
        )
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(movies['content'])
        print("TF-IDF vectorizer trained successfully with movie content!")
        return True
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def extract_genre_names(genres_str):
    
    try:
        if pd.isna(genres_str) or genres_str == '':
            return ''
        if isinstance(genres_str, str):
            genres_list = ast.literal_eval(genres_str)
            return ' '.join([genre['name'] for genre in genres_list]) if genres_list else ''
        return ''
    except:
        return ''

def clean_title(title):
    if pd.isna(title):
        return ""
    return re.sub("[^a-zA-Z0-9 ]", "", str(title)).lower()

def search_by_title(title):
    clean_query = clean_title(title)
    print(f"Searching for: '{title}' (cleaned: '{clean_query}')")
   
    exact_matches = movies[movies['clean_title'].str.contains(clean_query, case=False, na=False)]
    if not exact_matches.empty:
        print(f"Found {len(exact_matches)} exact matches")
        return exact_matches.head(5)
    
    query_vec = vectorizer.transform([clean_query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    indices = np.argsort(similarity)[-5:][::-1]
    results = movies.iloc[indices]
    
    print(f"Found {len(results)} similar matches")
    return results

def get_similar_movies_by_content(movie_id, n_recommendations=5):
    """Get similar movies based on content (overview, genres, etc.)"""
    try:
        movie_idx = movies[movies['id'] == movie_id].index[0]
        
        similarity_scores = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix).flatten()
        
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        return movies.iloc[similar_indices]
        
    except Exception as e:
        print(f"Error finding similar movies by content: {e}")
        return pd.DataFrame()

def parse_genres(genres_str):
    try:
        if pd.isna(genres_str):
            return []
        if isinstance(genres_str, str):
            genres_list = ast.literal_eval(genres_str)
            return [genre['name'] for genre in genres_list] if genres_list else []
        return []
    except:
        return []

def get_poster_url(poster_path):
    if pd.isna(poster_path) or poster_path == '' or poster_path is None:
        return "https://via.placeholder.com/300x450/2a3a4a/ffffff?text=No+Poster"
    
    clean_path = str(poster_path)
    if clean_path and not clean_path.startswith('/'):
        clean_path = '/' + clean_path
    
    return f"https://image.tmdb.org/t/p/w300{clean_path}"

data_loaded = load_data()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok", 
        "movies_loaded": not movies.empty,
        "total_movies": len(movies)
    })

@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        data = request.json
        title = data.get('title', '')
        
        if len(title) < 3:
            return jsonify({"error": "Title too short"}), 400
        
        print(f"Searching for: '{title}'")
        results = search_by_title(title)
        movies_list = []
        
        for _, row in results.iterrows():
            poster_path = row.get('poster_path', '') if 'poster_path' in row else ''
            
            movies_list.append({
                "id": int(row['id']),
                "title": row['title'],
                "genres": parse_genres(row['genres']),
                "year": pd.to_datetime(row['release_date']).year if pd.notna(row['release_date']) else None,
                "rating": float(row['vote_average']) if pd.notna(row['vote_average']) else None,
                "overview": row['overview'] if pd.notna(row['overview']) else "",
                "poster_url": get_poster_url(poster_path)
            })
        
        print(f"Found {len(movies_list)} results")
        for movie in movies_list:
            print(f" - {movie['title']} (ID: {movie['id']})")
        
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
        
        print(f"Finding content-based recommendations for movie ID: {movie_id}")
        similar_movies = get_similar_movies_by_content(movie_id, n_recommendations=5)
        movies_list = []
        
        for _, row in similar_movies.iterrows():
            poster_path = row.get('poster_path', '') if 'poster_path' in row else ''
            
            movies_list.append({
                "id": int(row['id']),
                "title": row['title'],
                "genres": parse_genres(row['genres']),
                "year": pd.to_datetime(row['release_date']).year if pd.notna(row['release_date']) else None,
                "rating": float(row['vote_average']) if pd.notna(row['vote_average']) else None,
                "overview": row['overview'] if pd.notna(row['overview']) else "",
                "poster_url": get_poster_url(poster_path)
            })
        
        print(f"Found {len(movies_list)} content-based recommendations")
        return jsonify({"movies": movies_list})
    
    except Exception as e:
        print(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/columns', methods=['GET'])
def debug_columns():
    """Debug endpoint to see what columns are available"""
    return jsonify({
        "columns": list(movies.columns) if not movies.empty else []
    })

@app.route('/api/debug/search/<title>', methods=['GET'])
def debug_search(title):
    """Debug endpoint to see what movies are found for a search"""
    results = search_by_title(title)
    movies_list = []
    
    for _, row in results.iterrows():
        movies_list.append({
            "id": int(row['id']),
            "title": row['title'],
            "clean_title": row['clean_title'],
            "genres": parse_genres(row['genres'])
        })
    
    return jsonify({"search_query": title, "results": movies_list})

@app.route('/')
def home():
    return jsonify({
        "message": "Movie Recommendation API is working!",
        "status": "success", 
        "total_movies": len(movies),
        "endpoints": {
            "health": "/api/health",
            "search": "/api/search (POST)",
            "recommend": "/api/recommend (POST)"
        }
    })

if __name__ == '__main__':
    print("Starting Movie Recommendation API...")
    print(f"Current directory: {current_dir}")
    print(f"Movies dataset shape: {movies.shape}")
    if not movies.empty:
        print(f"Available columns: {list(movies.columns)}")
    app.run(debug=True, port=5002, host='0.0.0.0')
