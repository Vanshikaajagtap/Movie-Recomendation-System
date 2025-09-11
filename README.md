# Movie-Recomendation-System

A content-based movie recommendation system that suggests similar movies based on title similarity using TF-IDF vectorization and cosine similarity.

## How It Works

### 1. Data Loading and Preprocessing
- Loads two datasets: `tmdb_5000_movies.csv` (movie details) and `tmdb_5000_credits.csv` (cast/crew info)
- Cleans movie titles by removing special characters using regex
- Creates a clean title column for better text processing

### 2. TF-IDF Vectorization
- Uses `TfidfVectorizer` with n-gram range (1,2) to capture single words and word pairs
- Transforms all cleaned movie titles into numerical TF-IDF vectors
- This converts text data into a format that can be used for mathematical similarity calculations

### 3. Search Function
- Takes a search query and cleans it using the same preprocessing
- Transforms the query into a TF-IDF vector
- Calculates cosine similarity between the query and all movies
- Returns the top 5 most similar movies based on title similarity

### 4. Recommendation Engine
- `find_similar_movies()` function takes a movie ID
- Finds the movie title for that ID
- Uses the search function to find movies with similar titles
- Returns similar movies with their titles and genres

## Running the Code
1. Ensure both CSV files are in the same directory as your script

2. Run the code in a Jupyter notebook environment (required for widgets)

3. Type a movie title in the search box that appears

4. View the similar movie recommendations below

## Example Output
When searching for "Avengers: Infinity War", the system might return:

- Avengers: Endgame
- The Avengers
- Avengers: Age of Ultron
- Avengers Assemble
- Ultimate Avengers
