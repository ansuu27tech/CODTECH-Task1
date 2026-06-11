import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_data(filepath):
    """Load the movie dataset from CSV."""
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return None
    df = pd.read_csv(filepath)
    return df

def clean_and_combine_features(df):
    """Combine text features for TF-IDF Vectorization."""
    # Fill missing values with empty string
    for feature in ['genre', 'overview', 'keywords']:
        df[feature] = df[feature].fillna('')
    
    # Combine the features
    df['combined_features'] = df['genre'] + ' ' + df['overview'] + ' ' + df['keywords']
    return df

def get_recommendations(title, df, cosine_sim):
    """Return top 10 recommended movies based on similarity score."""
    try:
        # Get the index of the movie that matches the title
        idx = df[df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        print(f"\nMovie '{title}' not found in the dataset.")
        return []

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    # Ignore the first one since it is the movie itself (index 0)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices].tolist()

def plot_visualizations(df, cosine_sim):
    """Generate and save data visualizations."""
    print("\nGenerating visualizations...")
    
    # 1. Genre Distribution
    genres = df['genre'].str.split(' ').explode()
    genre_counts = genres.value_counts()
    
    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Genre Distribution')
    plt.xlabel('Genres')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('genre_distribution.png')
    plt.close()
    
    # 2. Similarity Statistics
    sim_values = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
    plt.figure(figsize=(8, 5))
    plt.hist(sim_values, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Cosine Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('similarity_statistics.png')
    plt.close()
    
    print("Visualizations saved as 'genre_distribution.png' and 'similarity_statistics.png'.")

def main():
    print("--- Movie Recommendation Engine Using Machine Learning ---")
    
    # 1. Read Dataset
    df = load_data('data/movies.csv')
    if df is None:
        return
        
    print("\nDataset Sample:")
    print(df[['title', 'genre']].head())

    # 2. Data Cleaning and Text Feature Engineering
    df = clean_and_combine_features(df)
    
    # 3. TF-IDF Vectorization
    print("\nVectorizing text features...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    
    # 4. Compute Cosine Similarity Matrix
    print("Computing similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 5. Visualizations
    plot_visualizations(df, cosine_sim)
    
    # 6. Recommendation Example Output
    search_movie = "The Matrix"
    print(f"\nTop 10 Recommendations for '{search_movie}':")
    recommendations = get_recommendations(search_movie, df, cosine_sim)
    
    if recommendations:
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")

if __name__ == "__main__":
    main()
