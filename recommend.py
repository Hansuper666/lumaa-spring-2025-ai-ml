import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(csv_path, nrows):
    """
    Load the movies metadata from a CSV file.
    """
    # Load CSV file
    df = pd.read_csv(csv_path, low_memory=False, nrows=nrows)
    
    df['overview'] = df['overview'].fillna('')
    
    df = df.dropna(subset=['original_title']).reset_index(drop=True)
    
    return df[['original_title', 'overview']]

def build_tfidf_matrix(data):
    """
    Convert the movie overviews into a TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['overview'])
    return tfidf_matrix, vectorizer

def recommend_items(query, data, tfidf_matrix, vectorizer, top_n=5):
    """
    Given a user query, compute cosine similarity between the query and each movie overview.
    Return the top_n movies along with their similarity scores.
    """
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between the query and all movie overviews
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    
    recommendations = data.iloc[top_indices].copy()
    recommendations['similarity'] = cosine_sim[top_indices]
    
    return recommendations

if __name__ == "__main__":
    # Check if a user query was provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python recommend.py \"<your movie preference description>\"")
        sys.exit(1)
    
    user_query = sys.argv[1]
    
    csv_path = "movies_metadata.csv"
    
    print("Loading data...")
    data = load_data(csv_path, nrows=5000)

    print("number of data: ", data.shape[0])
    
    print("Building TF-IDF matrix and recommendations...")
    tfidf_matrix, vectorizer = build_tfidf_matrix(data)

    recommendations = recommend_items(user_query, data, tfidf_matrix, vectorizer, top_n=5)
    
    print("\nTop Recommendations:")
    for idx, row in recommendations.iterrows():
        print(f"{row['original_title']} - Similarity Score: {row['similarity']:.2f}")
