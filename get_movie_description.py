import sys
import pandas as pd

def load_data(csv_path, nrows):
    """
    Load the movies metadata from a CSV file.
    """
    df = pd.read_csv(csv_path, low_memory=False, nrows=nrows)
    df['overview'] = df['overview'].fillna('')
    df = df.dropna(subset=['original_title']).reset_index(drop=True)
    return df[['original_title', 'overview']]

def get_movie_overview(movie_name, data):
    """
    Given a movie name, return its description (overview) from the dataset.
    """
    # Perform a case-insensitive match
    movie = data[data['original_title'].str.lower() == movie_name.lower()]
    if movie.empty:
        return None
    return movie.iloc[0]['overview']

if __name__ == "__main__":
    # Check if a movie name was provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python get_movie_description.py \"<movie name>\"")
        sys.exit(1)
    
    movie_name = sys.argv[1]
    csv_path = "movies_metadata.csv"
    
    data = load_data(csv_path, nrows=5000)
    
    overview = get_movie_overview(movie_name, data)
    
    if overview:
        print(f"\nDescription for '{movie_name}':\n{overview}")
    else:
        print(f"\nMovie '{movie_name}' not found in the dataset.")
