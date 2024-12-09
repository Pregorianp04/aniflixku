from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Daftar genre
genres = ["Action", "Comedy", "Kids", "Parody", "Historical", "Military", "Thriller", "Sports", "Horror", "School", "Adventure", "Fantasy", "Romance", "Drama"]

# Import dataset
# URL Google Drive dengan File ID
url_anime = "https://drive.google.com/uc?id=1tW3EhTpNeaHR3dSQroyD9NY38JjZjRFu"
url_rating = "https://drive.google.com/uc?id=1P-pjMrStiWe8vM04ti1M1hq3b6QaDp9Y"

# Baca file CSV dari Google Drive
animes = pd.read_csv(url_anime)
ratings = pd.read_csv(url_rating)

# Data preparation
ratings = ratings.dropna()
animes = animes.dropna()
ratings = pd.merge(animes, ratings, left_on='anime_id', right_on='anime_id')

# Pivot table
user_item_matrix = ratings.pivot_table(index='userId', columns='anime_id', values='rating').fillna(0).values

# Validasi data
if np.all(user_item_matrix.sum(axis=1) == 0):
    raise ValueError("Semua baris dalam user_item_matrix kosong. Tidak ada data yang bisa diproses.")

# Manual cosine similarity
def manual_cosine_similarity(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  
    normalized_matrix = matrix / norms
    similarity = np.dot(normalized_matrix, normalized_matrix.T)
    return similarity

user_similarity = manual_cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity)

def filter_by_genre(animes, preferred_genres):
    return animes[animes['genre'].str.contains('|'.join(preferred_genres), case=False, na=False)]

def recommend_anime(user_id, preferred_genres, num_recommendations=10):
    # Filter anime berdasarkan genre
    genre_filtered_animes = filter_by_genre(animes, preferred_genres)

    # Ambil similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:]
    similar_users_ratings = user_item_matrix[similar_users.index]

    # Hitung weighted ratings
    weighted_ratings = np.dot(similar_users_ratings.T, similar_users.values)
    recommendation_scores = weighted_ratings / similar_users.sum()

    # Filter hanya anime yang belum dirating oleh user
    unrated_animes = np.where(user_item_matrix[user_id] == 0)[0]
    recommendations = pd.Series(recommendation_scores[unrated_animes], index=unrated_animes).sort_values(ascending=False)

    # Filter hasil rekomendasi berdasarkan genre
    recommended_ids = recommendations.index[recommendations.index.isin(genre_filtered_animes['anime_id'])]
    recommended_names = animes[animes['anime_id'].isin(recommended_ids)]['name'].tolist()

    return recommended_names[:num_recommendations]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            selected_genres = request.form.getlist('genres[]')
            user_id = 1  # Contoh user_id
            recommendations = recommend_anime(user_id, selected_genres)
            return render_template('index.html', genres=genres, recommendations=recommendations)
        except Exception as e:
            return render_template('index.html', genres=genres, error=str(e))
    return render_template('index.html', genres=genres)

if __name__ == '__main__':
    app.run(debug=True)
