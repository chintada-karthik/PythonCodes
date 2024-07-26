import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Sample data: User-Book-Ratings matrix
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 5],
    'book_id': [101, 102, 103, 101, 104, 101, 102, 103, 104],
    'rating': [5, 3, 4, 5, 2, 4, 3, 4, 5]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a pivot table (user-item matrix)
pivot_table = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# Convert pivot table to a sparse matrix
sparse_matrix = csr_matrix(pivot_table.values)

# Fit the KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
knn.fit(sparse_matrix)

def get_recommendations(user_id, n_recommendations=3):
    user_index = pivot_table.index.tolist().index(user_id)
    distances, indices = knn.kneighbors(sparse_matrix[user_index], n_neighbors=n_recommendations + 1)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        book_index = pivot_table.columns[indices.flatten()[i]]
        recommendations.append(book_index)
    
    return recommendations

# Example usage
user_id = 1
recommended_books = get_recommendations(user_id)
print(f"Recommended books for user {user_id}: {recommended_books}")