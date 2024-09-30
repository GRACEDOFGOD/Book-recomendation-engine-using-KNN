# Book-recomendation-engine-using-KNN
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Step 2: Load the dataset
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding='latin-1')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding='latin-1')
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding='latin-1')

# Step 3: Data Cleaning
# Rename columns for easier access
books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']
users.columns = ['User-ID', 'Location', 'Age']

# Filter data
ratings = ratings[ratings['Book-Rating'] > 0]

# Remove users with fewer than 200 ratings and books with fewer than 100 ratings
user_counts = ratings['User-ID'].value_counts()
ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 200].index)]

book_counts = ratings['ISBN'].value_counts()
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 100].index)]

# Step 4: Create the user-book matrix
user_book_matrix = ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

# Convert the matrix to a sparse matrix for efficiency
user_book_sparse_matrix = csr_matrix(user_book_matrix.values)

# Step 5: Train the K-Nearest Neighbors (KNN) model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_book_sparse_matrix)

# Step 6: Book Recommendation Function
def get_recommends(book_title):
    # Find the book's ISBN
    book_info = books[books['Book-Title'] == book_title]
    if book_info.empty:
        return f"Book '{book_title}' not found in the dataset."
    
    book_isbn = book_info['ISBN'].values[0]
    
    # Get the book index from the matrix
    try:
        book_idx = user_book_matrix.columns.get_loc(book_isbn)
    except KeyError:
        return f"Book '{book_title}' has no ratings."
    
    # Use KNN to find similar books
    distances, indices = model_knn.kneighbors(user_book_sparse_matrix[:, book_idx], n_neighbors=6)
    
    # Retrieve book titles for the recommended books
    recommendations = []
    for idx, distance in zip(indices.flatten(), distances.flatten()):
        if idx == book_idx:
            continue  # Skip the input book itself
        recommended_book_isbn = user_book_matrix.columns[idx]
        recommended_book_title = books[books['ISBN'] == recommended_book_isbn]['Book-Title'].values[0]
        recommendations.append([recommended_book_title, distance])
    
    return [book_title, recommendations]

# Step 7: Test the Recommendation System
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))
