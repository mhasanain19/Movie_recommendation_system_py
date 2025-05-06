import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt

# Step 1: Load data
# Download the MovieLens 100K dataset and place the "u.data" file in the same directory.
data = pd.read_csv('u.data', sep='\t', names=["UserId", "MovieId", "Rating", "Timestamp"])

# Preview the data
print(data.head())

# Step 2: Prepare the data for Surprise library
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['UserId', 'MovieId', 'Rating']], reader)

# Step 3: Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Step 4: Use Singular Value Decomposition (SVD) algorithm
svd = SVD()

# Train the model
svd.fit(trainset)

# Step 5: Make predictions on the testset
predictions = svd.test(testset)

# Step 6: Evaluate the performance
rmse = accuracy.rmse(predictions)
print(f'Root Mean Squared Error (RMSE): {rmse}')


# Step 7: Recommend movies
def get_movie_recommendation(user_id, num_recommendations=5):
    # Get all movies that the user hasn't rated yet
    movie_ids = data['MovieId'].unique()
    rated_movies = data[data['UserId'] == user_id]['MovieId'].values
    unrated_movies = [movie_id for movie_id in movie_ids if movie_id not in rated_movies]

    # Predict ratings for all unrated movies
    movie_predictions = []
    for movie_id in unrated_movies:
        predicted_rating = svd.predict(user_id, movie_id).est
        movie_predictions.append((movie_id, predicted_rating))

    # Sort the movies by predicted rating and return the top recommendations
    movie_predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = movie_predictions[:num_recommendations]

    return top_movies


# Get top 5 movie recommendations for a specific user
user_id = 1  # User ID to get recommendations for
recommendations = get_movie_recommendation(user_id)
print("Top 5 Movie Recommendations for User ID 1:")
for movie_id, rating in recommendations:
    print(f"Movie ID: {movie_id}, Predicted Rating: {rating}")

# Step 8: Visualize recommendations (optional)
# Let's visualize the top 10 predicted movie ratings for the user
top_movie_ids = [movie_id for movie_id, rating in recommendations]
top_movie_ratings = [rating for movie_id, rating in recommendations]

plt.figure(figsize=(10, 6))
plt.bar(top_movie_ids, top_movie_ratings, color='skyblue')
plt.xlabel('Movie IDs')
plt.ylabel('Predicted Rating')
plt.title(f'Top Movie Recommendations for User {user_id}')
plt.show()