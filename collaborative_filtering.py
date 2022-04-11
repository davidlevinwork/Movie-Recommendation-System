import sys
import heapq
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


# <editor-fold desc="Helpers">
def addNewUser(data, rating, userID):
    """
    Function role is for debug purposes: append rating r for items on behalf of a debug user.
    """
    new_val = pd.DataFrame(data, columns=['movieId'])
    new_val[['userId', 'rating', ]] = userID, rating
    return new_val
# </editor-fold>


class collaborative_filtering:
    def __init__(self):
        self.data = None                    # The given data frame
        self.matrix = None                  # The matrix that will hold the updated data
        self.users_map = None               # Matrix that maps between user & index
        self.movies_map = None              # Matrix that maps between movie & index
        self.data_matrix_df = None          # Matrix that holds all data: rows (users) & columns (movies) & NaN
        self.similarity_matrix = None       # The similarity matrix
        self.user_based_matrix = []         # The predicted user based matrix
        self.item_based_matrix = []         # The predicted item based matrix

    def create_user_based_matrix(self, data):
        """
        Function role is to create the user based matrix.
        """
        # Save the entire DF
        self.data = data
        # Add a new fake user to the ratings table
        ratings = self.create_fake_user(data[0])
        # Set the "new given data" after adding the fake user
        updated_data = (ratings, data[1])
        # Build the user X movies matrix
        self.data_matrix_df = self.createUserMovieMatrix(updated_data)
        # Generate predicted ratings matrix
        self.user_based_matrix = self.createPredictedRatingMatrixUserBased()

    def create_item_based_matrix(self, data):
        """
        Function role is to create the item based matrix.
        """
        # Save the entire DF
        self.data = data
        # Add a new fake user to the ratings table
        ratings = self.create_fake_user(data[0])
        # Set the "new given data" after adding the fake user
        updated_data = (ratings, data[1])
        # Build the user X movies matrix
        self.data_matrix_df = self.createUserMovieMatrix(updated_data)
        # Generate predicted ratings matrix
        self.item_based_matrix = self.createPredictedRatingMatrixItemBased()

    def createUserMovieMatrix(self, data):
        """
        Function role is to build the user X movies matrix (in a form of data frame).
        """
        ratings = data[0]
        # Drop all the unnecessary duplication
        unique_user = ratings['userId'].drop_duplicates()
        unique_movie = ratings['movieId'].drop_duplicates()
        # Save the dictionary that we created for the users & values
        self.users_map = {key: value for value, key in enumerate(np.sort(unique_user.to_numpy()).tolist())}
        self.movies_map = {key: value for value, key in enumerate(np.sort(unique_movie.to_numpy()).tolist())}
        # Create the matrix that will hold the updated data
        self.matrix = ratings.pivot(index='userId', columns='movieId', values='rating').to_numpy()

        return pd.DataFrame(self.matrix)

    def create_fake_user(self, ratings):
        """
        Function role is for debug purpose: add a "debug" user to ensure that the system works as expected.
        """
        # Unloved genre
        thriller = [1892, 142488, 540, 2707, 3005, 5266, 457, 1343, 1061]
        # Favorites generes
        drama = [14, 31, 147, 300, 306, 307, 337, 428, 1041, 1093, 1095, 1103, 1104, 1124, 1172, 1185, 1186, 1231, 1237]
        action = [55247, 3578, 1287, 2019, 7090, 2013, 1801, 73321, 2421]

        # Create a unique id for the fake user (this is NOT magic number - this is what was written in the instructions)
        userID = 283238

        # Add the values to the table
        ratings = ratings.append(addNewUser(thriller, 1, userID), ignore_index=True)
        ratings = ratings.append(addNewUser(drama, 5, userID), ignore_index=True)
        ratings = ratings.append(addNewUser(action, 5, userID), ignore_index=True)

        return ratings

    def createPredictedRatingMatrixUserBased(self):
        """
        Function role is to create the predicted rating matrix.
        We will use the result as the desired output: user_based_matrix.
        """
        # Calculate the mean rating of each voter (while ignoring NaN values)
        mean_matrix = np.nanmean(self.data_matrix_df, axis=1).reshape(-1, 1)
        # Calculate the normalized matrix (subtract from 'matrix' the mean user rating)
        ratings_diff = (self.matrix - mean_matrix)
        # replace NaN values with 0 (after the mean calculation)
        ratings_diff[np.isnan(ratings_diff)] = 0

        # Calculate USER x USER similarity matrix (the main diagonal will be 1) (1 --> perfect fit | -1 --> poor fit)
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
        # Save the user similarity matrix
        self.similarity_matrix = user_similarity

        # Generate predicted ratings matrix
        pred_helper = np.array([np.abs(user_similarity).sum(axis=1)]).T
        user_based_matrix = mean_matrix + user_similarity.dot(ratings_diff) / pred_helper

        return user_based_matrix

    def createPredictedRatingMatrixItemBased(self):
        """
        Function role is to create the predicted rating matrix.
        We will use the result as the desired output: item_based_matrix.
        """
        # Calculate the mean rating of each voter (while ignoring NaN values)
        mean_matrix = np.nanmean(self.data_matrix_df, axis=1).reshape(-1, 1)
        # Calculate the normalized matrix (subtract from 'matrix' the mean user rating)
        ratings_diff = (self.matrix - mean_matrix)
        # replace NaN values with 0 (after the mean calculation)
        ratings_diff[np.isnan(ratings_diff)] = 0

        # Calculate USER x USER similarity matrix (the main diagonal will be 1) (1 --> perfect fit | -1 --> poor fit)
        item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')
        # Save the user similarity matrix
        self.similarity_matrix = item_similarity

        # Generate predicted ratings matrix
        pred_helper = np.array([np.abs(item_similarity).sum(axis=1)])
        item_based_matrix = mean_matrix + ratings_diff.dot(item_similarity) / pred_helper

        return item_based_matrix

    def getTopKMovies(self, movies_identifiers):
        """
        Function role is to return the top K movies.
        """
        id_to_name = {}
        titles = self.data[1]['title']
        movies = self.data[1]['movieId']
        for movie_id, title in zip(movies, titles):
            id_to_name[movie_id] = title

        topK = []
        for identifier in movies_identifiers:
            topK.append(id_to_name[identifier])

        return topK

    def predict_movies(self, user_id, k, is_user_based=True):
        """
        Function role is to predict the k movies that are the best for the given user - the movies names.
        """
        # Set the matrix type
        matrix = None
        if is_user_based:
            matrix = self.user_based_matrix
        else:
            matrix = self.item_based_matrix

        # Get original user index
        origin_user_id = self.users_map[int(user_id)]
        # Get the predictions of the NaN values for the given user (the movies that the user didnt add)
        nan_predictions = self.getPredictedNanMoviesValues(matrix, origin_user_id)
        # Get the movies identifiers
        movies_identifiers = self.getMoviesIdentifiers(nan_predictions, k)
        # Get the top K movies by the identifies that we calculated & reverse it
        result = self.getTopKMovies(movies_identifiers)
        result.reverse()
        return result

    def predict_movies_IDs(self, user_id, k, is_user_based=True):
        """
        Function role is to predict the k movies that are the best for the given user - the movies id's.
        """
        # Set the matrix type
        matrix = None
        if is_user_based:
            matrix = self.user_based_matrix
        else:
            matrix = self.item_based_matrix

        # Get original user index
        origin_user_id = self.users_map[int(user_id)]

        # Get the predictions of the NaN values for the given user (the movies that the user didnt add)
        nan_predictions = self.getPredictedNanMoviesValues(matrix, origin_user_id)
        # Get the movies identifiers

        return self.getMoviesIdentifiers(nan_predictions, k)

    def getMoviesIdentifiers(self, nan_predictions, k):
        """
        Function role is to return the movies identifiers value (ID's).
        """
        # Create list out of keys and values separately from the movie map
        key_list = list(self.movies_map)

        movies_identifiers = []
        nan_predictions_length = len(nan_predictions)

        for _ in range(k):
            # Get the top K indexes by rating value
            movies_identifier = nan_predictions[nan_predictions_length - k][0]
            # Get the movie ID of the given index
            movie = key_list[movies_identifier]
            # Add the movie ID to the array
            movies_identifiers.append(movie)
            # Update the iterator value
            nan_predictions_length += 1

        return movies_identifiers

    def getPredictedNanMoviesValues(self, matrix, userID):
        """
        Function role is to return the predicted movies (the movies that the user didnt rate them = NaN values).
        """
        # Get the predicted rating row of the given user while creating tuple of (index, rating value)
        predicted_ratings = []
        for index, rating in enumerate(matrix[userID]):
            predicted_ratings.append((index, rating))

        # Get the user NaN data values from the row of the given user
        nan_indexes = np.argwhere(np.isnan(self.matrix[userID]))

        nan_predictions = []
        for index in nan_indexes:
            nan_predictions.append(predicted_ratings[index[0]])

        # Sort the NaN predictions values by the second element in the tuple: rating
        nan_predictions.sort(key=lambda val: val[1])
        return nan_predictions
