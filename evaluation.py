# David Levin 316554641

import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error


def setData(data_set):
    """
    Function role is to drop all the unnecessary data from the given test set & create new custom list.
    """
    updated_test_set = data_set[data_set['rating'] >= 4]
    return updated_test_set.groupby('userId')['movieId'].apply(list).reset_index(name='list').values


def precision_10(test_set, cf, is_user_based=True):
    """
    Function role is to implement the precisionK evaluation Index, when k=10.
    """
    # Initialize variables
    val, helper, k = 0, 0, 10

    # Drop all the unnecessary data from the given test set & create new custom list
    data = setData(test_set)

    # Iterate over all the data (after the dropping process)
    for user in range(len(data)):
        counter = 0
        # Predict the K movies for the specific user (get the moviesID's, NOT movies names !!!)
        top_k = cf.predict_movies_IDs(data[user][0], k, is_user_based)
        # If we calculate it right - add one to the counter
        for movieID in data[user][1]:
            if movieID in top_k:
                counter += 1
        helper += counter / 10

    val = helper / len(data)
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based=True):
    """
    Function role is to implement the ARHA evaluation Index, when k=10.
    """
    # Initialize variables
    val, helper, k = 0, 0, 10

    # Drop all the unnecessary data from the given test set & create new custom list
    data = setData(test_set)

    # Iterate over all the data (after the dropping process)
    for user in range(len(data)):
        counter = 0
        # Predict the K movies for the specific user (get the moviesID's, NOT movies names !!!)
        top_k = cf.predict_movies_IDs(data[user][0], 10, is_user_based)
        top_k.reverse()
        # If we calculate it right - add one to the counter & divide it
        for i in range(k):
            if top_k[i] in data[user][1]:
                counter += 1 / (i + 1)
        helper += counter

    # Print the result
    val = helper / len(data)
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based=True):
    """
    Function role is to implement the RSME evaluation Index, when k=10.
    """
    # Set the matrix type
    matrix = None
    if is_user_based:
        matrix = cf.user_based_matrix
    else:
        matrix = cf.item_based_matrix

    val, difference = 0, []
    # Get each of the column in the table
    usersID = test_set['userId']
    ratings = test_set['rating']
    moviesID = test_set['movieId']

    for index in range(len(test_set)):
        # Calculate the index of the specific user in the users matrix
        user_index_matrix = cf.users_map[usersID[index]]
        # Calculate the index of the specific movie in the movies matrix
        movie_index_matrix = cf.movies_map[moviesID[index]]
        # Sum the difference values
        helper = matrix[user_index_matrix][movie_index_matrix] - ratings[index]
        difference.append(helper ** 2)

    # Print the result
    val = sqrt(sum(difference) / len(test_set))
    print("RMSE: " + str(val))
