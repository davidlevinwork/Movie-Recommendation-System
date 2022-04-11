# Movie-Recommendation-System
Implementation of a recommendation system based on Collaborating Filtering: User based & Item based. Construction of prediction matrices using various evaluation indices: ARHA, RMSE, Precision@k etc.

1. [General](#General)
    - [Background](#background)
    - [The Recommendation System](#the-recommendation-system)
    - [Estimations](#estimations)
2. [Dependencies](#dependencies) 

## General

### Background
Implementation of a recommendation system based on Collaborating Filtering: User based &amp; Item based. Construction of prediction matrices using various evaluation indices: ARHA, RMSE, Precision@k etc.

The recommendation system is divided into 2 main parts as follows:
- The recommendation system based on Collaborating Filtering: 

### The Recommendation System
The ``collaborative_filtering.py`` file has a class for implementing a CF recommendation system.
For the construction of the prediction matrix, we will use the cosine similarity index, that computes the index of the angle between the two
Vectors in space: ![](https://github.com/davidlevinwork/Movie-Recommendation-System/blob/main/Cosine_Similarity.jpg) </br>

The system implements two types of recommendations:
1) User based system
2) Item based system

Given any username and value k, after the user chooses whether he wants to get a user-based or item-based prediction, the system will return the most k recommended movies for the user.

### Estimations
The ``test.csv`` file is a test file of user ratings, which will useb by the various recommendation systems to evaluate.
I used 3 different evaluation metrics:
1) P@K: How many relevant items are present in the top-k recommendations of your system.
2) ARHR: Commonly used metric for ranking evaluation of Top-N recommender systems, that only takes into account where the first relevant result occurs. We get more credit for recommending an item in which user rated on the top of the rank than on the bottom of the rank. Higher is better.
3) RMSE: Root mean square error computes the mean value of all the differences squared between the true and the predicted ratings and then proceeds to calculate the square root out of the result.

## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
