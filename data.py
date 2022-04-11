import sys
import seaborn as sns
import matplotlib.pyplot as plt


def watch_data_info(data):
    for d in data:
        print(d.head())
        print(d.info())
        print(d.describe(include='all').transpose())


def print_data(data):
    """
    Function role is to answer the question of the first part of the exercise.
    I split the first question into 3 sub-questions (because the question consists of 3 sub-questions), such that:
        quest_1 = How many unique users rated the movies?                                 --> originally question 1
        quest_2 = How many unique movies have been rated?                                 --> originally question 1
        quest_3 = How much ratings Exist in the given data?                               --> originally question 1
        quest_4 = What is the minimum and maximum number of ratings given to a movie?     --> originally question 2
        quest_5 = What is the minimum and maximum number of ratings a rating user uses?   --> originally question 3
    """
    # Delete all the "dirty value" from rating column
    clean_rows = data[0][~data[0]['rating'].isnull()]

    # quest_1: How many unique users rated the movies?
    quest_1 = len(clean_rows['userId'].unique())
    print(f'How many unique users rated the movies? --> {quest_1}')
    # quest_2: How many unique movies have been rated?
    quest_2 = len(clean_rows['movieId'].unique())
    print(f'How many unique movies have been rated? --> {quest_2}')
    # quest_3: How much ratings Exist in the given data?
    quest_3 = len(clean_rows.index)
    print(f'How much ratings Exist in the given data? --> {quest_3}')

    # Delete all the "dirty value" from movie Id column
    clean_rows = clean_rows[~clean_rows['movieId'].isnull()]

    # quest_4: What is the minimum number of ratings given to a movie?
    quest_4 = clean_rows['movieId'].value_counts().min()
    print(f'What is the minimum number of ratings given to a movie? --> {quest_4}')
    # quest_4: What is the maximum number of ratings given to a movie?
    quest_5 = clean_rows['movieId'].value_counts().max()  # for movies
    print(f'What is the maximum number of ratings given to a movie? --> {quest_5}')

    # quest 6: What is the minimum number of ratings a rating user uses?
    quest_6 = clean_rows['userId'].value_counts().min()
    print(f'What is the minimum number of ratings a rating user uses? --> {quest_6}')
    # quest 7: What is the maximum number of ratings a rating user uses?
    quest_7 = clean_rows['userId'].value_counts().max()   # for users
    print(f'What is the maximum number of ratings a rating user uses? --> {quest_7}')


def plot_data(data, plot=True):
    """
    Function role is to print a plot that will show the amount of votes for each rating value.
    """
    dictionary = {}
    # Sum the amount of voting for each rating value
    for rating in data[0].rating:
        if rating in dictionary.keys():
            dictionary[rating] += 1
        else:
            dictionary[rating] = 1

    # Create the arrays for the plot process
    ratings = []
    ratings_counter = []
    for key in sorted(dictionary.items()):
        ratings.append(key[0])
        ratings_counter.append(key[1])

    # Create the plot
    plt.bar(ratings, ratings_counter, width=0.4, color=['red', 'green'])
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel('Ratings Values')
    plt.ylabel('Ratings Occurrences')
    plt.title('plot_data function')
    plt.show()
