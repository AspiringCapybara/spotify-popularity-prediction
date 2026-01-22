import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor     # Initially tried LinearRegression

from data_processing.preprocessing import df_ready     # DataFrame of dataset ready for use in the model

# Used knowledge obtained from Coursera's Google Advanced Data Analytics Professional Certificate
# Referred to Sci-kit learn's documentation on train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# Create a copy of df_ready but keep only the predictor variables
df_all_predictors = df_ready.drop(['streams', 'streams_raw'], axis=1).copy()

# Get list of independent variables
feature_list = df_all_predictors.columns.tolist()

# Define features (i.e., independent variables / predictor variables) and target (dependent variable) - separate X and y variables
# Consulted ChatGPT on how to define multiple variables as X
X = df_ready[feature_list]
y = df_ready['streams']

# Split data into random training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,     # Include 30% of data in test set (i.e, use 70% of data for model training)
    random_state=42,    # Setting a seed to control how data is shuffled, enabling others to replicate findings
)


def preview_dataset_table(X_test):
    """View first 5 rows of selected portion of data"""

    print(X_test.head(5))


# Code from Sci-kit learn's documentation on RandomForestRegressor
regressor = RandomForestRegressor(
    n_estimators=200,   # Number of decision trees (base learners) in ensemble
    max_depth=15,   # Maximum depth of the tree
    min_samples_leaf=2,    # Minimum number of samples required to be at a leaf node
    min_samples_split=5,    # Minimum number of samples required to split an internal node
    random_state=42    # Random seed
)
regressor.fit(X_train, y_train)     # Fit model on training data


if __name__ == '__main__':
    # This line runs only if model.py was run directly (not when it's imported into other files)
    # Print R-squared score of model for evaluation
    print(regressor.score(X_test, y_test))

# First time running model after changing to RandomForestRegressor, R-squared score is 0.03438.
# Realised that features included may be giving too little predictive power so decided to not drop "released_year", "released_month"
# R-squared score improved to 0.3762
# Using 200 trees instead of 100 -> R-squared improved slightly to 0.3779
# Using 300 trees instead of 100 -> R-squared improved slightly to 0.3801
# Decided not to drop "in_spotify_playlists", "in_spotify_charts" -> R-squared improved significantly to 0.5352
# Since we are predicting Spotify streams, the number of times a song appeared in spotify playlists and charts would likely have a big impact on the number of streams
# After 2 rounds of performing GridSearchCV to tune hyperparameters, R-squared based on test data is 0.5378


def cross_val():
    """Performing cross-validation to find best set of hyperparameters using GridSearchCV"""
    # Adapted from example code in Google Advanced Data Analytics Professional Certificate
    # Referred to RandomForestRegressor and GridSearchCV documentation

    # Combinations of hyperparameter values to try
    param_grid = {
        'max_depth': [5, 10, 15, None],
        'n_estimators': [100, 200],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [5, 10]
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='r2',    # Scoring metric used
        cv=5   # 5-fold cross-validation
    )

    # Run grid search with all sets of parameters
    grid_search.fit(X_train, y_train)

    # Retrieve the best model and score
    best_estimator_number = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters:", best_params)
    print("Best R2 score:", best_score)

    # Parameters tried on first CV attempt:
    """param_grid = {
        'max_depth': [4, 6, 8, 10, None],
        'n_estimators': [100, 200, 300],
        'min_samples_leaf': [1, 5, 10, 15]
    }"""
    # Results (first try):
    # Best parameters {'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 200}
    # Best R2 score (to 4 d.p.): 0.7644 - suggests a bit of overfitting to training data

    # Parameters tried on second CV attempt:
    """    param_grid = {
        'max_depth': [5, 10, 15, None],
        'n_estimators': [100, 200],
        'min_samples_leaf': [2, 4],
        'min_samples_split': [5, 10]
    }"""
    # Results (second try):
    # Best parameters {'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
    # Best R2 score (to 4 d.p.): 0.7644 - suggests a bit of overfitting to training data
    # Consulted ChatGPT - obtained conclusion that this overfitting is normal for random forests on small datasets and high variance data (e.g., music)

"""Pickling (saving) trained model"""
# Adapted code from Google Data Analytics Professional Certificate

# Saving model (pickling) - model will be loaded (unpickling) in app.py
# with open closes the file automatically after the file is done being used
with open('model.pickle', 'wb') as file:    # wb for binary write mode
    pickle.dump(regressor, file)

