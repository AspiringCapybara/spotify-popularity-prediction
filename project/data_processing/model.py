import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from data_processing.data_pipeline import df_ready


df_all_predictors = df_ready.drop(['streams', 'streams_raw'], axis=1).copy()
feature_list = df_all_predictors.columns.tolist()

X = df_ready[feature_list]
y = df_ready['streams']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)


def preview_dataset_table(X_test):
    """View first 5 rows of selected portion of data"""

    print(X_test.head(5))


# Model
regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    min_samples_split=5,
    random_state=42
)
regressor.fit(X_train, y_train)


def cross_val():
    """
    Performing a one-time hyperparameter tuning using GridSearchCV with 5-fold cross validation.
    Prints best set of hyperparameters and its R2 score.
    Not used in final training.
    """

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
        scoring='r2',
        cv=5
    )

    grid_search.fit(X_train, y_train)
    best_estimator_number = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters:", best_params)
    print("Best R2 score:", best_score)


# Saving trained model
with open('model.pickle', 'wb') as file:
    pickle.dump(regressor, file)


if __name__ == '__main__':
    print(regressor.score(X_test, y_test))      # R-squared
