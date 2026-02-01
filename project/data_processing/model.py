import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from .data_pipeline import df_ready


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

# Model
regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=4,
    min_samples_split=5,
    random_state=42
)
regressor.fit(X_train, y_train)

# Saving trained model
with open('model.pickle', 'wb') as file:
    pickle.dump(regressor, file)
