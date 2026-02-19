import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'popular_spotify_songs.csv')

# Loading the dataset
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='cp1252')


# Cleaning the data
df.drop_duplicates('track_name')
df = df.dropna(how='any')

# Feature selection for model
df = df.drop(['track_name', 'artist(s)_name', 'artist_count', 'released_day', 'in_spotify_charts',
              'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts'], axis=1)

# Instrumentalness is not helpful for modelling due to low variance
df = df.drop(['instrumentalness_%'], axis=1).copy()

# Convert mode to numeric
pd.set_option('future.no_silent_downcasting', True)
df['mode'] = df['mode'].replace('Major', 1)
df['mode'] = df['mode'].replace('Minor', 0)
df['mode'] = df['mode'].astype(int)

if len(df) > 478:
    df = df.drop(df.index[[478]]).reset_index(
        drop=True)      # Remove row with erroneous data

# Log-normalise target variable
df['streams_raw'] = df['streams']
df['streams'] = np.log(df['streams'])

# Log-normalise selected features
epsilon = 1e-8
df['speechiness_%_log'] = np.log(df['speechiness_%'] + epsilon)
df['liveness_%_log'] = np.log(df['liveness_%'] + epsilon)
df['acousticness_%_log'] = np.log(df['acousticness_%'] + epsilon)
df = df.drop(['speechiness_%', 'liveness_%', 'acousticness_%'], axis=1).copy()

# Define features
X = df.drop('streams', axis=1)
y = df['streams']

# Features to be preprocessed
numerical_features = ['bpm', 'speechiness_%_log', 'liveness_%_log',
                      'acousticness_%_log', 'danceability_%', 'valence_%', 'energy_%']
categorical_features = ['key', 'released_month']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False,
         handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=4,
        min_samples_split=5,
        random_state=42
    ))
])

# Model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)

pipeline.fit(X_train, y_train)

# Save trained model
with open('model.pickle', 'wb') as file:
    pickle.dump(pipeline, file)
