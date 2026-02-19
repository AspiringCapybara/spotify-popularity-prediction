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

if len(df) > 478:     # Safeguard added so code works for both full and dummy dataset
    df = df.drop(df.index[[478]]).reset_index(
        drop=True)      # Remove row with erroneous data
else:
    df = df.reset_index(drop=True)
df = df.astype({'streams': 'int64'})

# Log-normalise selected features
epsilon = 1e-8
df['speechiness_%_log'] = np.log(
    df['speechiness_%'] + epsilon)
df['liveness_%_log'] = np.log(
    df['liveness_%'] + epsilon)
df['acousticness_%_log'] = np.log(
    df['acousticness_%'] + epsilon)
df = df.drop(
    ['speechiness_%', 'liveness_%', 'acousticness_%'], axis=1).copy()


def onehotencode(df, column_name):
    """One-hot encode a variable"""

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = encoder.fit_transform(df[[column_name]])
    feature_names = encoder.get_feature_names_out(
        input_features=[column_name])
    encoded_data_df = pd.DataFrame(
        data=encoded_data, columns=feature_names, index=df.index)
    df = pd.concat(objs=[df, encoded_data_df], axis=1)
    df = df.drop(columns=[column_name])
    return df


df = onehotencode(df, 'key')
df = onehotencode(df, 'released_month')

# Min-max scale selected variables
scaler = MinMaxScaler()
pd.set_option('display.max_rows', None)
scaler_input = df.loc[:, ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'speechiness_%_log',
                          'liveness_%_log', 'acousticness_%_log']]
df = pd.DataFrame(scaler.fit_transform(scaler_input))

variable_names_list = scaler_input.columns.tolist()
variable_names_dict = {}

for i in range(len(variable_names_list)):
    variable_names_dict[i] = variable_names_list[i]

df = df.rename(columns=variable_names_dict)
columns_to_replace = scaler_input.columns.tolist()
df = df.drop(columns_to_replace, axis=1)
df = pd.concat(objs=[df, df], axis=1)

df['streams_raw'] = df['streams']
df['streams'] = np.log(df['streams'])
