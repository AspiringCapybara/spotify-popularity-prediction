import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


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
df_no_nulls = df.dropna(how='any')

# Feature selection for model
df_cleaned = df_no_nulls.drop(['track_name', 'artist(s)_name', 'artist_count', 'released_day', 'in_spotify_charts',
                              'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts'], axis=1)

# Instrumentalness is not helpful for modelling due to low variance
df_processed = df_cleaned.drop(['instrumentalness_%'], axis=1).copy()

pd.set_option('future.no_silent_downcasting', True)
df_processed['mode'] = df_processed['mode'].replace('Major', 1)
df_processed['mode'] = df_processed['mode'].replace('Minor', 0)
df_processed['mode'] = df_processed['mode'].astype(int)

df_processed = df_processed.drop(df_processed.index[[478]]).reset_index(
    drop=True)      # Remove row with erroneous data
df_processed = df_processed.astype({'streams': 'int64'})

# Log-normalise selected features
epsilon = 1e-8
df_processed['speechiness_%_log'] = np.log(
    df_processed['speechiness_%'] + epsilon)
df_processed['liveness_%_log'] = np.log(df_processed['liveness_%'] + epsilon)
df_processed['acousticness_%_log'] = np.log(
    df_processed['acousticness_%'] + epsilon)
df_normalised = df_processed.drop(
    ['speechiness_%', 'liveness_%', 'acousticness_%'], axis=1).copy()


def onehotencode(df, column_name):
    """One-hot encode a variable"""

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = encoder.fit_transform(df[[column_name]])
    feature_names = encoder.get_feature_names_out(input_features=[column_name])
    encoded_data_df = pd.DataFrame(
        data=encoded_data, columns=feature_names, index=df.index)
    df_encoded = pd.concat(objs=[df, encoded_data_df], axis=1)
    df_encoded = df_encoded.drop(columns=[column_name])
    return df_encoded


df_encoded = onehotencode(df_normalised, 'key')
df_encoded = onehotencode(df_encoded, 'released_month')

# Min-max scale selected variables
scaler = MinMaxScaler()
pd.set_option('display.max_rows', None)
scaler_input = df_encoded.loc[:, ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'speechiness_%_log',
                                  'liveness_%_log', 'acousticness_%_log']]
df_scaled = pd.DataFrame(scaler.fit_transform(scaler_input))

variable_names_list = scaler_input.columns.tolist()
variable_names_dict = {}

for i in range(len(variable_names_list)):
    variable_names_dict[i] = variable_names_list[i]

df_scaled = df_scaled.rename(columns=variable_names_dict)
columns_to_replace = scaler_input.columns.tolist()
df_temp = df_encoded.drop(columns_to_replace, axis=1)
df_ready = pd.concat(objs=[df_temp, df_scaled], axis=1)

df_ready['streams_raw'] = df_ready['streams']
df_ready['streams'] = np.log(df_ready['streams'])
