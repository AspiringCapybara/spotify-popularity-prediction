import pandas as pd


def preprocess_input(num_playlists, release_year, release_month,
                     mode, key, tempo, energy, danceability,
                     valence, acousticness, liveness, speechiness):
    """Convert raw user input into a DataFrame that matches the structure of the training set's features"""

    if mode == 'Major':
        mode = 1
    elif mode == 'Minor':
        mode = 0

    input_dict = {
        'in_spotify_playlists': int(num_playlists),
        'released_year': int(release_year),
        'released_month': int(release_month),
        'mode': int(mode),
        'key': key,
        'bpm': int(tempo),
        'energy_%': float(energy),
        'danceability_%': float(danceability),
        'valence_%': float(valence),
        'acousticness_%': float(acousticness),
        'liveness_%': float(liveness),
        'speechiness_%': float(speechiness),
    }

    input_df = pd.DataFrame([input_dict])

    return input_df
