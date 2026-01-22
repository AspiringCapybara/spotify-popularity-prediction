import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Import preprocessed data from data_processing/preprocessing.py
from data_processing.preprocessing import df_ready


# Variables that need not be preprocessed: num_playlists, release_year

def minmaxscale(input_var, column):
    """Scale data entered by user to between 0 and 1 (for selected variables) based on dataset used to train model"""

    # From preprocessed dataset, get minimum and maximum preprocessed values for variable
    # Referred to pandas documentation when writing this code
    column_min = df_ready[column].min()
    column_max = df_ready[column].max()

    scaled_input_value = (input_var - column_min) / (column_max - column_min)

    return scaled_input_value


def preprocess_input(release_month, mode, key,
                     tempo, energy, danceability, valence,
                     acousticness, liveness, speechiness
                     ):
    """Preprocess data entered by user in the web app (for random forest model to make prediction)"""

    # Initialise all variables for month to be 0
    release_month_2 = 0
    release_month_3 = 0
    release_month_4 = 0
    release_month_5 = 0
    release_month_6 = 0
    release_month_7 = 0
    release_month_8 = 0
    release_month_9 = 0
    release_month_10 = 0
    release_month_11 = 0
    release_month_12 = 0

    # One-hot encode release_month variable
    # If release_month is 'Jan', all variables for month remain as 0
    if release_month == 'Feb':
        release_month_2 = 1
    elif release_month == 'Mar':
        release_month_3 = 1
    elif release_month == 'Apr':
        release_month_4 = 1
    elif release_month == 'May':
        release_month_5 = 1
    elif release_month == 'Jun':
        release_month_6 = 1
    elif release_month == 'Jul':
        release_month_7 = 1
    elif release_month == 'Aug':
        release_month_8 = 1
    elif release_month == 'Sep':
        release_month_9 = 1
    elif release_month == 'Oct':
        release_month_10 = 1
    elif release_month == 'Nov':
        release_month_11 = 1
    elif release_month == 'Dec':
        release_month_12 = 1

    # One-hot encode mode variable
    if mode == 'Major':
        mode = 1
    elif mode == 'Minor':
        mode = 0

    # Initialise all variables for key of song to be 0
    key_Asharp = 0
    key_B = 0
    key_Csharp = 0
    key_D = 0
    key_Dsharp = 0
    key_E = 0
    key_F = 0
    key_Fsharp = 0
    key_G = 0
    key_Gsharp = 0

    # One-hot encode key variable
    # If key is 'A', all variables for key remain as 0
    if key == 'A#':
        key_Asharp = 1
    elif key == 'B':
        key_B = 1
    elif key == 'C#':
        key_Csharp = 1
    elif key == 'D':
        key_D = 1
    elif key == 'D#':
        key_Dsharp = 1
    elif key == 'E':
        key_E = 1
    elif key == 'F':
        key_F = 1
    elif key == 'F#':
        key_Fsharp = 1
    elif key == 'G':
        key_G = 1
    elif key == 'G#':
        key_Gsharp = 1

    # Log-normalise values entered by user for selected variables

    # Small constant added before logging (prevents original value from being 0 before taking log)
    epsilon = 1e-8

    speechiness = np.log(speechiness + epsilon)
    liveness = np.log(liveness + epsilon)
    acousticness = np.log(acousticness + epsilon)

    # Min-max scale values entered by user for selected variables
    tempo = minmaxscale(tempo, 'bpm')
    danceability = minmaxscale(danceability, 'danceability_%')
    valence = minmaxscale(valence, 'valence_%')
    energy = minmaxscale(energy, 'energy_%')
    speechiness = minmaxscale(speechiness, 'speechiness_%_log')
    liveness = minmaxscale(liveness, 'liveness_%_log')
    acousticness = minmaxscale(acousticness, 'acousticness_%_log')

    # Return preprocessed form input values to app.py
    return (
        release_month_2, release_month_3, release_month_4, release_month_5,
        release_month_6, release_month_7, release_month_8, release_month_9,
        release_month_10, release_month_11, release_month_12,
        mode,
        key_Asharp, key_B, key_Csharp, key_D, key_Dsharp, key_E,
        key_F, key_Fsharp, key_G, key_Gsharp,
        tempo, energy, danceability, valence,
        acousticness, liveness, speechiness
    )
