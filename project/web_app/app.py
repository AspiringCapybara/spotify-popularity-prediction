import pickle      # For loading pickled model
import datetime as dt
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Import code for validation of form input (abstracted out to validation.py)
from web_app.validation import server_side_validation

# Import code for preprocessing of form input data for random forest model (abstracted out to input_preprocessing.py)
from web_app.input_preprocessing import preprocess_input

# From model.py, import DataFrame of the dataset used to train and test the random forest model
from data_processing.model import X


# Configure application
app = Flask(__name__)


# Loading random forest model (unpickling)
with open('/workspaces/218464328/project/data_processing/model.pickle', 'rb') as file:     # rb for binary read mode
    model = pickle.load(file)


@app.route("/", methods=['GET', 'POST'])
def index():
    """Show main page and form"""

    if request.method == 'GET':
        return render_template('index.html')    # Display main page

    if request.method == 'POST':

        # Form input was optimised with help from ChatGPT: More concise and non-repetitive code achieved using multiple assignment trick (i.e., tuple unpacking)
        num_playlists, release_year, release_month, mode, key, tempo, energy, danceability, valence, acousticness, liveness, speechiness = (
            request.form.get(value) for value in [
                'num_playlists', 'year', 'month', 'mode', 'key', 'bpm',
                'energy', 'danceability', 'valence',
                'acousticness', 'liveness', 'speechiness'
            ]
        )

        # Convert numerical form inputs (arrive as string) to integer or float
        num_playlists = int(num_playlists)
        release_year = int(release_year)
        tempo = int(tempo)
        energy = float(energy)
        danceability = float(danceability)
        valence = float(valence)
        acousticness = float(acousticness)
        liveness = float(liveness)
        speechiness = float(speechiness)

        predictor_values = [
            ('num_playlists', num_playlists),
            ('year', release_year),
            ('month', release_month),
            ('mode', mode),
            ('key', key),
            ('bpm', tempo),
            ('energy', energy),
            ('danceability', danceability),
            ('valence', valence),
            ('acousticness', acousticness),
            ('liveness', liveness),
            ('speechiness', speechiness)
        ]  # Keeps track of which variable each value belongs to, so that error message printed can be specific

        # Server-side validation of form input
        server_side_validation(num_playlists, release_year, release_month, mode, key,
                               tempo, energy, danceability, valence, acousticness, liveness, speechiness)

        # Preprocess data submitted by user (only those requiring preprocessing) into the app
        (
            release_month_2, release_month_3, release_month_4, release_month_5,
            release_month_6, release_month_7, release_month_8, release_month_9,
            release_month_10, release_month_11, release_month_12,
            mode,
            key_Asharp, key_B, key_Csharp, key_D, key_Dsharp, key_E,
            key_F, key_Fsharp, key_G, key_Gsharp,
            tempo, energy, danceability, valence,
            acousticness, liveness, speechiness
        ) = preprocess_input(
            release_month, mode, key,
            tempo, energy, danceability, valence,
            acousticness, liveness, speechiness
        )

        # Create a new DataFrame (with the same columns as X) to store preprocessed input data from user
        preprocessed_colnames = X.columns.tolist()
        input_data = pd.DataFrame(columns=preprocessed_colnames)

        # Insert final values into DataFrame to be inputted into model
        for_prediction = [
            ('in_spotify_playlists', num_playlists),
            ('released_year', release_year),
            ('release_month_2', release_month_2),
            ('release_month_3', release_month_3),
            ('release_month_4', release_month_4),
            ('release_month_5', release_month_5),
            ('release_month_6', release_month_6),
            ('release_month_7', release_month_7),
            ('release_month_8', release_month_8),
            ('release_month_9', release_month_9),
            ('release_month_10', release_month_10),
            ('release_month_11', release_month_11),
            ('release_month_12', release_month_12),
            ('mode', mode),
            ('key_A#', key_Asharp),
            ('key_B', key_B),
            ('key_C#', key_Csharp),
            ('key_D', key_D),
            ('key_D#', key_Dsharp),
            ('key_E', key_E),
            ('key_F', key_F),
            ('key_F#', key_Fsharp),
            ('key_G', key_G),
            ('key_G#', key_Gsharp),
            ('bpm', tempo),
            ('energy_%', energy),
            ('danceability_%', danceability),
            ('valence_%', valence),
            ('acousticness_%_log', acousticness),
            ('liveness_%_log', liveness),
            ('speechiness_%_log', speechiness)
        ]

        # Create temporary dictionary to store processed user input data
        input_values_dict = {}

        # Fill temporary dictionary with processed user input values for each predictor variable
        for colname, value in for_prediction:
            input_values_dict[colname] = value

        # Create dataframe to pass to model using temporary dictionary
        input_data = pd.DataFrame([input_values_dict])

        # Ensures column names match those of data used to train the model
        # Columns missing from current input (like columns created from one-hot encoding) are filled with 0
        # This line was from ChatGPT when debugging ValueError
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Predict number of streams of song inputted by user
        # Returns a NumPy array of predicted value
        predicted_streams_arr = model.predict(input_data)

        # Get first value in returned array and take its exponent (opposite of np.log()) since log was taken for 'streams' column in dataset
        predicted_streams = round(np.exp(predicted_streams_arr[0]))   # Round predicted number of streams shown to nearest integer

        # Render output template
        return render_template('index.html', streams=predicted_streams)


if __name__ == '__main__':
    app.run(debug=True)
