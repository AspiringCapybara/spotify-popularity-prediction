import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from flask import Flask, request, render_template
from web_app.validation import server_side_validation
from web_app.input_preprocessing import preprocess_input
from data_processing.model import X


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data_processing" / "model.pickle"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=['GET', 'POST'])
def index():
    """Show main page and form"""

    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        num_playlists, release_year, release_month, mode, key, tempo, energy, danceability, valence, acousticness, liveness, speechiness = (
            request.form.get(value) for value in [
                'num_playlists', 'year', 'month', 'mode', 'key', 'bpm',
                'energy', 'danceability', 'valence',
                'acousticness', 'liveness', 'speechiness'
            ]
        )

        num_playlists = int(num_playlists)
        release_year = int(release_year)
        tempo = int(tempo)
        energy = float(energy)
        danceability = float(danceability)
        valence = float(valence)
        acousticness = float(acousticness)
        liveness = float(liveness)
        speechiness = float(speechiness)

        server_side_validation(num_playlists, release_year, release_month, mode, key,
                               tempo, energy, danceability, valence, acousticness, liveness, speechiness)

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

        preprocessed_colnames = X.columns.tolist()
        input_data = pd.DataFrame(columns=preprocessed_colnames)
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

        input_values_dict = {}
        for colname, value in for_prediction:
            input_values_dict[colname] = value
        input_data = pd.DataFrame([input_values_dict])
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        predicted_streams_arr = model.predict(input_data)
        predicted_streams = round(np.exp(predicted_streams_arr[0]))

        return render_template('index.html', streams=predicted_streams)


if __name__ == '__main__':
    app.run(debug=True)
