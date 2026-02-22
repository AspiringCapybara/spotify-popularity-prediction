import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from flask import Flask, request, render_template
from web_app.validation import server_side_validation
from web_app.input_preprocessing import preprocess_input

# Unused import added to ensure pickle can resolve import
from data_processing.log_transformer import LogFeatureTransformer


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "data_processing" / "artifacts" / "model.pickle"

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

        server_side_validation(
            num_playlists, release_year, release_month,
            mode, key, tempo, energy, danceability,
            valence, acousticness, liveness, speechiness
        )

        input_df = preprocess_input(
            num_playlists, release_year, release_month,
            mode, key, tempo, energy, danceability,
            valence, acousticness, liveness, speechiness
        )

        predicted_streams_arr = model.predict(input_df)
        predicted_streams = round(np.exp(predicted_streams_arr[0]))

        return render_template('index.html', streams=predicted_streams)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
