import datetime as dt


def server_side_validation(num_playlists, release_year, release_month, mode, key, tempo, energy, danceability, valence, acousticness, liveness, speechiness):
    """Server-side validation of form input"""

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
    ]

    for var, value in predictor_values:
        if value is None or value == "":
            error_message = f"{var} cannot be empty!"
            raise Exception(error_message)

    try:
        num_playlists = int(num_playlists)
    except ValueError:
        print("Number of playlists must be an integer.")
    if num_playlists < 31 or num_playlists > 52898:
        raise ValueError("Number of playlists entered is out of range.")

    try:
        release_year = int(release_year)
    except ValueError:
        print("Release year must be an integer.")
    current_year = dt.datetime.now().year
    # Songs in training data are released from 1930 onwards
    if release_year > current_year or release_year < 1930:
        raise Exception("Release year must be between 1930 and current year.")

    try:
        release_month = int(release_month)
    except ValueError:
        raise Exception(
            "Release month must be one of the options in the drop-down list.")
    if release_month < 1 or release_month > 12:
        raise Exception("Invalid month entered.")

    modes = ('Major', 'Minor')
    if mode not in modes:
        raise Exception("Mode must be 'Major' or 'Minor'.")

    keys = ('A#', 'B', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#')
    if key not in keys:
        raise Exception("Invalid key entered.")

    try:
        tempo = int(tempo)
    except ValueError:
        print("Song tempo (in bpm) must be an integer.")
    if tempo < 65 or tempo > 206:
        raise ValueError("Tempo entered is out of range.")

    try:
        energy = int(energy)
    except ValueError:
        print("Energy (in %) must be an integer.")
    if energy < 14 or energy > 97:
        raise ValueError("Energy entered is out of range.")

    try:
        danceability = int(danceability)
    except ValueError:
        print("Danceability (in %) must be an integer.")
    if danceability < 23 or danceability > 96:
        raise ValueError("Danceability entered is out of range.")

    try:
        valence = int(valence)
    except ValueError:
        print("Valence (in %) must be an integer.")
    if valence < 4 or valence > 97:
        raise ValueError("Valence entered is out of range.")

    try:
        acousticness = int(acousticness)
    except ValueError:
        print("Acousticness (in %) must be an integer.")
    if acousticness < 0 or acousticness > 97:
        raise ValueError("Acousticness entered is out of range.")

    try:
        liveness = int(liveness)
    except ValueError:
        print("Liveness (in %) must be an integer.")
    if liveness < 3 or liveness > 97:
        raise ValueError("Liveness entered is out of range.")

    try:
        speechiness = int(speechiness)
    except ValueError:
        print("Speechiness (in %) must be an integer.")
    if speechiness < 2 or speechiness > 64:
        raise ValueError("Speechiness entered is out of range.")
