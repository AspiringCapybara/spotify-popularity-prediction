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
    ]  # Keeps track of which variable each value belongs to, so that error message printed can be specific

    # Handling null input values (server-side validation)
    for var, value in predictor_values:    # Unpacks the tuples in predictor_values
        if value is None or value == "":   # Using ChatGPT, debugged problem of input value of 0 being assumed as falsy
            error_message = f"{var} cannot be empty!"
            raise Exception(error_message)

    # Validating number of playlists song is in
    # Handling non-integer input
    try:
        num_playlists = int(num_playlists)
    except ValueError:
        print("Number of playlists must be an integer.")

    # If number of playlists song is in is outside of the training data's range
    if num_playlists < 31 or num_playlists > 52898:
        raise ValueError("Number of playlists entered is out of range.")

    # Validating release year of song
    # Handling non-integer input
    try:
        release_year = int(release_year)
    except ValueError:
        print("Release year must be an integer.")

    # Handling invalid year inputted
    # Referred to datetime library's documentation - https://docs.python.org/3/library/datetime.html
    current_year = dt.datetime.now().year
    if release_year > current_year or release_year < 1930:     # Songs in training data are released from 1930 onwards
        raise Exception("Release year must be between 1930 and current year.")

    # I=User input for released_month in index.html is passed to web app as integer from 1 to 12
    try:
        release_month = int(release_month)
    except ValueError:
        raise Exception("Release month must be one of the options in the drop-down list.")

    if release_month < 1 or release_month > 12:
        raise Exception("Invalid month entered.")

    # Available selections for mode of song
    modes = ('Major', 'Minor')
    if mode not in modes:
        raise Exception("Mode must be 'Major' or 'Minor'.")

    # Available selections for key of song
    keys = ('A#', 'B', 'C#', 'D', 'D#', 'E', 'F', 'G', 'G#')
    if key not in keys:
        raise Exception("Invalid key entered.")

    # Validating tempo of song
    # Handling non-integer input
    try:
        tempo = int(tempo)
    except ValueError:
        print("Song tempo (in bpm) must be an integer.")

    # If tempo is outside of the training data's range
    if tempo < 65 or tempo > 206:
        raise ValueError("Tempo entered is out of range.")

    # Validating energy of song
    # Handling non-integer input
    try:
        energy = int(energy)
    except ValueError:
        print("Energy (in %) must be an integer.")

    # If energy is outside of the training data's range
    if energy < 14 or energy > 97:
        raise ValueError("Energy entered is out of range.")

    # Validating danceability of song
    # Handling non-integer input
    try:
        danceability = int(danceability)
    except ValueError:
        print("Danceability (in %) must be an integer.")

    # If danceability is outside of the training data's range
    if danceability < 23 or danceability > 96:
        raise ValueError("Danceability entered is out of range.")

    # Validating valence of song
    # Handling non-integer input
    try:
        valence = int(valence)
    except ValueError:
        print("Valence (in %) must be an integer.")

    # If valence is outside of the training data's range
    if valence < 4 or valence > 97:
        raise ValueError("Valence entered is out of range.")

    # Validating acousticness of song
    # Handling non-integer input
    try:
        acousticness = int(acousticness)
    except ValueError:
        print("Acousticness (in %) must be an integer.")

    # If acousticness is outside of the training data's range
    if acousticness < 0 or acousticness > 97:
        raise ValueError("Acousticness entered is out of range.")

    # Validating liveness of song
    # Handling non-integer input
    try:
        liveness = int(liveness)
    except ValueError:
        print("Liveness (in %) must be an integer.")

    # If acousticness is outside of the training data's range
    if liveness < 3 or liveness > 97:
        raise ValueError("Liveness entered is out of range.")

    # Validating speechiness of song
    # Handling non-integer input
    try:
        speechiness = int(speechiness)
    except ValueError:
        print("Speechiness (in %) must be an integer.")

    # If speechiness is outside of the training data's range
    if speechiness < 2 or speechiness > 64:
        raise ValueError("Speechiness entered is out of range.")
