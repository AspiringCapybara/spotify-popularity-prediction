import pandas as pd
import os

# This code (os) was from ChatGPT - debugs FileNotFoundError when running Flask from project/ folder
# This code ensures the CSV file (dataset) is always loaded using a path relative to data_processing/

# Get the directory of the current script file
current_dir = os.path.dirname(__file__)

# Build the full path to the CSV
csv_path = os.path.join(current_dir, 'popular_spotify_songs.csv')


"""Loading the dataset"""   # Source of data: Kaggle - https://www.kaggle.com/datasets/ahmadrazakashif/spotify-popularity-songs/data?select=Popular_Spotify_Songs+%281%29.csv

# Reference to Pandas' documentation was made: https://pandas.pydata.org/docs/reference/frame.html#
# Trying common encodings (i.e., representations of the text in the file) to successfully read CSV file containing data -- Solution to UnicodeDecodeError derived from Google search
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_path, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='cp1252')

# Overview of raw dataset - using df.info():
# 953 rows and 24 columns, nulls in "in_shazam_charts" and "key"

"""Cleaning the data"""

# Remove duplicated song names if any, keeping only the first song with the same name
df.drop_duplicates('track_name')
df_no_nulls = df.dropna(how='any')  # Remove rows containing null values, if any

# Overview of dataset after dropping duplicates and nulls:
# 817 rows and 24 columns

# Execute this only when cleaning.py is run (and not when it's imported into another file)
if __name__ == "__main__":
    print(df_no_nulls)  # Print dataframe
