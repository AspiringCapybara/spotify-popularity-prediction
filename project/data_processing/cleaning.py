import pandas as pd
import os


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


if __name__ == "__main__":
    print(df_no_nulls)
