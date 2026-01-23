import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandasql import sqldf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from data_processing.cleaning import df_no_nulls


# Feature selection for model
df_cleaned = df_no_nulls.drop(['track_name', 'artist(s)_name', 'artist_count', 'released_day', 'in_spotify_charts',
                              'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts'], axis=1)


def scatter_plot_matrix(df_cleaned):
    """Visualise relationships between pairs of remaining variables using scatter plots"""

    sns.pairplot(df_cleaned)
    plt.show()


# Instrumentalness is not helpful for modelling due to low variance
df_processed = df_cleaned.drop(['instrumentalness_%'], axis=1).copy()

pd.set_option('future.no_silent_downcasting', True)
df_processed['mode'] = df_processed['mode'].replace('Major', 1)
df_processed['mode'] = df_processed['mode'].replace('Minor', 0)
df_processed['mode'] = df_processed['mode'].astype(int)


def all_rows_columns(df_processed):
    """Look at all rows and summary of columns in df_processed"""

    pd.set_option('display.max_rows', None)
    print(df_processed)
    df_processed.info()


df_processed = df_processed.drop(df_processed.index[[478]]).reset_index(
    drop=True)      # Remove row with erroneous data
df_processed = df_processed.astype({'streams': 'int64'})


def print_min_max():
    """Define SQL queries to find minimum and maximum values of numerical audio characteristics in songs"""

    query_minmaxbpm = "SELECT MIN(bpm) AS min_bpm, MAX(bpm) AS max_bpm FROM df_processed"
    minmaxbpm = sqldf(query_minmaxbpm)
    print(minmaxbpm)

    query_minmaxdance = "SELECT MIN(`danceability_%`) AS min_dance, MAX(`danceability_%`) AS max_dance FROM df_processed"
    minmaxdance = sqldf(query_minmaxdance)
    print(minmaxdance)

    query_minmaxval = "SELECT MIN(`valence_%`) AS min_val, MAX(`valence_%`) AS max_val FROM df_processed"
    minmaxval = sqldf(query_minmaxval)
    print(minmaxval)

    query_minmaxenergy = "SELECT MIN(`energy_%`) AS min_en, MAX(`energy_%`) AS max_en FROM df_processed"
    minmaxenergy = sqldf(query_minmaxenergy)
    print(minmaxenergy)

    query_minmaxacou = "SELECT MIN(`acousticness_%`) AS min_ac, MAX(`acousticness_%`) AS max_ac FROM df_processed"
    minmaxacou = sqldf(query_minmaxacou)
    print(minmaxacou)

    query_minmaxlive = "SELECT MIN(`liveness_%`) AS min_live, MAX(`liveness_%`) AS max_live FROM df_processed"
    minmaxlive = sqldf(query_minmaxlive)
    print(minmaxlive)

    query_minmaxsp = "SELECT MIN(`speechiness_%`) AS min_sp, MAX(`speechiness_%`) AS max_sp FROM df_processed"
    minmaxsp = sqldf(query_minmaxsp)
    print(minmaxsp)


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
    df_encoded = df_encoded.drop(columns=[column_name], axis=1)
    return df_encoded


df_encoded = onehotencode(df_normalised, 'key')
df_encoded = onehotencode(df_encoded, 'released_month')


def correlation_heatmap(df_encoded):
    """EDA with Correlation Heatmap"""

    corr_matrix = df_encoded.corr(method='pearson', numeric_only=True)
    sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=corr_matrix.columns,
                yticklabels=corr_matrix.columns, vmin=0.0, vmax=1.0)
    plt.show()


def calculate_VIF(df_encoded):
    """Calculate VIF for each predictor variable to quantify the extent of multicollinearity"""

    df_predictors_only = df_encoded.drop(['streams'], axis=1).copy()
    X = sm.add_constant(df_predictors_only)

    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    print(vif_data)


# Min-max scale selected variables
scaler = MinMaxScaler()
pd.set_option('display.max_rows', None)
scaler_input = df_encoded.loc[:, ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'speechiness_%_log',
                                  'liveness_%_log', 'acousticness_%_log']]
df_scaled = pd.DataFrame(scaler.fit_transform(scaler_input))


def check_data_type(df_scaled):
    """Check data type of column names"""

    print(type(df_scaled.columns.tolist()[0]))  # Column names are int


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
