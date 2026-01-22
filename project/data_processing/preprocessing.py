import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pandasql import sqldf  # Reference: https://docs.kanaries.net/topics/Pandas/pandasql
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_processing.cleaning import df_no_nulls    # Import variable df_no_nulls from cleaning.py


# Feature selection: Removing columns that will not be used in model
# This code was written with reference to Pandas' documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop
df_cleaned = df_no_nulls.drop(['track_name', 'artist(s)_name', 'artist_count', 'released_day', 'in_spotify_charts',
                              'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts'], axis=1)


def scatter_plot_matrix(df_cleaned):
    """Visualise relationships between pairs of remaining variables using scatter plots"""
    # Reference to Seaborn's documentation was made when writing the code: https://seaborn.pydata.org/generated/seaborn.pairplot.html
    sns.pairplot(df_cleaned)
    plt.show()


# Drop "instrumentalness" column since most points have "instrumentalness_%" = 0 - not helpful for modelling due to low variance
# .copy() added after debugging unpredictable behaviour with ChatGPT
# .copy() ensures that the original DataFrame for df_cleaned isn't modified (if df_processed is later modified), thus preventing unintentional changes being made to data
df_processed = df_cleaned.drop(['instrumentalness_%'], axis=1).copy()

# Raise error (instead of FutureWarning) if silent downcasting (i.e., automatic change of data type without informing user) occurs
# This line of code is from Google search
pd.set_option('future.no_silent_downcasting', True)
# Convert "mode" to '1' and '0' since "mode" column contains boolean
df_processed['mode'] = df_processed['mode'].replace('Major', 1)
df_processed['mode'] = df_processed['mode'].replace('Minor', 0)
# Manage data type conversions so behaviour will be predictable when implicit downcasting removed (This line is from Google search)
df_processed['mode'] = df_processed['mode'].astype(int)


def all_rows_columns(df_processed):
    """Look at all rows and summary of columns in df_processed"""

    pd.set_option('display.max_rows', None)     # Displays all rows
    print(df_processed)
    df_processed.info()     # Prints summary of columns and data types


# Drop row at position 478 due to erroneous data in "streams" column
# Used ChatGPT (.reset_index(drop=True)) to debug issue of wrong row being dropped due to indexing issue
df_processed = df_processed.drop(df_processed.index[[478]]).reset_index(drop=True)

# Convert data type of values in "streams" column from object to integer
df_processed = df_processed.astype({'streams': 'int64'})

# Overview of dataset now:
# 816 rows and 10 columns


def print_min_max():
    """Define SQL queries to find minimum and maximum values of numerical audio characteristics in songs"""

    query_minmaxbpm = "SELECT MIN(bpm) AS min_bpm, MAX(bpm) AS max_bpm FROM df_processed"
    minmaxbpm = sqldf(query_minmaxbpm)
    print(minmaxbpm)

    # Used ChatGPT to debug error caused by "%" character in column names
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

    # Summary:
    # Min BPM: 65, Max BPM: 206
    # Min danceability_%: 23, Max danceability_%: 96
    # Min valence_%: 4, Max valence_%: 97
    # Min energy_%: 14, Max energy_%: 97
    # Min acousticness_%: 0, Max acousticness_%: 97
    # Min liveness_%: 3, Max liveness_%: 97
    # Min speechiness_%: 2, Max speechiness_%: 64


# Take log of these skewed features, since they have log-normal distribuions (based on sns.pairplot())
# Google search (log-normalisation) and knowledge from Coursera's Google Advanced Data Analytics Professional Certificate was used in writing these code (excluding epsilon)
# ChatGPT was used to fix problem of some log-normalised values becoming -inf (Negative infinity) due to original value being 0 (which is valid)
epsilon = 1e-8     # Small constant added before logging
df_processed['speechiness_%_log'] = np.log(df_processed['speechiness_%'] + epsilon)
df_processed['liveness_%_log'] = np.log(df_processed['liveness_%'] + epsilon)
df_processed['acousticness_%_log'] = np.log(df_processed['acousticness_%'] + epsilon)

# Drop original columns that are not log-normalised
# .copy() prevents unpredictable behaviour caused by DataFrame being wrongly overwritten
df_normalised = df_processed.drop(['speechiness_%', 'liveness_%', 'acousticness_%'], axis=1).copy()


def onehotencode(df, column_name):
    """One-hot encode variable"""
    # Reference to Sci-kit learn documentation was made when writing the following code: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder.fit_transform
    # ChatGPT was used alongside documentation in understanding how to implement OneHotEncoder
    # Debugged NameError with ChatGPT's help

    # sparse_output=False so a sparse matrix in CSR format doesn't get returned
    # Avoiding "Dummy variable trap": drop='first' is used so we don't introduce perfect multicollinearity (redundant information)
    encoder = OneHotEncoder(drop='first', sparse_output=False)

    # Returns transformed array
    encoded_data = encoder.fit_transform(df[[column_name]])

    # Returns transformed feature names as output (which is n-dimensional array of str objects)
    # Passes list of strings to manually specify column names
    feature_names = encoder.get_feature_names_out(input_features=[column_name])

    # Create a DataFrame of the transformed features (i.e., convert transformed array to DataFrame)
    encoded_data_df = pd.DataFrame(data=encoded_data, columns=feature_names, index=df.index)

    # Combine transformed features (columns) with original data
    df_encoded = pd.concat(objs=[df, encoded_data_df], axis=1)

    # Since one-hot encoding done, drop original column to avoid redundancy
    df_encoded = df_encoded.drop(columns=[column_name], axis=1)

    return df_encoded


df_encoded = onehotencode(df_normalised, 'key')
df_encoded = onehotencode(df_encoded, 'released_month')


def correlation_heatmap(df_encoded):
    """EDA with Correlation Heatmap"""
    # Referred to Google and documentation for Pandas and Seaborn

    # Calculate correlation matrix (method is standard correlation coefficient)
    # Include only float, int or boolean data
    corr_matrix = df_encoded.corr(method='pearson', numeric_only=True)

    # Plot correlation heatmap using correlation matrix (correlation coefficients set to between 0 and 1 inclusive)
    sns.heatmap(corr_matrix, cmap='coolwarm', xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns ,vmin=0.0, vmax=1.0)      # All column names are shown
    plt.show()

    # Result: No 2 predictor variables have correlation > 0.5, suggesting likely there is no significant multicollinearity (to be confirmed using VIF)


def calculate_VIF(df_encoded):
    """Calculate VIF for each predictor variable to quantify the extent of multicollinearity"""
    # Reference made to notes I took down from Coursera's Google Advanced Data Analytics Professional Certificate

    # Create a copy of df_encoded but keep only the predictor variables
    df_predictors_only = df_encoded.drop(['streams'], axis=1).copy()
    # Adds a constant to the predictors matrix so the intercept term is included, preventing misleading VIF values or computational errors
    X = sm.add_constant(df_predictors_only)

    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    # Loop through each column and calculate its VIF
    # X.shape[1] returns number of columns in the DataFrame X
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)

    # Result: All independent variables have VIF value of between 1.05 and 2.33 - indicates moderate correlation, which is acceptable


# Reference was made to Sci-kit learn's documentation on MinMaxScaler
# Scale variables
scaler = MinMaxScaler()

pd.set_option('display.max_rows', None)     # Displays all rows

# Fit scaler to data, then transform it
# df_scaled stores scaled values as a DataFrame
# Input data is selected (using labels) to be all rows and columns from df_encoded, excluding 'streams' (the dependent variable) and boolean columns
scaler_input = df_encoded.loc[:, ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'speechiness_%_log',
                                   'liveness_%_log', 'acousticness_%_log']]    # .loc voids problems with index shifting if more columns were to be included / dropped

# Scaled values in each column (now all values are between 0 and 1 inclusive)
df_scaled = pd.DataFrame(scaler.fit_transform(scaler_input))


"""Rename columns in df_scaled to reflect what variable each column represents"""


def check_data_type(df_scaled):
    """Check data type of column names"""

    print(type(df_scaled.columns.tolist()[0]))  # Column names are int


# Referred to Pandas documentation (.columns) and obtained help from ChatGPT (.tolist)
# Retrieve column names from scaler_input as a list
variable_names_list = scaler_input.columns.tolist()
variable_names_dict = {}    # Creating mapper (dict) for df_scaled.rename()

# Loop through columns in scaler_input
for i in range(len(variable_names_list)):
    # Set each value as the correct column name (based on scaler_input)
    variable_names_dict[i] = variable_names_list[i]

# Rename all columns in df_scaled from indexes to variable names
df_scaled = df_scaled.rename(columns=variable_names_dict)

# Create df_ready by replacing the unscaled columns in df_encoded with the scaled ones (drop old then add new back in)
columns_to_replace = scaler_input.columns.tolist()     # List of columns to drop
df_temp = df_encoded.drop(columns_to_replace, axis=1)
df_ready = pd.concat(objs=[df_temp, df_scaled], axis=1)    # Join back the scaled data


"""Initially, when linear model in model.py was deployed, R-squared score was 0.0399, meaning model only explained about 4% of the variance in the dependent variable.
A problem could be a mismatch of scale between dependent variable (0 to over 1 billion) and predictor variables (between 0 and 1).
Thus, log-normalising the "streams" variable during preprocessing could help in fixing the issue."""

# Create a new "streams_raw" column to preserve the original values for number of streams (that are not log-normalised)
df_ready['streams_raw'] = df_ready['streams']
df_ready['streams'] = np.log(df_ready['streams'])      # Log-normalisation
