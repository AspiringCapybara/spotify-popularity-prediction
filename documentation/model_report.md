# Model Development and Evaluation Report

## 1. Objective

To model Spotify stream counts using structured audio features, release timing and playlist exposure metadata, with emphasis on:

- Understanding limitations of predicting stream counts
- Performance evaluation across popularity ranges
- Identifying dominant features driving prediction

This document focuses strictly on modeling methodology and evaluation.

---

## 2. Data Preprocessing Pipeline

All preprocessing was implemented using a unified `Scikit-learn` Pipeline to ensure consistency between training and inference.

### Cleaning Steps

- Dropped rows containing missing and invalid values
- Removed features that are not relevant to the prediction task (e.g., `in_deezer_charts`, `artist_count`)
- Removed `instrumentalness_%` due to near-zero variance
- Converted `mode` from categorical string to binary indicator

### Target Transformation

Stream counts exhibit a heavy right-skewed distribution. 

Thus, a log transformation was applied to stabilise variance and improve regression performance:
<p style="text-align: center;">y = log(streams)</p>

---

## 3. Feature Engineering

Features were grouped and processed as follows:

### Numerical Features

These were scaled using `MinMaxScaler`.

- `bpm`
- `speechiness_%`
- `liveness_%`
- `acousticness_%`
- `danceability_%`
- `valence_%`
- `energy_%`

### Categorical Features

These were encoded using `OneHotEncoder(drop='first')`.

### Pass-through Features

No processing was needed for these features.

- `in_spotify_playlists`
- `released_year`

### Custom Transformation

A custom `LogFeatureTransformer` was implemented to apply log-normalisation to skewed percentage features within the pipeline.

---

## 4. Model Selection

A `RandomForestRegressor` was selected due to:

- Ability to capture non-linear patterns well
- Robustness to variations in feature scaling
- Good performance under limited hyperparameter tuning

The following hyperparameters were selected using `GridSearchCV`:

- `n_estimators = 100`
- `max_depth = 15`
- `min_samples_leaf = 4`
- `min_samples_split = 5`
- `random_state = 42`

---

## 5. Evaluation Methodology

Dataset was split into 70% training and 30% testing. 

All preprocessing steps were fit exclusively on the training set within the pipeline to prevent data leakage.

Evaluation metrics:

- R2
- MAE
- RMSE

Diagnostic analysis included:

- Predicted vs actual (log scale)
- Residual plots
- Absolute error by stream count quantiles

---

## 6. Performance Summary

[Predicted vs Actual (log-log scale)](visualisations/predicted_vs_actual.png)

Final test set performance:

- R2 ≈ 0.55
- MAE (log scale) ≈ 0.43
- RMSE (log scale) ≈ 0.89

Performance is strongest in mid-range stream counts and degrades for extreme upper-tail stream counts.

Given the absence of features capturing external factors (e.g., artist history, marketing strategies), an R2 of 0.55 indicates that structured audio features and playlist exposure alone explain over half the variance in log-transformed stream counts.

---

## 7. Error Behaviour Across Popularity Levels

[Absolute Error by Quantile](visualisations/abs_error_by_quantile.png)

Quantile-based analysis shows:

- Lower error variance for moderately popular songs
- Systemic under-prediction in upper quantiles
- Compression in predicted values at extreme stream counts

This suggests that extreme popularity is influenced by variables not captured in the dataset, resulting in prediction compression at the upper tail.

---

## 8. Feature Contribution Analysis

[Permutation Feature Importance](visualisations/permut_feature_importance.png)

Permutation feature importance indicates that:

- Playlist exposure is the dominant predictive feature
- Audio features provide incremental but smaller contributions
- Removing playlist exposure feature substantially degrades R2

The ablation experiment confirms that while metadata alone explains a meaningful portion of variance, exposure-related features overwhelmingly drive most predictive power. This indicates that song visibility on a platform is a stronger determinant of stream count than intrinsic audio characteristics and release timing.

---

## 9. Model Generalisation and Stability

- Random seed fixed for reproducibility
- Pipeline includes preprocessing to prevent inconsistency between training and inference
- Evaluation performed only on held-out test set 

---

## 10. Deployment Architecture

Model training was performed offline due to dataset licensing constraints. Training and deployment are intentionally decoupled.

The trained pipeline is serialised as `data_processing/artifacts/model.pickle`.

The Dockerised Flask application loads this artifact strictly for inference.

---

## 11. Limitations

- Dataset unavailable in repository due to licensing constraints
- Cross-sectional dataset and model design prevents modelling stream growth over time
- No marketing or external exposure data
- No historical features on artist performance
- Model performance may not generalise to newly-emerging genres or platform shifts

---

## 12. Reproducibility Note

All figures and metrics shown in notebooks were generated from the original dataset.

Notebooks are presented with outputs preserved for inspection.