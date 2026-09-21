# Model Development and Evaluation Report

## 1. Objective

To model Spotify stream counts using structured audio features, release timing and playlist exposure metadata, with emphasis on:

- Understanding limitations of predicting stream counts
- Performance evaluation across popularity ranges
- Identifying dominant features driving prediction

This document focuses strictly on modeling methodology and evaluation.

---

## 2. Data Preprocessing Pipeline

Feature preprocessing was implemented using a unified `Scikit-learn` Pipeline to ensure consistency between training and inference. Data cleaning and target preparation were performed before the pipeline was fit.

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

These selected numerical features were scaled using `MinMaxScaler`.

- `bpm`
- `speechiness_%_log`
- `liveness_%_log`
- `acousticness_%_log`
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

A custom `LogFeatureTransformer` was implemented to apply log transformations to selected skewed percentage features within the pipeline.

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

The dataset was split into 70% training and 30% testing.

Data-dependent preprocessing steps within the Scikit-learn pipeline were fit using the training data only, helping prevent leakage from the held-out test set.

Evaluation metrics:

- R2
- MAE
- RMSE

Diagnostic analysis included:

- Residual analysis in log space
- Predicted vs actual raw stream counts displayed on log-log axes
- Mean raw-scale relative error across stream-count quartiles
- Median absolute log error across stream-count quartiles
- Raw-scale absolute error across stream-count quartiles
- Inspection of extreme prediction errors

---

## 6. Performance Summary

[Predicted vs Actual (raw scale)](visualisations/predicted_vs_actual_raw.png)

Final test set performance:

- R2 ≈ 0.56
- MAE (log scale) ≈ 0.41
- RMSE (log scale) ≈ 0.88

Median absolute log error is relatively similar across stream-count quartiles, suggesting that typical multiplicative prediction accuracy is broadly comparable across popularity ranges. However, raw absolute prediction errors become substantially larger for highly streamed songs, and isolated extreme prediction failures remain present.

An R2 of approximately 0.56 on the held-out test set indicates that the available audio features, release metadata and playlist exposure capture meaningful predictive signal in log-transformed stream counts, while leaving substantial variation unexplained.

---

## 7. Error Behaviour Across Popularity Levels

[Raw Absolute Error by Quantile](visualisations/raw_abs_error_by_quantile.png)

Quantile-based analysis was performed using several complementary error measures.

Raw-scale mean relative error was found to be highly sensitive to an unusually low-stream observation with only 2762 actual streams. Because percentage-based errors divide by the actual value, this single extreme overprediction substantially inflated the mean relative error of the lowest-stream quartile.

The observation is unusually low for a dataset of popular Spotify songs and may represent a data anomaly. However, without independent evidence confirming that the value is erroneous, it was retained rather than removed solely because of its magnitude.

Median absolute log error was therefore used as a complementary measure of typical prediction performance. This metric was relatively similar across the four popularity quartiles, suggesting that typical multiplicative error does not vary dramatically with popularity.

In contrast, raw absolute error generally increases with stream count. Highly streamed songs therefore tend to exhibit much larger errors in absolute numbers of streams, even when their multiplicative errors are comparable to those of less-popular songs.

The predicted vs actual plot in Section 6 also shows upper-tail compression for some of the most highly streamed songs, indicating difficulty in reproducing certain extreme outcomes using the available predictors alone.

---

## 8. Feature Contribution Analysis

[Permutation Feature Importance](visualisations/permut_feature_importance.png)

Permutation feature importance indicates that:

- Playlist exposure is the dominant predictive feature
- Audio features provide incremental but smaller contributions
- Removing playlist exposure feature substantially degrades R2

The ablation experiment confirms that, within this model, the remaining audio and release-timing features still explain a meaningful portion of variance, but playlist exposure contributes substantially more predictive power than the individual audio and release-timing features examined.

---

## 9. Reproducibility and Evaluation Stability

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
- No direct marketing, promotional-spend or artist-popularity variables beyond playlist exposure metadata
- No historical features on artist performance
- Model performance may not generalise to newly-emerging genres or platform shifts

---

## 12. Reproducibility Note

All figures and metrics shown in notebooks were generated from the original dataset.

Notebooks are presented with outputs preserved for inspection.

---

## 13. Technical Appendix

The complete experimental analysis that includes, in greater detail, residual diagnostics and additional evaluation plots is available in:

[model_evaluation.ipynb](../data_processing/notebooks/model_evaluation.ipynb)

This notebook contains the full model evaluation workflow and preserved outputs for transparency.

---

[Return to main README](../README.md)