# 🎧 Predicting Spotify Song Streams

## 📝 Project Overview

This project explores a deceptively simple question: 

<p style="text-align: center;"><strong>To what extent can a song's Spotify stream count be predicted from its metadata and audio features alone?</strong></p>

It features a full-stack data-driven web application that estimates a song's Spotify stream count using supervised machine learning. Beyond generating predictions, this project investigates the strengths and limitations of feature-based prediction in cultural markets, where outcomes are considerably influenced by external factors that are often hard to quantify (e.g., marketing, branding, artist popularity).

## 😮 What Makes This Problem Interesting

Spotify stream counts follow a highly-skewed distribution and are influenced by many variables that are difficult to fully capture in typical audio data sets. As such, predicting stream counts is a challenging yet realistic modelling task.

This project aims to:

<ul>
    <li>examine how well-structured metadata can explain a song's popularity</li>
    <li>analyse where and why predictions may fail</li>
    <li>highlight the limitations of purely data-driven approaches in artistic fields</li>
</ul>

The project prioritises interpretability, error analysis and rigorous evaluation over optimising for maximal accuracy or creating the ideal predictive model.

## 🙂 Who This Project Is For

This project is intended for readers interested in applied data science, machine learning evaluation and the limitations of feature-based predictive modelling in real-world cultural markets.

It is particularly relevant for those interested in modelling heavy-tailed outcomes and evaluating models beyond headline metrics.

## 🌱 Project Evolution

The original version of this project was built as my CS50x final project. Since then, I substantially extended and enhanced it with deeper error analysis, improved evaluation, clearer model interpretation and containerisation (Docker) to better align with production-oriented workflows. The project structure was also refactored specifically for portfolio presentation.

## 🔑 Key Takeaway (TL;DR)

**Streaming popularity is driven far more by exposure than by audio characteristics or release timing. As a result, the biggest hits are hard to predict without data on promotion, artist popularity and other external factors.**

### ⚠️ Dataset Note

The original dataset used for analysis is not included in this repository due to licensing restrictions. The analysis notebooks assume the presence of `data_processing/popular_spotify_songs.csv` and are presented with pre-generated outputs for review.

All figures and results shown in this repository were generated using the original dataset.

---

## 🧠 High-Level Modelling Approach 

- A Random Forest Regressor was trained on the "Most Streamed Spotify Songs of 2023" dataset from Kaggle.
- Data were cleaned and exploratory data analysis performed.
- Log-normalisation was performed on the target variable (stream count) to better model the heavy-tailed distribution and stabilise variance.
- Audio features and metadata were selected, encoded and scaled through a reproducible preprocessing pipeline.
- The model's hyperparameters were tuned using `GridSearchCV`.

## 📊📈 Evaluation Strategy

Model performance is not assessed using a single metric alone. On top of R2, this project highlights:
- residual analysis across different ranges of popularity,
- interpretation of prediction errors on a log scale,
- and inspection of where the model consistently under-predicts or over-predicts stream counts.

This approach reflects real-world data science practice, in which understanding *where* and *how* a model can fail often offers deeper insights than purely focusing on analysing headline metrics.

## 🔍 Summary of Key Insights

The following findings illustrate how the conclusion is reached from the data and evaluation results.

- **Playlist exposure is the dominant driver of a song's streaming volume.**

Permutation importance and ablation analysis show that removing the playlist-related feature causes a large drop in R2 and a sharp increase in error. This demonstrates that the exposure a song has gained dominates intrinsic audio characteristics and release timing when explaining a song's stream count.

- **The model performs best for songs with mid-range stream counts.**

Residual analysis and absolute error analysis by stream count quantiles show relatively low errors for moderately-popular songs, whereas model performance degrades for extreme outliers.

- **Highly popular songs are systematically under-predicted.**

The predicted vs actual plot (log-log scale) illustrates upper-tail compression of the model's predictions, reflecting the influence of external factors (e.g., marketing, branding, artist popularity) that are not captured in the dataset.

- **Overall model performance is realistic for stream count prediction.**

An R2 of about 0.55, together with reasonable MAE and RMSE, demonstrate that the model captured meaningful signals. 

## 🤔 Limitations and Future Work

- Incorporating external variables such as artist popularity
- Comparing predictions across time periods to study changes over time
- Displaying confidence intervals of models to users rather than point estimates of stream count alone
- Evaluating alternative loss functions to reduce upper-tail compression in stream count predictions

## 🖥️ Application Architecture

This project is implemented as a Flask-based web application where:

<ul>
    <li>Users input song metadata into the form on the interactive webpage.</li>
    <li>Inputs are validated server-side and processed using the same preprocessing pipeline used during training.</li>
    <li>The trained ML model returns a predicted Spotify stream count for the song, which is displayed on the browser.</li>
</ul>

UI elements (e.g., sliders, collapsible sections) are implemented using Bootstrap, JavaScript and Jinja2 macros to improve the application's usability.

## ⚙️ Setup and Running the App (Docker)

This project includes a Docker configuration to ensure that the web app runs consistently across different environments.

### 1. Clone the repository

In the terminal, run:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Build the Docker Image

From the project root (i.e., where the Dockerfile is located), run:

```bash
docker build -t spotify-stream-predictor .
```

### 3. Run the container

Then, once the image is built, run:

```bash
docker run -p 5000:5000 spotify-stream-predictor
```

Once the container is running, open your browser and visit this page, which is where the web app will be available:

http://localhost:5000


### (Optional) Running Without Docker

This project uses a `requirements.txt` file to manage Python dependencies. To install all the required packages and run the web app, run from the project root directory:
```bash
pip install -r requirements.txt
python -m web_app.app
```

## 🗃️ Full Project Structure

```
project/
├── data_processing/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── eda.ipynb
│   ├── model_evaluation.ipynb
│   ├── model.pickle                # Pre-trained model used for inference in the web app
│   └── model.py
├── documentation/
│   ├── visualisations/
│   │   ├── abs_error_by_quantile.png
│   │   ├── corr_heatmap.png
│   │   ├── permut_feature_importance.png
│   │   ├── predicted_vs_actual.png
│   │   ├── residuals_vs_predictions.png
│   │   └── scatter_plot_matrix.png
│   ├── acknowledgements.md
│   └── practices.md
├── web_app/
│   ├── static/
│   │   ├── music.png
│   │   └── styles.css
│   ├── templates/
│   │   ├── index.html
│   │   ├── layout.html
│   │   └── rangeslider_js.html
│   ├── __init__.py
│   ├── app.py
│   ├── input_preprocessing.py
│   └── validation.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

*Note: For simplicity in this portfolio project, data loading, preprocessing, training and inference logic share modules within `data_processing/`. These concerns would typically be separated in a production system.*

## 🔧 Technologies Used

### Environment and Deployment

- Docker for containerised, reproducible deployment of the Flask web application

### Front-end

- HTML with Jinja2 for dynamic server-side templates
- CSS via Bootstrap for responsive layout and styling
- JavaScript for client-side interactivity

### Back-end

- Flask framework for routing, request handling and template rendering
    - `request`
    - `render_template`
- Python for inference and data flow between different components

### Data Analysis and Preprocessing

The following are used mainly in Jupyter Notebooks for exploratory data analysis (EDA), feature engineering and model evaluation:

- `pandas` and `numpy` for data manipulation and feature engineering
- `matplotlib` and `seaborn` for EDA and diagnostics
- `pandasql` for running SQL-style queries on DataFrames during EDA
- `statsmodels` for statistical checks (e.g., multicollinearity using `variance_inflation_factor`)
- `scikit-learn` preprocessing tools
    - `OneHotEncoder`
    - `MinMaxScaler`

### Machine Learning

- `scikit-learn`
    - `RandomForestRegressor` for building the prediction model
    - `train_test_split` for partitioning the dataset

### Model Evaluation

- Jupyter Notebooks for:
    - Evaluation metrics analysis
    - Visualising predicted vs actual stream counts
    - Error analysis by stream count quantiles
    - Feature importance analysis
    - Ablation experiment

### Model Persistence

- `pickle` was used to save the trained model so it can be loaded without requiring retraining.

### Data Source

- Kaggle dataset: *Most Streamed Spotify Songs 2023*

---

## 📚 Appendices

Appendices are optional and not required to understand the core project.

- [Practices and Techniques Demonstrated](documentation/practices.md) — Core data science and software engineering practices demonstrated in this project

- [Acknowledgements](documentation/acknowledgements.md) — The help I obtained throughout this project and the external sources that I used

---