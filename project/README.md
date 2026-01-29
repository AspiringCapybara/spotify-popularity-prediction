# Predicting Spotify Song Streams

## Project Overview

This project explores a deceptively simple question: 

<p style="text-align: center;"><strong>To what extent can a song's Spotify stream count be predicted from its metadata and audio features alone?</strong></p>

It features a full-stack data-driven web application that estimates a song's Spotify stream count using supervised machine learning. Beyond generating predictions, this project investigates the strengths and limitations of feature-based prediction in cultural markets, where outcomes are considerably influenced by external factors that are often hard to quantify (e.g., marketing, branding, artist popularity).

## What Makes This Problem Interesting

Spotify stream counts follow a highly-skewed distribution and are influenced by many variables that are difficult to fully capture in typical audio data sets. As such, predicting stream counts is a challenging yet realistic modelling task.

This project aims to:

<ul>
    <li>examine how well-structured metadata can explain a song's popularity</li>
    <li>analyse where and why predictions may fail</li>
    <li>highlight the limitations of purely data-driven approaches in artistic fields</li>
</ul>

The project prioritises interpretability, error analysis and honest evaluation over optimising for maximal accuracy or creating the ideal predictive model.

## Project Evolution

The original version of this project was built as my CS50x final project. Since then, I substantially enhanced it with deeper error analysis, improved evaluation, clearer model interpretation and containerisation (Docker) to align with production-oriented workflows.

## High-Level Modelling Approach 

- A Random Forest Regressor was trained on the "Most Streamed Spotify Songs of 2023" dataset from Kaggle.
- Data were cleaned and exploratory data analysis performed.
- Log-normalisation was performed on the target variable (stream count) to better model the heavy-tailed distribution and stabilise variance.
- Audio features and metadata were selected, encoded and scaled through a reproducible preprocessing pipeline.
- The model's hyperparameters were tuned using `GridSearchCV`.

## Evaluation Strategy

KIV - to be completed

## Key Insights and Limitations

KIV - to be completed

## Application Architecture

This project is implemented as a Flask-based web application where:

<ul>
    <li>Users input song metadata into the form on the interactive webpage.</li>
    <li>Inputs are validated server-side and processed using the same preprocessing pipeline used during training.</li>
    <li>The trained ML model returns a predicted Spotify stream count for the song, which is displayed on the browser.</li>
</ul>

UI elements (e.g., sliders, collapsible sections) are implemented using Bootstrap, JavaScript and Jinja2 macros to improve the application's usability.


## ⚙️ Setup and Running the App

Clone the repository by running this code in the terminal:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

This project uses a `requirements.txt` file to manage Python dependencies. Next, to install all the required packages, run this command in the terminal:
```bash
pip install -r requirements.txt
```

To run the web app, run this command in the terminal:
```bash
cd project
python -m flask run
```

## 🗃️ Full Project Structure

KIV - to be updated at the end of revamping the project

```
project/
├── data_processing/
│   ├── __init__.py
│   ├── cleaning.py
│   ├── model.py
│   ├── popular_spotify_songs.csv
│   └── preprocessing.py
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
├── __init__.py
├── .flaskenv
├── acknowledgements.md
├── challenges.md
├── corr_heatmap.png
├── files.md
├── motivation.md
├── README.md
├── requirements.txt
├── scatter_plot_matrix.png
└── technical.md
```

## 🔧 Technologies Used

**Front-end:**

- HTML with Jinja2 for dynamic templates
- CSS via Bootstrap for responsive styling
- JavaScript for client-side interactivity

**Back-end:**

- Flask framework for routing and rendering (`request`, `render_template`)

**Data Cleaning / Preprocessing:**

- `Pandas` and `NumPy` for data manipulation
- `matplotlib` and `seaborn` for exploratory data visualisation
- `pandasql` for running SQL queries on DataFrames
- `statsmodels` for statistical checks (e.g., multicollinearity using `variance_inflation_factor`)
- `scikit-learn` for preprocessing techniques like `OneHotEncoder` and `MinMaxScaler`

**Machine Learning:**

- `scikit-learn`’s `RandomForestRegressor` for building the prediction model

**Saving and Loading Model:**

- `pickle` was used to save the trained model for it to be loaded without requiring retraining.

**Data Source:**

- Kaggle dataset: *Most Streamed Spotify Songs 2023*

## 📂 What Each File Does

[Please click here](documentation/files.md) for a brief description of what each file in the project (excluding Markdown files) does.

_(Note: You don’t need to understand the code in detail to use the project. Technical notes are available for those who are curious in the [Appendices](#appendices).)_

---

## Appendices

### Appendix A

#### Some Key Challenges Faced During the Project (and how I overcame them)

[Please click here to read](documentation/challenges.md) (for those who are interested)

---

### Appendix B

#### Deep Dives into Technical Details

[Please click here to read](documentation/technical.md) (for those who are interested)

---

### Appendix C

The following may provide greater insights into the context of this project:

- [Motivation & Background](documentation/motivation.md) — Why I chose to take CS50x and my computing background before taking the course

- [Concepts Applied from CS50x](documentation/cs50x_concepts.md) - How I applied concepts learnt in CS50x to this project

- [Acknowledgements](documentation/acknowledgements.md) — The help I obtained throughout this project and the external sources that I used

---

## 📚 Quick Reference: Additional Documentation

Some of these documents have already been linked throughout the README, but here’s a handy list in one place for easy access:

- [Challenges Faced](documentation/challenges.md)
- [Technical Details](documentation/technical.md)
- [Motivation & Background](documentation/motivation.md)
- [Concepts Applied from CS50x](documentation/cs50x_concepts.md)
- [Acknowledgements](documentation/acknowledgements.md)
- [File Descriptions](documentation/files.md)
