# Predicting Spotify Song Streams using Machine Learning
#### Video Demo: https://youtu.be/ydEIjLxPrCY
#### Description:

This is a Flask-based web application that uses supervised machine learning to estimate how many streams a song might receive on Spotify based on its metadata. Users can input details like release year and data on audio features of songs, then click "Predict" to get an instant stream count estimate.

This app gives users a fast and interactive way to explore how certain song features might impact streaming performance. It would be useful for music producers, data enthusiasts, or anyone curious about trends in digital music.

## 🧠 How The App Works

- The machine learning model used is a Random Forest Regressor trained on the "Most Streamed Spotify Songs of 2023" dataset from Kaggle.

- The data was cleaned, explored, and preprocessed before training.

- Model performance was evaluated using the R<sup>2</sup> score metric and optimised (with GridSearchCV) for better accuracy.

- When a user submits the form, their input is preprocessed and passed to the model, which returns a predicted stream count displayed on the web interface.

- After the first run, a `model.pickle` file will be created automatically in the project directory. This file stores the trained model and is used for subsequent predictions.

## ⚙️ Setup (How to Use the Project)

Clone the repository by running this code in the terminal:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Note:

In the code above:

- your-username → your GitHub username
- your-repo-name → the repository name you want to clone

_Also, make sure that you have Git installed. Download Git if you don’t have it installed yet._

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

[Please click here](files.md) for a brief description of what each file in the project (excluding Markdown files) does.

_(Note: You don’t need to understand the code in detail to use the project. Technical notes are available for those who are curious in the [Appendices](#appendices).)_

## 📈Possible Extensions / Improvements

This project can be extended and/or improved in the following ways:

**Enhanced Dataset**

- Incorporating a larger and more diverse dataset with additional predictor variables could improve model accuracy and generalisability.

**Additional Evaluation Metrics**

- Beyond R<sup>2</sup>, metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) can provide a more comprehensive assessment of model performance.

**Alternative Models**

- Other machine learning models, such as XGBoost, can be trained and compared to the current Random Forest model. Evaluating multiple models using consistent metrics may help identify the best-performing approach.

**API Integration**

- The web app can be extended to integrate with APIs beyond Spotify, allowing users to search for a song and automatically retrieve metadata to feed into the prediction model.

---

### 📝Concepts Applied from CS50x

This project builds on several key concepts introduced in CS50x:

- **Computational Thinking**

Structured the project as a simple pipeline: data cleaning → processing → modeling → deployment. During the early stages of this project, (preparing the data and building the model), I broke down the problem into logical, testable components by separating the entire program into smaller files to work on them one at a time. After the model was finalised, I focused on building and testing my web app.

- **Learn and Applying Unfamiliar Libraries** (by reading documentation)

Referred to official documentation of various libraries to guide me on how to use certain classes that I was not very familiar with.

- **Code Abstraction**

Some portions of code were abstracted into separate functions and/or modules (external files), which are then imported as needed. This approach improves code organisation, readability, and maintainability.

- **Python Programming**

The core application is written in Python, using libraries such as `Pandas`, `NumPy`, `Scikit-learn` and `Flask`.

- **SQL**

Used SQL queries for exploratory data analysis in DataFrames using `pandasql`.

- **HTML, Jinja2, JavaScript & CSS** (including Bootstrap)

A simple but functional and interactive front-end was built using HTML, JavaScript, and CSS, with some Bootstrap components for improved aesthetics. `Flask` was used to develop the back-end web application. These technologies were introduced in CS50x’s Lectures 8 and 9.

Jinja2 templating was used within the HTML files to handle repetitive elements, helping keep the code cleaner, more maintainable, and better organised.

---

## Appendices



### Appendix A

#### Some Key Challenges Faced During the Project (and how I overcame them)

[Please click here to read](challenges.md) (for those who are interested)

---

### Appendix B

#### Deep Dives into Technical Details

[Please click here to read](technical.md) (for those who are interested)

---

### Appendix C

The following may provide greater insights into the context of this project:

- [Motivation & Background](motivation.md) — Why I chose to take CS50x and my computing background before taking the course

- [Acknowledgements](acknowledgements.md) — The help I obtained throughout this project and the external sources that I used

---

## 📚 Quick Reference: Additional Documentation

Some of these documents have already been linked throughout the README, but here’s a handy list in one place for easy access:

- [Challenges Faced](challenges.md)
- [Technical Details](technical.md)
- [Motivation & Background](motivation.md)
- [Acknowledgements](acknowledgements.md)
- [File Descriptions](files.md)
