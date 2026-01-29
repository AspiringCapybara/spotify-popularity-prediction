## 📂 What Each File Does

This section briefly explains what selected non-Markdown files (i.e., the more important ones) in the project do.

For clarity, this section will be grouped by the two main areas of the project:

1️⃣ [Data Processing and Model Building](#📊-data-processing-and-model-building)

2️⃣ [Web App](#🕸️-web-app)

---

### 📊 Data Processing and Model Building

This part of the project prepares the data and builds a model that can make predictions about Spotify songs. If you're not familiar with coding, think of this as the "behind the scenes" work that makes the web app smart.

---

**cleaning.py**

This script handles loading and basic cleaning of the data.

*Technical details of the code in this script are explored in greater detail in the Optional Technical Deep Dive section under the [Appendix B](README.md#appendix-b) of the main README.*

---

**popular_spotify_songs.csv**

This is the dataset used in the project, downloaded from Kaggle. It contains metadata about popular Spotify tracks.

---

**preprocessing.py**

This script is responsible for preparing the dataset for training. It includes:
- choosing which pieces of information (or 'features') from the data will help the model make better predictions
- looking at patterns or trends in the data to understand it better
- converting the data into a format that works well for the machine learning model

*Technical details of the code in this script are explored in greater detail in the Optional Technical Deep Dive section under the [Appendix B](README.md#appendix-b) of the main README.*

---

**model.pickle**

The trained machine learning model (random forest regressor) that was saved for use in the web application (i.e., for loading in `web_app/app.py`).

---

**model.py**

This script helps a machine learning model make predictions based on past data. It takes cleaned data, teaches the model using part of that data (called the training set), and then checks how well the model works using new, unseen data (called the test set).

The goal is to see how accurately the model can make future predictions.

---

### 🕸️ Web App

This part of the project builds the web application, which is mainly the interface where users enter their data to get predictions from the machine learning model.

If you're not familiar with coding, think of the web application like a counter at a fast food restaurant. The user (customer) places an order (inputs data), and the kitchen (the model) quickly prepares and delivers the food (the result).

---

**.flaskenv**

This hidden file automatically tells Flask:

- which file to run (`FLASK_APP`)
- to enable debug mode (`FLASK_DEBUG=1`)

Thus, there is no need to set environment variables manually whenever the code needs to be run.

---

**app.py**

This script handles routing for the web application in the back-end. The trained random forest machine learning model that was saved in `data_processing/model.py` is unloaded here, allowing it to be used without needing to train it again.

---

**validation.py**

This script performs server-side validation of user-submitted data by checking for the following:

- null or empty values
- numeric values of incorrect data type (non-integer)
- numeric values that are out of range
- categorical values that fall outside the provided options

For each detected error, an exception is raised and a specific error message is displayed to the user.

The machine learning model runs only after all inputs pass validation when the form is submitted through the web app.

---

**input_preprocessing.py**

This script converts the data entered by the user into a format that the machine learning model can work with. For each variable, the data is converted using the same steps as taken in `data_processing/preprocessing.py`.

---

**templates/index.html**

HTML code for the user interface of the web app, containing the instructions for the user and the form that the user should submit to make a prediction.

---

**templates/layout.html**

Base template containing some boilerplate code that defines the shared HTML structure of the website.

Individual pages of the website inherit the template by extending it using Jinja2 syntax, removing the need to keep repeating the same HTML when creating new pages for the website.

---

**templates/rangeslider_js.html**

This script contains the JavaScript code, which is defined using a macro, used to make the range sliders (in the webpage) dynamic.

Defining using a macro allows the chunk of code to be reused multiple times in `index.html` for each of the many range sliders in the webpage (just like how a function can be called many times once it is defined).

---

**static/styles.css**

Stylesheet for `index.html` containing the CSS code that controls the appearance and layout of the web app’s user interface.

---

[Return to main README](/project/README.md)
