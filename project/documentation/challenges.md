### Key Challenges I Faced
---
**1. Very poor initial model performance**

Initially, I used a Linear Regression model, but the R² value was very low (~0.04), indicating that it explained almost none of the variance in the target variable ('streams').

At first, I suspected that this was due to a scale mismatch between the predictor variables and dependent (i.e., target) variable ('streams').  The predictor variables were either log-normalised or scaled to between 0 and 1 using `MinMaxScaler`, but the 'streams' values remained in the millions or billions. To address this, I log-normalised the 'streams' variable using `NumPy`, but the R<sup>2</sup> value surprisingly became even lower (~ 0.0247).

After consulting ChatGPT and analysing the data further, I realised the relationship between the features and the target was likely non-linear. I then switched to a `RandomForestRegressor`, which is more suitable for capturing complex patterns. However, the R<sup>2</sup> only improved marginally to 0.03438.

I later realised that the problem was likely due to the features, rather than the model itself. The project initially only focused on making predictions on Spotify streams based on audio characteristics alone, which proved to be too narrow a scope. I had excluded variables like "released_year" and "released_month", assuming they were not predictive. Reintroducing them significantly improved the performance, as seen from R<sup>2</sup> being increased to 0.376. This suggests that temporal factors meaningfully affect the number of streams a song gets.

---

**2. Model performance could still be better**

While there was a tremendous improvement from the initial model, I felt that the model had the potential to be better. Thus, I experimented with using 200 and 300 decision trees in the random forest model, instead of the initial 100.

This marginally improved the R<sup>2</sup> by less than 0.01. I also tested different combinations of hyperparameters using GridSearchCV to find the best combination (i.e., the one that gives the highest R<sup>2</sup>).

The best combination gave an R<sup>2</sup> of about 0.77 on the training data, suggesting a good fit on the training data, though it did not generalise well to unseen data. The best combination of hyperparameters did not significantly boost the R<sup>2</sup> on test data.

Following this, I shifted my focus back to feature selection. I used 100 decision trees in the random forest, but this time I included the columns "in_spotify_playlists" and "in_spotify_charts" that I had previously dropped. Once again, the model’s performance was significantly boosted, with R<sup>2</sup> rising to 0.5352.

Interestingly, while trying to resolve a separate challenge faced (detailed below under _Spotify API Restrictions_), I accidentally discovered that dropping the "in_spotify_charts" column boosted the R<sup>2</sup> further to 0.5557. This suggests that the column may have introduced noise or irrelevant variance, potentially distracting the model and reducing its predictive accuracy.

Considering the small size of the data set (which makes it prone to overfitting, which happens when the model captures noise in the training data instead of patterns that generalise well to new data) and its potential for noise, I am satisfied with the model achieving this level of performance.

Capturing more than half of the variance in 'streams' is a reasonable result for a real-world music prediction project with limited data. This is especially true given the number of unknown external factors that influence a song’s popularity beyond the features in the dataset (e.g., social media, publicity, fame of artist(s)).

---

**3. Spotify API restrictions and implementation changes**

The original version of this project used Spotify’s API to retrieve song metadata in the back-end, allowing the user to simply enter a valid song name. The app would then predict the number of streams that song might receive using a random forest model.

However, during development, I discovered that Spotify’s API explicitly prohibits the use of its data in any AI or machine learning applications, including educational projects. This made the original implementation incompatible with their terms of service.

To work around this, I explored alternative data sources (such as the Million Song Dataset) to extract similar metadata.

Unfortunately, none of the datasets I found included a crucial variable used in my model: in_spotify_playlists (i.e., the number of Spotify playlists a song is featured in). As an option, I considered removing this variable, but doing so significantly reduced the model's performance, dropping the R<sup>2</sup> score from 0.5557 to 0.3791.

Given this limitation, I redesigned the app to have users manually input metadata (e.g., danceability, energy, duration, playlist count, etc.). This approach not only avoids any API violations but also enables predictions for hypothetical songs, making the app more flexible and exploratory.

---

**4. ModuleNotFoundError when running web app using `flask run`**

At one point, the web app suddenly refused to start when I used the usual `flask run` command. Flask kept throwing a `ModuleNotFoundError` and no port was forwarded, even though the files and import statements were not changed. After some investigation, I realised the issue was caused by how Python was handling the relative imports across the different directories in my project.

After exploring several troubleshooting options (e.g., changing `from web_app.validation import server_side_validation` to `from project.web_app.validation import server_side_validation`, adding empty `__init__.py` files to explicitly tell Python to treat each folder as a regular package, etc.), including suggestions from ChatGPT, the solution that ultimately worked was running the app using:

```bash
cd project
python -m flask run
```

I learnt that this was because `python -m flask run` runs Flask as a Python module, so Python first sets up the import paths correctly based on the current directory and the project's folder structure. The import paths may not always be correctly set up the same way when `flask run` is used instead.

---

[Return to main README](/project/README.md)
