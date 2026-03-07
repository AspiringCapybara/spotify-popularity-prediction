## 📝 Practices and Techniques Demonstrated
 
This project demonstrates the application of several core data science and software engineering practices in a practical machine learning problem:

- **Pipeline Design and Problem Decomposition**

    - Broke down the project and problem into logical, testable components.
    - Structured the project as a simple end-to-end training pipeline from raw data to trained model artifact: data cleaning → processing → modelling → evaluation.
    - The trained pipeline was then serialised and integrated into a separate inference pipeline for deployment.

- **Modular, Maintainable Project Structure**

    - Abstracted core logic into separate functions and/or modules (external files), improving code organisation and readability.
    - Cleanly separated experimentation (notebooks and training code) from production inference logic.

- **Data Cleaning**

    - Addressed issues with data quality by removing incomplete or invalid records, preventing downstream model distortion and reducing noise.

- **Decision-Making Informed by Exploratory Data Analysis (EDA)**

    - Used EDA to identify skewed distributions and relationships between features.
    - Heavy-tailed behaviour of the target variable motivated log-normalisation and model evaluation mainly in log space (rather than raw space).
    - Retained preprocessing steps (scaling, encoding) to preserve flexibility for future model experimentation.
    - Chose Random Forest model due to non-linear relationships observed between predictors and target variable.

- **Model Evaluation Beyond Headline Metrics**

    - Evaluated model performance using relative error analysis across popularity quantiles, predicted vs actual plots (log-log scale), and metrics including MAE, RMSE and R2.
    - Identified systematic under-prediction in the upper tail that reflected unobserved external factors driving popularity.
    - Explicitly prioritised interpretability of results and diagnostic insights over maximising model accuracy, given the limits of feature-based prediction in cultural markets.

- **Diagnostic Analysis and Model Interpretation**

    - Used grouped permutation importance and conducted an ablation experiment to explicitly test model dependences on the playlist exposure feature.
    - Confirmed that playlist exposure dominates intrinsic audio features and release timing, strengthening conclusions from error analysis.

- **Full-stack integration of ML inference into a web app**

    - Built a simple, functional and interactive front-end using HTML, JavaScript, and CSS, with some Bootstrap components for improved aesthetics and Jinja2 templating to handle repetitive elements. 
    - Used `Flask` to develop the back-end web application, including server-side validation of user inputs. 

- **Reproducibility and Deployment Awareness (Docker)**

    - Containerised the inference application using Docker to ensure reproducible inference behaviour and simplified setup for users and reviewers evaluating the project.
    - Included dependencies, preprocessing logic and model loading within one deployable unit.
    - The original dataset is intentionally excluded due to licensing restrictions and the serialised `model.pickle` enables the application to be run independently of the training environment.

---

[Return to main README](../README.md)
