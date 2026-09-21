## 📝 Practices and Techniques Demonstrated
 
This project demonstrates the application of several core data science and software engineering practices in a practical machine learning problem:

- **Pipeline Design and Problem Decomposition**

    - Broke down the project and problem into logical, testable components.
    - Structured the project as a simple end-to-end training pipeline from raw data to trained model artifact: data cleaning → processing → modelling → evaluation.
    - The trained pipeline was then serialised and integrated into a separate inference workflow for deployment.

- **Modular, Maintainable Project Structure**

    - Abstracted core logic into separate functions and/or modules (external files), improving code organisation and readability.
    - Separated experimentation (notebooks and training code) from web application inference logic.

- **Data Cleaning**

    - Addressed issues with data quality by removing incomplete or invalid records before modelling.
    - Retained an observation with unusually low stream count where its validity could not be independently disproven, rather than removing it solely because it produced an extreme model error.

- **Decision-Making Informed by Exploratory Data Analysis (EDA)**

    - Used EDA to identify skewed distributions and relationships between features.
    - Heavy-tailed behaviour of the target variable motivated a log transformation of stream count and greater emphasis on evaluating multiplicative prediction errors.
    - Retained preprocessing steps (scaling, encoding) within a reproducible pipeline to preserve flexibility for future model experimentation.
    - Chose a Random Forest model to capture non-linear relationships observed between predictors and the target variable.

- **Model Evaluation Beyond Headline Metrics**

    - Evaluated model performance using MAE, RMSE and R2 alongside residual analysis, predicted-vs-actual plots and error analysis across stream count quartiles.
    - Examined raw-scale relative error and identified its sensitivity to unusually small actual values, where a single extreme overprediction can dominate the group mean.
    - Used median absolute log error as a complementary, more robust measure of typical multiplicative prediction error across popularity ranges. 
    - Compared raw-scale absolute error across popularity quartiles, showing that similar multiplicative errors can correspond to much larger absolute errors for highly streamed songs.
    - Identified upper-tail compression in model predictions, indicating difficulty in reproducing some of the most extreme stream counts using the available features alone.
    - Explicitly prioritised interpretability, diagnostic analysis and understanding model limitations over maximising model accuracy.

- **Diagnostic Analysis and Model Interpretation**

    - Used grouped permutation importance and an ablation experiment to examine and explicitly test the model's dependence on playlist exposure.
    - Found that playlist exposure contributes substantially more to predictive performance than individual audio characteristics or release timing.
    - Investigated extreme prediction failures rather than automatically removing inconvenient observations, distinguishing potential data anomalies from confirmed data errors.

- **Full-stack integration of ML inference into a web app**

    - Built a simple, functional and interactive front-end using HTML, JavaScript, and CSS, with some Bootstrap components for improved aesthetics and Jinja2 templating to handle repetitive elements. 
    - Used `Flask` to develop the back-end web application, including server-side validation of user inputs. 

- **Reproducibility and Deployment Awareness (Docker)**

    - Containerised the inference application using Docker to ensure reproducible inference behaviour and simplified setup for users and reviewers evaluating the project.
    - Included dependencies, preprocessing logic and model loading within one deployable unit.
    - The original dataset is intentionally excluded due to licensing restrictions, while the serialised `model.pickle` enables the application to run independently of the training dataset.

---

[Return to main README](../README.md)
