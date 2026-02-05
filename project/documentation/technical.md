## Technical Design Details

This file documents key modelling and preprocessing decisions that influenced model behaviour, evaluation and extensibility.

### Data Ingestion

The raw dataset required defensively handling text encoding to ensure it can be reproducibly loaded in different environments. A fallback encoding strategy was used to prevent the pipeline from failing due to inconsistencies in Unicode.

### Feature Selection and Noise Reduction

Features were dropped due to near-zero variance or a lack of observable relationship with the target variable. This reduced noise and subsequently simplified model interpretation.

### Handling of Skewed Distributions

A few continuous predictor variables exhibited significant right skew. They were thus log-normalised to stabilise variance and reduce the influence of extreme values on model splits.

### Categorical Encoding of Predictors

Categorical variables were one-hot encoded with a category dropped to avoid perfect multicollinearity as a result of redundant representations (i.e., the "dummy variable trap"). Although multicollinearity is less problematic for tree-based models that split data based on threshold values rather than feature magnitudes, this choices reduces feature dimensionality and training load.

### Diagnosing Multicollinearity

Correlation analysis and VIF were used as sanity checks to confirm that no features displayed extreme collinearity. This ensured a more stable representation of feature importance.

### Scaling and Model Flexibility

Although Random Forests do not require feature scaling, MinMax scaling was retained so the model remains compatible with alternative models explored earlier and for potential future extensions of the project.

### KIV - Model evaluation, Jupyter notebooks, fuse in concepts appleid

---

[Return to main README](/project/README.md)
