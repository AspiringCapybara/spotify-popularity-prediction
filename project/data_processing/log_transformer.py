import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.epsilon = 1e-8

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Log-normalise selected features
        X['speechiness_%_log'] = np.log(X['speechiness_%'] + self.epsilon)
        X['liveness_%_log'] = np.log(X['liveness_%'] + self.epsilon)
        X['acousticness_%_log'] = np.log(X['acousticness_%'] + self.epsilon)

        X = X.drop(["speechiness_%", "liveness_%", "acousticness_%"], axis=1)

        return X
