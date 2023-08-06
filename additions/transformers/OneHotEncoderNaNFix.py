from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class OneHotEncoderNaNFix(TransformerMixin, BaseEstimator):
    def __init__(self, one_hot_indices) -> None:
        """
        replaces one hot encoded columns that were ignored (all zero) with NaN
        :param one_hot_indices: The beginning indices of the one hot encoded columns
                                as well as the beginning indice of the column after
                                the last encoded categorical column
                                e.g. for [a, b, c] with 3, 2, 2 unique values
                                      OHC() -> [a0, a1, a2, b0, b1, c0, c1]
                                      thus indices = [0, 3, 5, 7]
        """
        self.one_hot_indices = one_hot_indices
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for i, lower in enumerate(self.one_hot_indices[:-1]):
            upper = self.one_hot_indices[i+1]
            cats = upper - lower
            no_vals = (~X[:, lower:upper].any(axis=1)).repeat(cats).reshape((X.shape[0], cats))
            #print(no_vals.shape)
            #print(X[:, lower:upper].shape)
            #print(X[0, lower:upper])
            np.putmask(X[:, lower:upper], no_vals, np.nan)
        return X