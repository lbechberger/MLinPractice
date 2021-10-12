import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, cols):

        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, df):
        # there is nothing to fit
        return self

    def transform(self, X_df: pd.DataFrame):
        df = X_df.drop(columns=self.cols)
        return df

