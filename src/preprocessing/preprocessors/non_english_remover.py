import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NonEnglishRemover(BaseEstimator, TransformerMixin):

    def fit(self, df):
        # there is nothing to fit
        return self

    def transform(self, X_df: pd.DataFrame):        
        df = X_df[X_df.language == 'en']
        return df

