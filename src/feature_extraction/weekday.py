"""
Feature that defines the weekday the tweet was made

Created: 04.11.21, 11:35

Author: LDankert
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the weekday
class Weekday(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "weekday")

    # returns a 7 column array, one for each weekday
    def _get_values(self, inputs):
        inputs = pd.to_datetime(inputs[0]).dt.dayofweek.to_list()
        enc = OneHotEncoder(sparse = False)
        result = np.asarray(inputs)
        result = result.reshape(len(result), 1)
        result = enc.fit_transform(result)
        return result
