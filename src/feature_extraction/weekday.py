"""
Feature that defines the weekday the tweet was made

Created: 04.11.21, 11:35

Author: LDankert
"""

import numpy as np
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the weekday
class Weekday(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], input_column)


    # returns a 7 column array, one for each weekday
    def _get_values(self, inputs):
        result = []
        for weekday in np.array(inputs[0]):
            if weekday == "Monday":
                result.append([1,0,0,0,0,0,0])
            elif weekday == "Tuesday":
                result.append([0,1,0,0,0,0,0])
            elif weekday == "Wednesday":
                result.append([0,0,1,0,0,0,0])
            elif weekday == "Thursday":
                result.append([0,0,0,1,0,0,0])
            elif weekday == "Friday":
                result.append([0,0,0,0,1,0,0])
            elif weekday == "Saturday":
                result.append([0,0,0,0,0,1,0])
            elif weekday == "Sunday":
                result.append([0,0,0,0,0,0,1])
            else:
                raise Exception (f"The input does not fit a weekday, the input was: {weekday}")
        result = np.array(result)
        return result
