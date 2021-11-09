#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that represents the day period as integer

Created on Fri Oct 29 12:57:14 2021

@author: ldankert
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the character-based length as a feature
class DayPeriod(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "day_period")


    # returns 4 columns, one for each daytime
    def _get_values(self, inputs):
        result = []
        for time in inputs[0]:
            time = int(time.split(":")[0])
            if time < 6:
                result.append(0)
            elif time < 12:
                result.append(1)
            elif time < 18:
                result.append(2)
            elif time <= 24:
                result.append(3)
        enc = OneHotEncoder(sparse = False)
        result = np.array(result)
        result = result.reshape(len(result), 1)
        result = enc.fit_transform(result)
        return result
