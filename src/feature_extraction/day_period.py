#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that represents the day period as integer

Created on Fri Oct 29 12:57:14 2021

@author: ldankert
"""

import numpy as np
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the character-based length as a feature
class DayPeriod(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], input_column)


    # returns 4 columns, one for each daytime
    def _get_values(self, inputs):
        result = []
        for period in np.array(inputs[0]):
            if period == "Night":
                period_number = [1,0,0,0]
            elif period == "Morning":
                period_number = [0,1,0,0]
            elif period == "Afternoon":
                period_number = [0,0,1,0] 
            elif period == "Evening":
                period_number = [0,0,1,0]
            else:
                raise Exception(f"The day period is not defined, it was: {period}")
            result.append(period_number)
        result = np.array(result)
        return result
