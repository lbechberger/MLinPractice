#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the month in which a tweet was published as feature.

Created on Tue Oct 12 12:33:37 2021

@author: dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np
import datetime


class MonthExtractor(FeatureExtractor):
    """Extracts the month in which a tweet has been published from the given input column."""
    
    
    def __init__(self, input_column):
        super().__init__([input_column], "tweet_month")
        
        
    def _get_values(self, inputs):
        """Extracts the month from a string containing a date."""
        result = np.array([datetime.datetime.strptime(date, "%Y-%m-%d").month for date in inputs[0]])
        result = result.reshape(-1, 1)
        return result
        