# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:01:56 2021

@author: Yannik
modified by dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class Likes(FeatureExtractor):
    """Collects the number of likes for a tweet and stores them as seperate feature"""
    
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_feature".format(input_column))
        
    def _get_values(self, inputs):
        
        result = np.array(inputs[0])
        result = result.reshape(-1, 1)

        return result
