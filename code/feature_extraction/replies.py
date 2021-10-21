# -*- coding: utf-8 -*-
"""
Extracts number of replies from a tweet

Created on Fri Oct 22 00:58:01 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np

class RepliestExtractor(FeatureExtractor):
    """Collects the number of replies for a Tweet and stores them as seperate feature"""
    
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_feature".format(input_column))
        
        
    def _get_values(self, inputs):
        result = np.array(inputs[0])
        return result.reshape(-1,1)