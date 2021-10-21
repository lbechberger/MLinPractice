# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:39:28 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np

class RetweetExtractor(FeatureExtractor):
    """Collects the number of retweets for a Tweet and stores them as seperate feature"""
    
    def __init__(self, input_column):
        super.__init__(input_column, "feature_{0}".format[input_column])
        
    def _get_values(self, inputs):
        result = np.array(inputs[0])
        return result.reshape(-1,1)