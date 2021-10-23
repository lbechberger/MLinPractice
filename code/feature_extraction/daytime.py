# -*- coding: utf-8 -*-
"""
Extracts time from a Tweet and one-hot encodes it

Created on Sat Oct 23 17:51:46 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Daytime(FeatureExtractor):
    
    def __init__(self, input_column):
        super().__init__([input_column], "day_{0}".format(input_column))
        
    def _get_values(self, inputs):
        
        
        result = []
        for i in inputs[0]:
            time = i.split(":")
            hour = int(time[0])
            
            if hour in range(0, 7):
                result.append(0)
            if hour in range(8, 15):
                result.append(1)
            if hour in range(16, 23):
                result.append(2)
        
        #one-hot encoding
        result = np.array(result)
        #result = result.reshape(-1, 1)
        result = OneHotEncoder(sparse = False).fit_transform(result)
        result = result.reshape(-1,1)
        return result
        