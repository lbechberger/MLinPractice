# -*- coding: utf-8 -*-
"""
Extracts time from a tweet and one-hot encodes it

Created on Sat Oct 23 17:51:46 2021

@author: Yannik
modified by dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


class Daytime(FeatureExtractor):
    
    
    def __init__(self, input_column):
        super().__init__([input_column], "day_{0}".format(input_column))
        
        
    def _get_values(self, inputs):
        
        daytime = []
        for i in inputs[0]:
            time = i.split(":")
            hour = int(time[0])
            
            # night hours
            if hour in range(0, 4):
                daytime.append(0)
            
            # morning hours
            if hour in range(5, 9):
                daytime.append(1)
                
            # midday
            if hour in range(10, 14):
                daytime.append(2)
                
            # afternoon
            if hour in range(15, 18):
                daytime.append(3)
                
            # evening hours
            if hour in range(19, 24):
                daytime.append(4)
        
        result = np.array(pd.get_dummies(daytime))
        
        return result