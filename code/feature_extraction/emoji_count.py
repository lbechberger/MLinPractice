#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that tells whether photos are present or not.
Created on Wed Sep 29 12:29:25 2021
@author: maximilian
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the photo-bool as a feature
class EmojiCount(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_photo_bool".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # 0 if no photos, return 1 else
    def _get_values(self, inputs):
        
        column = inputs[0].str
        
        results = []
        values = 0
        
        for tweet in column.split():
            for word in tweet:
                if str(word.encode('unicode-escape').decode('ASCII')).__contains__('\\'):
                    values =+ 1 
                    
            results.append(values)
                
        result = np.array(values)
        result = result.reshape(-1,1)
        
        return result
