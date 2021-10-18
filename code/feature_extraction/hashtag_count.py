#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of hashtags in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""

import numpy as np
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class HashtagCount(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_hashtag_count".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        hashtags_list = inputs[0].astype(str).values.tolist()

        values = []
        for row in hashtags_list:
            if ast.literal_eval(row) == []:
                values.append(0)
            else:
                values.append(len(ast.literal_eval(row)))
                
        result = np.array(values)
        result = result.reshape(-1,1)
        
        return result
