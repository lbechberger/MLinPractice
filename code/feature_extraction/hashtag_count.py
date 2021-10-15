#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of hashtags in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class HashtagCount(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_hashtag_count".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        column = inputs[0].str
        column = [' '.join([word for word in tweet if word.__contains__('#')]) for tweet in column.split()]
        
        count = int(column.count('#'))

        return count
