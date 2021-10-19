#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column (e.g tweet column).
"""

import numpy as np
from src.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class CharacterLength(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_charlength".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        result = np.array(inputs[0].str.len())
        result = result.reshape(-1,1)
        return result
