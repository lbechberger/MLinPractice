# -*- coding: utf-8 -*-
"""
Extracts number of replies from a tweet.

Created on Fri Oct 22 00:58:01 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np

class RepliesExtractor(FeatureExtractor):
    """Collects the number of replies for a Tweet and stores them as seperate feature."""
    
    def __init__(self, input_column):
        """Constructor, calls super Constructor."""
        super().__init__([input_column], "{0}_feature".format(input_column))
        
        
    # don't need to fit, so don't overwrite _set_variables()
        
        
    def _get_values(self, inputs):
        """Returnes the given input column as a feature."""
        result = np.array(inputs[0])
        result = result.reshape(-1,1)
        
        return result