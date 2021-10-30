# -*- coding: utf-8 -*-
"""
Feature extraction for retweets.

Created on Thu Oct 21 22:39:28 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class RetweetExtractor(FeatureExtractor):
    """Collects the number of retweets for a Tweet and stores them as seperate feature."""

    
    def __init__(self, input_column):
        """Constructor, calls super Constructor."""
        super().__init__([input_column], "{0}_feature".format(input_column))
        
        
    # don't need to fit, so don't overwrite _set_variables()
        
        
    def _get_values(self, inputs):
        """Returns the input column as a seperate feature."""
        result = np.array(inputs[0])
        result = result.reshape(-1,1)
        
        return result