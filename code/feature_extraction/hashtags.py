# -*- coding: utf-8 -*-
"""
Creates binary column showing whether tweet uses hastag or not.

Created on Fri Oct 22 01:44:28 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class Hashtags(FeatureExtractor):
    """Determines if there is Hastag in a Tweet."""
    
    
    def __init__(self, input_column):
        """Constructor, calls super Constructor."""
        super().__init__([input_column], "{0}_binary".format(input_column))
        
        
    # don't need to fit, so don't overwrite _set_variables()
        
    
    def _get_values(self, inputs):
        """
        Appends array with 0 if no hashtag and with 1 if there is a hashtag.
        Therefore returns binary column.
        """
        result = np.array([0 if len(_) <= 2 else 1 for _ in inputs[0]])
        result = result.reshape(-1, 1)
        
        return result