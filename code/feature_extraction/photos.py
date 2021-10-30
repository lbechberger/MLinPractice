#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature extraction for photos

Created on Thu Oct 21 09:44:54 2021

@author: dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class Photos(FeatureExtractor):
    """Determines whether a Tweet contains any photos."""
    
    
    def __init__(self, input_column):
        """Constuctor, calls super Constructor."""
        super().__init__([input_column], "{0}_binary".format(input_column))
      
        
    # don't need to fit, so don't overwrite _set_variables()
      
    
    def _get_values(self, inputs):
        """
        Appends array with 0 if no photo and with 1 if there is a photo.
        Therefore returns binary column.
        """
        result = np.array([0 if len(_) <= 2 else 1 for _ in inputs[0]])
        result = result.reshape(-1, 1)
        
        return result