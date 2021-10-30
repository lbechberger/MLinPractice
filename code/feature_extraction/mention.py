#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determines whether a Tweet has a mentions.

Created on Thu Oct 21 10:22:30 2021

@author: dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class Mentions(FeatureExtractor):
    """Determines whether a Tweet has mentions and returnes it in a binary form."""
    
    
    def __init__(self, input_column):
        """Constructor, calls super Constructor."""
        super().__init__([input_column], "{0}_binary".format(input_column))
        
        
    # don't need to fit, so don't overwrite _set_variables()
    
    
    def _get_values(self, inputs):
        """
        Appends array with 0 if no mentions and with 1 if there is a mention.
        Therefore returns binary column.
        """
        result = np.array([0 if len(_) <= 2 else 1 for _ in inputs[0]])
        result = result.reshape(-1, 1)
        
        return result