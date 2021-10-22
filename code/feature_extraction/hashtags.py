# -*- coding: utf-8 -*-
"""
Creates binary column showing wheter tweet uses hastag or not

Created on Fri Oct 22 01:44:28 2021

@author: Yannik
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class Hashtags(FeatureExtractor):
    
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_binary".format(input_column))
        
    
    def _get_values(self, inputs):
         
        result = np.array([0 if len(_) <= 2 else 1 for _ in inputs[0]])
        result = result.reshape(-1, 1)
        
        return result