#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:22:30 2021

@author: dhesenkamp
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np


class Mentions(FeatureExtractor):
    
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_mentions".format(input_column))
    
    
    def _get_values(self, inputs):
        
        result = np.array([0 if len(_) <= 2 else 1 for _ in inputs[0]])
        result = result.reshape(-1, 1)
        
        return result
