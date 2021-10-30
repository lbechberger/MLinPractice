#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect bigrams from a Tweet, not finished!

Created on Thu Oct  7 14:53:52 2021

@author: lbechberger
"""

import ast
import nltk
from code.feature_extraction.feature_extractor import FeatureExtractor


class BigramFeature(FeatureExtractor):
    """Collects bigrams."""
    
    
    def __init__(self, input_column):
        """Constructor, calls super Constructor."""
        super().__init__([input_column], "{0}_bigrams".format(input_column))
    
    
    # don't need to fit, so don't overwrite _set_variables()
    
    
    def _set_variables(self, inputs):
        """"""
        overall_text = []
        for line in inputs:
            tokens = ast.literal_eval(line.item())
            overall_text += tokens
        
        self._bigrams = nltk.bigrams(overall_text)