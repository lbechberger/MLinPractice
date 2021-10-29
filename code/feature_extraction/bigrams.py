#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:53:52 2021
<<<<<<< HEAD
=======

>>>>>>> a7c7fdb (unit test and TDD example)
@author: ml
"""

import ast
import nltk
from code.feature_extraction.feature_extractor import FeatureExtractor

<<<<<<< HEAD

class BigramFeature(FeatureExtractor):

    def __init__(self, input_column):
        super().__init__([input_column], "{0}_bigrams".format(input_column))

    def _set_variables(self, inputs):

=======
class BigramFeature(FeatureExtractor):
    
    
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_bigrams".format(input_column))
    
    def _set_variables(self, inputs):
        
>>>>>>> a7c7fdb (unit test and TDD example)
        overall_text = []
        for line in inputs:
            tokens = ast.literal_eval(line.item())
            overall_text += tokens
<<<<<<< HEAD

        self._bigrams = nltk.bigrams(overall_text)
=======
        
        self._bigrams = nltk.bigrams(overall_text)
>>>>>>> a7c7fdb (unit test and TDD example)
