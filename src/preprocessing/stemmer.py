#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stemming the inputed words

Created on Thu Nov 09 19:50:23 2021

@author: ldankert
"""

from src.preprocessing.preprocessor import Preprocessor
from nltk.stem import PorterStemmer

class Stemmer(Preprocessor):
    """Stemmes every word in the input string"""
    
    def __init__(self, input_column, output_column):
        """Initialize the Stemmer with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Stemmes the tweet."""
        print("\tStemmer")
        stemmed = []
        ps = PorterStemmer()
        for tweet in inputs[0]:
            stemmed.append([ps.stem(word) for word in tweet])
        return stemmed
