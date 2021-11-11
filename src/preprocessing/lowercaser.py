#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transforms everything in lowercase

Created on Thu Nov 09 19:19:51 2021

@author: ldankert
"""

from src.preprocessing.preprocessor import Preprocessor
import nltk

class Lowercaser(Preprocessor):
    """Lowercases every word in the input string"""
    
    def __init__(self, input_column, output_column):
        """Initialize the Lowercaser with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Lowercase the tweet."""
        print("\tLowercaser")
        lowercased = []
        for tweet in inputs[0]:
            lowercased.append(tweet.lower())

        return lowercased
