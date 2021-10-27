#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove common stopwords from the tweet.

Created on Thu Oct  7 12:21:12 2021

@author: dhesenkamp
"""

from code.preprocessing.preprocessor import Preprocessor
import gensim


class StopwordRemover(Preprocessor):
    """Remove common stopwords from the given input column"""
    
    
    def __init__(self, input_column, output_column):
        """Init StopwordRemover with given input-/output columns."""
        super().__init__([input_column], output_column)
        
        
    # def _set_variables(self, inputs):
    
    
    def _get_values(self, inputs):
        """Remove stopwords from given column."""
        column = [gensim.parsing.preprocessing.remove_stopwords(tweet) for tweet in inputs[0]]
        
        return column
    