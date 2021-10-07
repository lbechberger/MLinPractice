#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove common stopwords from the tweet.

Created on Thu Oct  7 12:21:12 2021

@author: dhesenkamp
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords


class StopwordRemover(Preprocessor):
    """Remove common stopwords from the given input column"""
    
    
    def __init__(self, input_column, output_column):
        """Init StopwordRemover with given input-/output columns."""
        super().__init__([input_column], output_column)
        
        
    def _set_variables(self, inputs):
        """Store stopwords for later reference"""
        self._stopwords = stopwords.words("english")
    
    
    def _get_values(self, inputs):
        """Remove stopwords from given column."""
        # code itself works fine, problem seems to be with accessing the tweet
        stopwords_removed = []
        
        for word in inputs[0]:
            if not word.lower() in self._stopwords:
                stopwords_removed.append(word)
        
        #stopwords_removed = [w for w in inputs[0] if not w.lower() in self._stopwords]
        #column = inputs[0].str.replace(self._stopwords, "")
        return stopwords_removed
    