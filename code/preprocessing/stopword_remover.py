#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove common stopwords from the tweet.

Created on Thu Oct  7 12:21:12 2021

@author: dhesenkamp
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ast


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
        stops = set(stopwords.words('english'))
        
        for tweet in inputs[0]:
            # problem with literal_eval? works fine in jupyter but raises error on terminal exec
            tweet_eval = ast.literal_eval(tweet)
            column = str([_ for _ in tweet_eval if _ not in stops])
        
        return column
    