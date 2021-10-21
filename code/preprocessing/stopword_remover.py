#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove common stopwords from the tweet.

Created on Thu Oct  7 12:21:12 2021

@author: dhesenkamp
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
import ast


class StopwordRemover(Preprocessor):
    """Remove common stopwords from the given input column"""
    
    
    def __init__(self, input_column, output_column):
        """Init StopwordRemover with given input-/output columns."""
        super().__init__([input_column], output_column)
        
        
    # def _set_variables(self, inputs):
    
    
    def _get_values(self, inputs):
        """Remove stopwords from given column."""
        # keeps raising invalid syntax error on terminal, works as supposed in jupyter
        stops = set(stopwords.words('english'))
        # add user defined stopwords
        # courtesy to https://www.kaggle.com/docxian/data-science-tweets
        stops = stops.union(set(['https', 't', 'co', '&amp;', 'amp']))
        
        for tweet in inputs[0]:
            tweet_eval = ast.literal_eval(tweet)
            column = str([_ for _ in tweet_eval if _ not in stops])
        
        return column
    