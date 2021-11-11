#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove stopwords from the tweet.

Created on Tue Nov  9 19:16:37 2021

@author: chbroecker
"""

from src.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords

class StopwordRemover(Preprocessor):
    """Removes the stopwords in the given column"""
    
    def __init__(self, input_column, output_column):
        """Initialize the StopwordRemover with the given input and output column."""
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Remove the stopwords, inputs have to be lowercase and tokenized"""

        print("\tStopwordRemover")
        stop_words = set(stopwords.words('english'))
        
        filtered_tweets = []
        for tweet_tokenized in inputs[0]:
            filtered_tweets.append([word for word in tweet_tokenized if not word in stop_words])
        
        return filtered_tweets
