#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer for tweets

Created on Wed Oct  6 17:26:39 2021

@author: dhesenkamp
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.tokenize import sent_tokenize, word_tokenize


class Tokenizer(Preprocessor):
    """Take a tweet (str) and tokenize it, i.e. return a str with the words of the tweet."""
    
    def __init__(self, input_column, output_column):
        """Init the Tokenizer with the given input/output columns."""
        super().__init__([input_column], output_column)
    
    # no _set_variables() function needed
    
    def _get_values(self, inputs):
        """Tokenize the tweet.
        
        Args:
            inputs -- list of input columns
        """
        tokenized = []
        
        for tweet in inputs[0]:
            sentences = sent_tokenize(tweet)
            tokenized_tweet = []
            for sentence in sentences:
                words = word_tokenize(sentence)
                tokenized_tweet += words
            tokenized.append(str(tokenized_tweet))
            
        return tokenized
