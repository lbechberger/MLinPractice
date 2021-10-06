#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize the tweet into individual words.

Created on Wed Oct  6 13:59:54 2021

@author: lbechberger
"""

from code.preprocessing.preprocessor import Preprocessor
import nltk

class Tokenizer(Preprocessor):
    """Tokenizes the given input column into individual words."""
    
    def __init__(self, input_column, output_column):
        """Initialize the Tokenizer with the given input and output column."""
        super().__init__([input_column], output_column)
    
    