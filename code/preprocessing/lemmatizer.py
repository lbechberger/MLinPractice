#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that reduces words to their lemmas.
@author: louiskhub
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import WordNetLemmatizer


class Lemmatizer(Preprocessor):
    """Reduces words to their lemmas."""

    def __init__(self, input_col, output_col):
        """Initialize the Lemmatizer with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # no need to implement _set_variables
    
    def _get_values(self, inputs):
        """Lemmatize the words."""
        
        lemmatizer = WordNetLemmatizer()
        column = lemmatizer.lemmatize(inputs[0])
        
        return column