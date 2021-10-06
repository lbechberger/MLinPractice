#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes stopwords.
@author: louiskhub
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords


class Stopword_remover(Preprocessor):
    """Removes stopwords."""

    def __init__(self, input_col, output_col):
        """Initialize the Stopword_remover with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # no need to implement _set_variables
    
    def _get_values(self, inputs):
        """Remove the stopwords."""
        
        stops = set(stopwords.words('english'))
        
        if inputs[0] not in stops:
            column = inputs[0]

        return column