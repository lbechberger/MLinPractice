#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes stopwords.
@author: louiskhub
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords


class Stopword_remover(Preprocessor):
    """Removes very common words (= stopwords)."""

    def __init__(self, input_col, output_col):
        """Initialize the Stopword_remover with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # no need to implement _set_variables
    
    def _get_values(self, inputs):
        """Remove the stopwords."""
        
        filtered_tokens = []
        stops = set(stopwords.words('english'))

        for w in inputs[0]:
            if w not in stops:
                filtered_tokens.append(w)

        return filtered_tokens
