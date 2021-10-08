#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes stopwords.
@author: louiskhub
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
import ast


class Stopword_remover(Preprocessor):
    """Removes very common words (= stopwords)."""

    def __init__(self, input_col, output_col):
        """Initialize the Stopword_remover with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # no need to implement _set_variables
    
    def _get_values(self, inputs):
        """Remove the stopwords."""
        
        filtered_col = []
        stops = set(stopwords.words('english'))
        stops.update(["'s", "\"", "\'", "“", "”", "´", "`"])
        
        for row in inputs[0]:
            filtered_row = []
            for w in row:
                if w not in stops:
                    filtered_row.append(w)
            filtered_col.append(filtered_row)

        return filtered_col
