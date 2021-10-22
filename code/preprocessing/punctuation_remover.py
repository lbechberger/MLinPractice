#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.

Created on Wed Sep 29 09:45:56 2021

@author: lbechberger/rfarah
"""

import string
from code.preprocessing.preprocessor import Preprocessor


class PunctuationRemover(Preprocessor):
    """Removes punctuation from the tweet and turn contractions into a whole word"""

    def __init__(self, input_column, output_column):
        """Initialize the PunctuationRemover with the given input and output column."""
        super().__init__([input_column], output_column)

    def _set_variables(self, inputs):
        """Store punctuation for later reference."""
        self._punctuation = [x for x in string.punctuation[1:-1]]

    def _get_values(self, inputs):
        """Replace punctuation with empty string."""

        output = []
        for tweet in inputs[0]:
            column = [word for word in tweet if word not in self._punctuation]
            output.append(column)

        return output
