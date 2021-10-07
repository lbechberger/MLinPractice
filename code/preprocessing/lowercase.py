#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that lowercases the original tweet text.

@author: marcelklehr
"""

from code.preprocessing.preprocessor import Preprocessor


class Lowercase(Preprocessor):

    # constructor
    def __init__(self, input_column, output_column):
        # input column "tweet", new output column
        super().__init__([input_column], output_column)

    # don't implement _set_variables()

    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # lowercase column
        column = inputs[0].str.lower()
        return column