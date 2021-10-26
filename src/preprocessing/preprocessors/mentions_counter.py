#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that counts the amount of mentions 
mentions array column -> length of array column
"""

from ast import literal_eval
from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_MENTIONS, COLUMN_MENTIONS_COUNT

class MentionsCounter(Preprocessor):

    # constructor
    def __init__(self):
        # input column "tweet", new output column
        super().__init__([COLUMN_MENTIONS], COLUMN_MENTIONS_COUNT)

    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # parses array string as array, then counts length of array
        column = inputs[0].apply(literal_eval).str.len()
        return column 
