#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that counts the amount of mentions 
mentions array column -> length of array column
"""

from ast import literal_eval
from src.preprocessing.preprocessor import Preprocessor
from src.util import COLUMN_MENTIONS, COLUMN_MENTIONS_COUNT

class MentionsCounter(Preprocessor):
    
    # constructor
    def __init__(self):
        # input column "tweet", new output column
        super().__init__([COLUMN_MENTIONS], COLUMN_MENTIONS_COUNT)
        
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        column = inputs[0].str.len()
        return column